# Libaries to Import -----------
import argparse
import math
import random
import time
from pathlib import Path
#-------------------------------

# Lists and Global Vairables ----
BODY = (0, 5, 6, 11, 12, 13, 14, 15, 16) #0=nose , 5,6 = left and right shoulder , 11,12 = left and right hips , 13,14 = left and right knees, 15,16 = left and right ankles 
WRISTS = (9, 10) # left and right wrists 
LEGS = {15: 13, 16: 14}  # ankle -> knee; used to calc shin angle later on 
NAMES = {9: "left wrist", 10: "right wrist"} # for false start detection 
STILL_RADIUS = 0.012  # 12 pixles; Maximum displacement from a reference pose before restarting the stillness timer
FALSE_RADIUS = 0.018 # 18 pixles ; Wrist displacement from its armed position that triggers a possible false start
CONTACT_RADIUS = 0.012 # 12 pixels ; Allowed ankle movement while checking a possible ground contact
SWING_DISTANCE = 0.04 # 40 pixels ; Distance an ankle must move from its previous position before it can qualify for another contact
CONTACT_HOLD = 0.08 # 80 miliseconds ; works with contact radius to determin shin angle 
#-------------------------------

# Functions --------------------
def distance(a, b):
    # find horizontal and veritcal diffrence then calculates using pythagorean therome 
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shin_angle(knee, ankle):
    """Acute angle to a horizontal ground line: vertical shin = 90 degrees."""
    # find verical and horizontal seperation , get absolute values, calcluate angle in radians, convert radians -> degrees
    return math.degrees(math.atan2(abs(knee[1] - ankle[1]), abs(knee[0] - ankle[0])))


def standing(p):
    return all(p[0][1] < p[s][1] < p[h][1] < p[k][1] < p[a][1]
               for s, h, k, a in ((5, 11, 13, 15), (6, 12, 14, 16)))


def crouched(p):
    # Broad side-view block-position heuristic: wrists below hips, torso lowered.
    hip_y = (p[11][1] + p[12][1]) / 2
    shoulder_y = (p[5][1] + p[6][1]) / 2
    ankle_y = (p[15][1] + p[16][1]) / 2
    return (all(p[i][1] > hip_y for i in WRISTS)
            and hip_y - shoulder_y < 0.6 * max(ankle_y - hip_y, 0.001))

# Classes -------------------

class Stillness:
    """Require uninterrupted stillness relative to a fixed reference pose."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.anchor = None
        self.since = None

    def ready(self, points, indices, now, duration, valid=True):
        if not valid or not all(i in points for i in indices):
            self.clear()
            return False
        if self.anchor is None or any(
            distance(points[i], self.anchor[i]) > STILL_RADIUS for i in indices
        ):
            self.anchor = {i: points[i] for i in indices}
            self.since = now
            return False
        return now - self.since >= duration


class StepTracker:
    """Estimate alternating contacts after a foot leaves its prior position.

    A contact candidate must stop near/below its starting ankle height.
    The angle is sampled at the start of that stable interval, not its end.
    This can miss contacts or mistake a paused foot for ground contact.
    """

    def __init__(self, points, direction, count):
        self.next_ankle = min(LEGS, key=lambda i: direction * points[i][0])
        self.origin = {i: points[i] for i in LEGS}
        self.ground = {i: points[i][1] for i in LEGS}
        self.count = count
        self.results = []
        self.swing = {i: False for i in LEGS}
        self.candidate = None

    def update(self, points, now):
        if len(self.results) >= self.count:
            return
        for ankle in LEGS:
            if distance(points[ankle], self.origin[ankle]) > SWING_DISTANCE:
                self.swing[ankle] = True
        ankle = self.next_ankle
        foot = points[ankle]
        if not self.swing[ankle] or foot[1] < self.ground[ankle] - 0.025:
            self.candidate = None
            return
        if self.candidate is None or distance(foot, self.candidate[0]) > CONTACT_RADIUS:
            self.candidate = (foot, now, shin_angle(points[LEGS[ankle]], foot))
            return
        if now - self.candidate[1] >= CONTACT_HOLD:
            angle = self.candidate[2]
            side = "left" if ankle == 15 else "right"
            self.results.append((side, angle))
            print(f"Estimated contact {len(self.results)}: {side}, shin {angle:.1f} degrees")
            self.origin[ankle] = foot
            self.swing[ankle] = False
            self.next_ankle = 16 if ankle == 15 else 15
            self.candidate = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    import cv2
    import pygame
    from ultralytics import YOLO

    video = None
    try:
        pygame.mixer.init()
        folder = Path(__file__).resolve().parent
        sounds = {name: pygame.mixer.Sound(str(folder / filename)) for name, filename in {
            "marks": "marks.mp3", "set": "Set.mp3", "gun": "Gun.mp3",
            "false": "FalseStart.mp3"}.items()}
        channel = pygame.mixer.Channel(0)
        model = YOLO("yolo11n-pose.pt")
        video = cv2.VideoCapture(args.camera)
        if not video.isOpened():
            raise RuntimeError("Cannot open camera. Check camera index and permissions.")

        state = "STANDING"
        gate = Stillness()
        deadline = None
        wrists = None
        steps = None
        previous_time = time.monotonic()
        previous_center = None
        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped supplying frames.")
            result = model.predict(frame, imgsz=640, verbose=False)[0]
            now = time.monotonic()
            fps = 1 / max(now - previous_time, 0.000001)
            previous_time = now
            view = result.plot()
            points = {}
            kp = result.keypoints
            # Reject multiple people instead of silently switching runner index.
            if kp is not None and len(kp.xy) == 1 and kp.conf is not None:
                xy = kp.xy[0].cpu().tolist()
                confidence = kp.conf[0].cpu().tolist()
                width = frame.shape[1]
                # Divide BOTH axes by width: distances/angles retain image aspect ratio.
                points = {i: (x / width, y / width) for i, (x, y) in enumerate(xy)
                          if confidence[i] >= 0.5 and math.isfinite(x) and math.isfinite(y)
                          and (x != 0 or y != 0)}
            required = BODY + WRISTS
            valid = all(i in points for i in required)
            message = ""
            if valid:
                center = tuple((points[11][j] + points[12][j]) / 2 for j in (0, 1))
                if previous_center is not None and distance(center, previous_center) > 0.2:
                    valid = False
                previous_center = center
            else:
                previous_center = None

            if not valid:
                gate.clear()
                message = "Keep exactly one runner and all required joints visible"
                if state in ("BLOCKS", "SETTLE", "ARMED", "RUNNING"):
                    channel.stop()
                    state = "ABORTED"
                    deadline = None
                    message = "Tracking lost: attempt aborted. Press R."
            elif state == "STANDING":
                if gate.ready(points, BODY, now, 2.0, standing(points)):
                    channel.play(sounds["marks"])
                    deadline = now + sounds["marks"].get_length()
                    state = "BLOCKS"
                    gate.clear()
            elif state == "BLOCKS":
                if gate.ready(points, required, now, 2.0,
                              now >= deadline and crouched(points)):
                    channel.play(sounds["set"])
                    # Allow the runner to rise into set before arming wrist detection.
                    deadline = now + sounds["set"].get_length() + 1.5
                    state = "SETTLE"
            elif state == "SETTLE":
                if now >= deadline:
                    wrists = {i: points[i] for i in WRISTS}
                    deadline = now + random.uniform(2.0, 3.5)
                    state = "ARMED"
            elif state == "ARMED":
                moved = max(WRISTS, key=lambda i: distance(points[i], wrists[i]))
                amount = distance(points[moved], wrists[moved])
                if amount > FALSE_RADIUS:
                    print(f"Possible false start: {NAMES[moved]}, displacement {amount:.4f}")
                    channel.play(sounds["false"])
                    state = "FALSE START"
                    deadline = None
                elif now >= deadline:
                    channel.play(sounds["gun"])
                    state = "RUNNING"
                    deadline = None
                    steps = StepTracker(points, 1 if args.direction == "right" else -1, args.steps)
            elif state == "RUNNING":
                steps.update(points, now)
                if len(steps.results) >= args.steps:
                    state = "DONE"

            lines = [f"{state} | FPS: {fps:.1f} | R reset | Q quit"]
            if message:
                lines.append(message)
            if state in ("FALSE START", "ABORTED", "DONE"):
                lines.append("Press R to begin another attempt")
            if steps:
                lines += [f"Estimated step {n}: {side} shin {angle:.1f} deg"
                          for n, (side, angle) in enumerate(steps.results, 1)]
            for row, line in enumerate(lines):
                cv2.putText(view, line, (15, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Sprint practice", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                channel.stop()
                state = "STANDING"
                gate.clear()
                deadline = wrists = steps = previous_center = None
    finally:
        if video is not None:
            video.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
