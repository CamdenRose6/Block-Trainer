# Libaries to Import -----------
import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path
#-------------------------------

# Lists and Global Vairables ----
BODY = (0, 5, 6, 11, 12, 13, 14, 15, 16) #0=nose , 5,6 = left and right shoulder , 11,12 = left and right hips , 13,14 = left and right knees, 15,16 = left and right ankles 
WRISTS = (9, 10) # left and right wrists 
LEGS = {15: 13, 16: 14}  # ankle -> knee; used to calc shin angle later on 
NAMES = {9: "left wrist", 10: "right wrist"} # for false start detection 
STILL_RADIUS = 0.012  # Fraction of frame width (12 pixels at width 1000); maximum displacement from a reference pose before restarting the stillness timer
FALSE_RADIUS = 0.018 # Fraction of frame width (18 pixels at width 1000); wrist displacement from its armed position that triggers a possible false start
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


class VideoRecorder:
    """Stream a silent, annotated 15-second clip at 30 output frames/second.

    Repeat the last available frame when inference is slower than output FPS,
    preserving playback duration without keeping the entire clip in memory.
    """
    def __init__(self, cv2, folder):
        self.cv2 = cv2
        self.folder = folder
        self.writer = None
        self.fps = 30
        self.duration = 15.0
        self.frames = 0
        self.last_frame = None

    def start(self, frame, now):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / (datetime.now().strftime("sprint_%Y%m%d_%H%M%S_%f") + ".mp4")
        height, width = frame.shape[:2]
        self.writer = self.cv2.VideoWriter(
            str(self.path), self.cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"Cannot create video: {self.path}")
        self.started = now
        self.frames = 0
        self.last_frame = None
        print(f"Recording 15 seconds to: {self.path}")

    def update(self, frame, now):
        if self.writer is None:
            return False
        elapsed = max(0.0, now - self.started)
        target = min(int(elapsed * self.fps) + 1, int(self.duration * self.fps))
        while self.frames < target:
            # Earlier output slots use the previous observation, not a future frame.
            current_slot = self.frames / self.fps
            sample = frame if self.last_frame is None or current_slot >= elapsed else self.last_frame
            self.writer.write(sample)
            self.frames += 1
        self.last_frame = frame.copy()
        if elapsed >= self.duration:
            self.close()
            return True
        return False

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.last_frame = None
            print(f"Saved video: {self.path}")


def draw_shin_angles(cv2, view, points):
    """Show each leg independently; missing/overlapping joints display --."""
    height, width = view.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.65, width / 600)
    panel_width = min(width, 280)
    cv2.rectangle(view, (0, max(0, height - 80)), (panel_width, height), (20, 20, 20), -1)
    for row, (ankle, knee) in enumerate(LEGS.items()):
        side = "Left" if ankle == 15 else "Right"
        value = "--"
        if knee in points and ankle in points and distance(points[knee], points[ankle]) > 1e-6:
            value = f"{shin_angle(points[knee], points[ankle]):.1f} deg"
        cv2.putText(view, f"{side} shin: {value}", (12, height - 48 + row * 30),
                    font, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    import cv2
    import pygame
    from ultralytics import YOLO

    video = None
    recorder = None
    try:
        pygame.mixer.init()
        folder = Path(__file__).resolve().parent
        recorder = VideoRecorder(cv2, folder / "recordings")
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
                if state in ("BLOCKS", "SETTLE", "ARMED"):
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
                    recorder.start(view, time.monotonic())
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

            lines = [f"{state} | FPS: {fps:.1f} | R reset | Q quit"]
            if message:
                lines.append(message)
            if state in ("FALSE START", "ABORTED", "DONE"):
                lines.append("Press R to begin another attempt")
            if recorder.writer is not None:
                remaining = max(0.0, recorder.duration - (time.monotonic() - recorder.started))
                lines.append(f"REC: {remaining:.1f}s remaining")
            for row, line in enumerate(lines):
                cv2.putText(view, line, (15, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
            draw_shin_angles(cv2, view, points)
            if recorder.update(view, time.monotonic()):
                channel.stop()
                state = "DONE"
                deadline = None
            cv2.imshow("Sprint practice", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                channel.stop()
                state = "STANDING"
                gate.clear()
                recorder.close()
                deadline = wrists = previous_center = None
    finally:
        if recorder is not None:
            recorder.close()
        if video is not None:
            video.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
