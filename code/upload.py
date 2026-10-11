"""Analyze a video file and save shin-angle overlays and per-frame measurements."""
import argparse
import csv
import math
from pathlib import Path


def shin_angle(knee, ankle):
    """Return 0–90 degrees relative to image horizontal, or None for coincident joints."""
    dx, dy = knee[0] - ankle[0], knee[1] - ankle[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def measure(result, threshold):
    """Only measure when exactly one person and confident joints are visible."""
    angles = [None, None]
    kp = result.keypoints
    if kp is None or len(kp.xy) != 1 or kp.conf is None:
        return angles, "Need exactly one visible athlete"
    xy = kp.xy[0].cpu().tolist()
    confidence = kp.conf[0].cpu().tolist()
    for side, (knee, ankle) in enumerate(((13, 15), (14, 16))):
        if all(confidence[i] >= threshold and all(math.isfinite(v) for v in xy[i])
               and tuple(xy[i]) != (0, 0) for i in (knee, ankle)):
            angles[side] = shin_angle(xy[knee], xy[ankle])
    return angles, "" if all(a is not None for a in angles) else "Some joints are hidden or uncertain"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, help="Input video; omit to open a file chooser")
    parser.add_argument("--model", default="yolo26m-pose.pt", help="Pose model name or weights path")
    parser.add_argument("--device", default=None, help="cpu, mps, or CUDA device number; default: automatic")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum joint confidence")
    parser.add_argument("--output", type=Path, help="Output .mp4; default: recordings/<input>_shin_angles.mp4")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.video is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        try:
            selected = filedialog.askopenfilename(title="Choose a video to analyze")
        finally:
            root.destroy()
        if not selected:
            return
        args.video = Path(selected)
    source = args.video.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Video does not exist: {source}")
    output = (args.output or Path(__file__).resolve().parent.parent / "recordings" /
              f"{source.stem}_shin_angles.mp4").expanduser().resolve()
    csv_path = output.with_suffix(".csv")
    if output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    if output == source or csv_path == source:
        parser.error("Output must differ from input")
    if output.exists() or csv_path.exists():
        parser.error("Output already exists; choose a different --output filename")

    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Video has no valid frame rate")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Video contains no readable frames")
        model = YOLO(args.model)
        if model.task != "pose":
            raise RuntimeError("Choose a pose model with knee and ankle keypoints")
        output.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {output}")
        options = {"imgsz": 640, "verbose": False}
        if args.device is not None:
            options["device"] = args.device
        with csv_path.open("w", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(["frame", "time_seconds", "left_shin_degrees", "right_shin_degrees"])
            count = 0
            while ok and frame is not None:
                result = model.predict(frame, **options)[0]
                angles, message = measure(result, args.confidence)
                view = result.plot()
                scale = max(0.3, min(0.7, width / 900))
                spacing = max(18, int(35 * scale / 0.7))
                lines = [f"Time: {count / fps:.2f}s"]
                for side, angle in zip(("Left", "Right"), angles):
                    value = "--" if angle is None else f"{angle:.1f} deg"
                    lines.append(f"{side} shin: {value}")
                if message:
                    lines.append(message)
                cv2.rectangle(view, (0, 0), (min(width, 440), spacing * len(lines) + 12), (20, 20, 20), -1)
                for row, line in enumerate(lines):
                    cv2.putText(view, line, (8, spacing * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(view)
                rows.writerow([count, f"{count / fps:.6f}",
                               *["" if angle is None else f"{angle:.3f}" for angle in angles]])
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} frames", flush=True)
                ok, frame = capture.read()
        print(f"Saved {count} frames: {output}\nMeasurements: {csv_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
