# Block Trainer
### Practice your start. Review your technique.

Block Trainer helps sprinters practice block starts independently, using movement-responsive starting commands, live shin-angle estimates, and video recordings for review.

## Why the Block Start Matters

The block start is a key part of short sprint events, from the 60m to the 200m. Practicing alone can make it difficult to recreate a race setting: imagining the commands divides your attention, while a prerecorded countdown may rush you into position.

Sprint Start responds to your movements to guide you through “On your marks,” “Set,” and the starting signal—so you can focus on your preparation and execution.

## Current Features

- **Movement-responsive commands:** Progress through the starting sequence based on your position and stillness, checking to make sure you don't jump the gun. 
- **Live shin angles:** View estimated left and right shin angles during your start.
- **Automatic recording:** Capture a 15-second video beginning at the “Set” call.
- **Video review:** Replay your start to examine your positioning and movement.

## Understanding Shin Angles

Shin angle describes the angle between the lower leg and the track. Reviewing how this angle changes through the start can help you examine your positioning during early acceleration.

The program reports an angle from **0° (horizontal)** to **90° (vertical)**. These are estimates from a two-dimensional camera image, so camera placement and joint visibility affect their accuracy. A lower angle is not automatically better; it should be considered alongside the athlete’s movement and the phase of the sprint.

For useful comparisons, keep the camera level and use a consistent side-on view.

## Video Analysis

Live feedback gives you an immediate view of your shin angles. The saved recording lets you revisit the start, pause at key moments, and compare attempts without relying on what you noticed during the sprint.

## Planned Features

- Foot and ankle stiffness analysis
- Initial ground-contact detection
- Thigh-separation measurements
- Reaction-time estimates

These features are in development and are not yet available.

## Analyze a saved video

Run the separate video analyzer from the repository folder:

```bash
python code/upload.py Small10-4.mp4
```

Omit the video path to choose a file in a dialog. Use `--model /path/to/weights.pt`
for existing YOLO pose weights, or `--device mps` to request Apple GPU processing.
The default model is `yolo26m-pose.pt`; Ultralytics may download it on first use.

The program saves a silent annotated MP4 and a CSV of per-frame left/right shin
angles under `recordings/`. It preserves the input frame rate and leaves the
original video untouched. Choose a new filename with `--output recordings/attempt2.mp4`
when repeating an analysis. Missing or uncertain joints produce blank CSV values
and `--` on screen; frames containing multiple people are not measured.
Angles are measured relative to image horizontal, so use a level, side-on camera.
