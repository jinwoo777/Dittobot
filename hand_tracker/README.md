# Hand tracker

The standalone preview and the RGB-D skill-draft pipeline use the same local
MediaPipe tracker. The tracker sends only bounded semantic evidence to the
skill-generation prompt: wrist, thumb tip, index MCP/tip, pinky MCP, and the
thumb-index midpoint. Its normalized image coordinates and MediaPipe-relative
`z` value are never treated as metric depth or robot geometry.

From the repository root:

```bash
python3 -m virtualenv .venv
.venv/bin/python -m pip install -e '.[api,vision,realsense,dev]' \
  -r hand_tracker/requirements.txt
.venv/bin/python hand_tracker/hand_camera.py --camera-index 4
```

For a bounded headless check:

```bash
.venv/bin/python hand_tracker/hand_camera.py \
  --camera-index 4 --max-frames 30 --no-display --no-mirror --print-empty-frames
```

The command prints one JSON object per selected frame. During normal RGB-D
teaching, `MVPApplication.create_recording_skill_draft()` runs this tracker on
the selected recording keyframes and includes `local_hand_tracking` in the
strict semantic prompt and persisted draft analysis parameters. Hardware
execution remains gated by the existing surface calibration, local metric TCP
trajectory, safety, and explicit hardware authorization checks.
