"""Run the same bounded MediaPipe tracker used by the skill-draft pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import cv2

from robot_skill_system.exceptions import NotConfiguredError
from robot_skill_system.perception.hand_tracking import MediaPipeHandLandmarkTracker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=4)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after this many frames; zero runs until q or Ctrl-C",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="disable the OpenCV preview window",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="keep the camera image unmirrored (recording-pipeline convention)",
    )
    parser.add_argument(
        "--print-empty-frames",
        action="store_true",
        help="also print JSON for frames where MediaPipe finds no hand",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.camera_index < 0:
        raise ValueError("camera index must be non-negative")
    if args.max_frames < 0:
        raise ValueError("max frames must be non-negative")

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        print(f"카메라 /dev/video{args.camera_index}를 열 수 없습니다.", file=sys.stderr)
        return 2

    processed_count = 0
    detected_count = 0
    tracker = MediaPipeHandLandmarkTracker()
    try:
        with tracker:
            while args.max_frames == 0 or processed_count < args.max_frames:
                success, frame_bgr = capture.read()
                if not success:
                    print("카메라 프레임을 읽을 수 없습니다.", file=sys.stderr)
                    return 3
                if not args.no_mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                evidence = tracker.track_rgb(processed_count, frame_rgb)
                processed_count += 1
                if evidence.hands:
                    detected_count += 1
                if evidence.hands or args.print_empty_frames:
                    print(evidence.model_dump_json(), flush=True)

                if not args.no_display:
                    tracker.draw_last_result(frame_bgr)
                    cv2.imshow("Dittobot MediaPipe Hands", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    except NotConfiguredError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    finally:
        capture.release()
        tracker.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(
        f"processed_frames={processed_count} detected_frames={detected_count}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
