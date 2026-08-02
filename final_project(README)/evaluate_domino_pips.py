from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from lab8.domino_world_detector import ClassicalDominoLabelProvider, DominoWorldDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the classical domino pip detector on snapshot images and save debug outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("final_project/snapshots"),
        help="Directory containing snapshot images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final_project/pip_eval"),
        help="Directory for CSV summaries and saved debug panels.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.png",
        help="Glob for snapshot files inside --input-dir.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rgb(path: Path, image_rgb) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    debug_dir = args.output_dir / "debug_panels"
    ensure_dir(debug_dir)

    detector = DominoWorldDetector(label_provider=ClassicalDominoLabelProvider())
    rows: list[dict[str, object]] = []

    image_paths = sorted(args.input_dir.glob(args.glob))
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            rows.append(
                {
                    "image": image_path.name,
                    "domino_index": -1,
                    "face_label": "read-failed",
                    "face_confidence": 0.0,
                    "divider_found": False,
                    "half_counts": "",
                    "debug_panel": "",
                }
            )
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        observations = detector.detect(image_rgb)
        if not observations:
            rows.append(
                {
                    "image": image_path.name,
                    "domino_index": -1,
                    "face_label": "no-domino",
                    "face_confidence": 0.0,
                    "divider_found": False,
                    "half_counts": "",
                    "debug_panel": "",
                }
            )
            continue

        for idx, obs in enumerate(observations):
            debug_name = f"{image_path.stem}__domino{idx:02d}.png"
            debug_path = debug_dir / debug_name
            if obs.debug_panel is not None:
                write_rgb(debug_path, obs.debug_panel)
                debug_rel = str(debug_path.relative_to(args.output_dir))
            else:
                debug_rel = ""

            rows.append(
                {
                    "image": image_path.name,
                    "domino_index": idx,
                    "face_label": obs.face_label or "",
                    "face_confidence": float(obs.face_confidence or 0.0),
                    "divider_found": bool(obs.divider_endpoints is not None),
                    "half_counts": "" if obs.half_counts is None else str(obs.half_counts),
                    "debug_panel": debug_rel,
                }
            )

    csv_path = args.output_dir / "predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "domino_index", "face_label", "face_confidence", "divider_found", "half_counts", "debug_panel"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} prediction rows to {csv_path}")
    print(f"Saved debug panels in {debug_dir}")


if __name__ == "__main__":
    main()
