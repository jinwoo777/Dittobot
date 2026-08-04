"""Deterministic transport artifacts for chronological RGB keyframes."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

MAXIMUM_TRANSPORT_FRAMES = 300
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_CELL_WIDTH_PX = 320
CONTACT_SHEET_IMAGE_HEIGHT_PX = 240
CONTACT_SHEET_LABEL_HEIGHT_PX = 24


def build_rgb_keyframe_zip(
    recording_id: str,
    keyframes: Sequence[tuple[int, Path]],
) -> bytes:
    """Create a stable ZIP archive containing selected JPEGs and a safe manifest."""

    _validate_keyframes(keyframes)
    entries = [
        {
            "sequence_number": sequence_number,
            "frame_index": frame_index,
            "archive_path": _archive_image_name(sequence_number, frame_index),
        }
        for sequence_number, (frame_index, _path) in enumerate(keyframes)
    ]
    manifest = {
        "schema_version": "1.0",
        "recording_id": recording_id,
        "frame_count": len(keyframes),
        "chronological": True,
        "frames": entries,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        _write_zip_entry(
            archive,
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for entry, (_frame_index, path) in zip(entries, keyframes, strict=True):
            _write_zip_entry(archive, str(entry["archive_path"]), path.read_bytes())
    return output.getvalue()


def build_rgbd_keyframe_zip(
    recording_id: str,
    keyframes: Sequence[tuple[int, Path, Path]],
) -> bytes:
    """Create a stable ZIP containing paired RGB/depth-preview evidence."""

    _validate_rgbd_keyframes(keyframes)
    entries = [
        {
            "sequence_number": sequence_number,
            "frame_index": frame_index,
            "rgb_archive_path": _archive_image_name(sequence_number, frame_index),
            "depth_archive_path": (
                f"depth/{sequence_number:03d}_frame_{frame_index:06d}.jpg"
            ),
        }
        for sequence_number, (frame_index, _rgb_path, _depth_path) in enumerate(keyframes)
    ]
    manifest = {
        "schema_version": "1.0",
        "recording_id": recording_id,
        "keyframe_pair_count": len(keyframes),
        "chronological": True,
        "pair_order": "rgb_then_aligned_depth_per_keyframe",
        "depth_visualization": "turbo_colormap_near_warm_invalid_black",
        "frames": entries,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        _write_zip_entry(
            archive,
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for entry, (_frame_index, rgb_path, depth_path) in zip(
            entries, keyframes, strict=True
        ):
            _write_zip_entry(
                archive,
                str(entry["rgb_archive_path"]),
                rgb_path.read_bytes(),
            )
            _write_zip_entry(
                archive,
                str(entry["depth_archive_path"]),
                depth_path.read_bytes(),
            )
    return output.getvalue()


def build_rgb_contact_sheet_pdf(keyframes: Sequence[tuple[int, Path]]) -> bytes:
    """Render chronological JPEGs as PDF pages accepted by vision-capable models."""

    _validate_keyframes(keyframes)
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:  # pragma: no cover - guarded by the api extra
        raise RuntimeError("Install the 'api' extra to enable PDF image fallback") from exc

    page_width_px = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_CELL_WIDTH_PX
    cell_height_px = CONTACT_SHEET_IMAGE_HEIGHT_PX + CONTACT_SHEET_LABEL_HEIGHT_PX
    page_height_px = CONTACT_SHEET_ROWS * cell_height_px
    frames_per_page = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_ROWS
    pages = []
    for page_start in range(0, len(keyframes), frames_per_page):
        page = Image.new("RGB", (page_width_px, page_height_px), "white")
        draw = ImageDraw.Draw(page)
        page_items = keyframes[page_start : page_start + frames_per_page]
        for offset, (frame_index, path) in enumerate(page_items):
            row, column = divmod(offset, CONTACT_SHEET_COLUMNS)
            x_px = column * CONTACT_SHEET_CELL_WIDTH_PX
            y_px = row * cell_height_px
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                fitted = ImageOps.contain(
                    rgb,
                    (CONTACT_SHEET_CELL_WIDTH_PX, CONTACT_SHEET_IMAGE_HEIGHT_PX),
                )
            image_x_px = x_px + (CONTACT_SHEET_CELL_WIDTH_PX - fitted.width) // 2
            image_y_px = y_px + (CONTACT_SHEET_IMAGE_HEIGHT_PX - fitted.height) // 2
            page.paste(fitted, (image_x_px, image_y_px))
            draw.text(
                (x_px + 6, y_px + CONTACT_SHEET_IMAGE_HEIGHT_PX + 5),
                f"sequence {page_start + offset + 1} | frame {frame_index}",
                fill="black",
            )
        pages.append(page)

    output = io.BytesIO()
    first_page, *remaining_pages = pages
    first_page.save(
        output,
        format="PDF",
        save_all=True,
        append_images=remaining_pages,
        resolution=100.0,
    )
    for page in pages:
        page.close()
    return output.getvalue()


def build_rgbd_contact_sheet_pdf(
    keyframes: Sequence[tuple[int, Path, Path]],
) -> bytes:
    """Render chronological RGB and aligned depth pairs into labeled PDF pages."""

    _validate_rgbd_keyframes(keyframes)
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:  # pragma: no cover - guarded by the api extra
        raise RuntimeError("Install the 'api' extra to enable PDF image fallback") from exc

    pairs_per_row = 2
    rows_per_page = 4
    pair_width_px = CONTACT_SHEET_CELL_WIDTH_PX * 2
    cell_height_px = CONTACT_SHEET_IMAGE_HEIGHT_PX + CONTACT_SHEET_LABEL_HEIGHT_PX
    page_width_px = pairs_per_row * pair_width_px
    page_height_px = rows_per_page * cell_height_px
    pairs_per_page = pairs_per_row * rows_per_page
    pages = []
    for page_start in range(0, len(keyframes), pairs_per_page):
        page = Image.new("RGB", (page_width_px, page_height_px), "white")
        draw = ImageDraw.Draw(page)
        page_items = keyframes[page_start : page_start + pairs_per_page]
        for offset, (frame_index, rgb_path, depth_path) in enumerate(page_items):
            row, column = divmod(offset, pairs_per_row)
            pair_x_px = column * pair_width_px
            y_px = row * cell_height_px
            for image_offset, path in enumerate((rgb_path, depth_path)):
                with Image.open(path) as source:
                    rgb = source.convert("RGB")
                    fitted = ImageOps.contain(
                        rgb,
                        (CONTACT_SHEET_CELL_WIDTH_PX, CONTACT_SHEET_IMAGE_HEIGHT_PX),
                    )
                cell_x_px = pair_x_px + image_offset * CONTACT_SHEET_CELL_WIDTH_PX
                image_x_px = (
                    cell_x_px + (CONTACT_SHEET_CELL_WIDTH_PX - fitted.width) // 2
                )
                image_y_px = y_px + (CONTACT_SHEET_IMAGE_HEIGHT_PX - fitted.height) // 2
                page.paste(fitted, (image_x_px, image_y_px))
            draw.text(
                (pair_x_px + 6, y_px + CONTACT_SHEET_IMAGE_HEIGHT_PX + 5),
                f"sequence {page_start + offset + 1} | frame {frame_index} | RGB + DEPTH",
                fill="black",
            )
        pages.append(page)

    output = io.BytesIO()
    first_page, *remaining_pages = pages
    first_page.save(
        output,
        format="PDF",
        save_all=True,
        append_images=remaining_pages,
        resolution=100.0,
    )
    for page in pages:
        page.close()
    return output.getvalue()


def _validate_keyframes(keyframes: Sequence[tuple[int, Path]]) -> None:
    if not keyframes:
        raise ValueError("at least one RGB keyframe is required")
    if len(keyframes) > MAXIMUM_TRANSPORT_FRAMES:
        raise ValueError("RGB transport is limited to 300 keyframes")
    previous_index = -1
    for frame_index, path in keyframes:
        if frame_index < 0 or frame_index <= previous_index:
            raise ValueError("RGB keyframes must have strictly increasing non-negative indices")
        if not path.is_file():
            raise ValueError("every RGB keyframe must be a local file")
        previous_index = frame_index


def _validate_rgbd_keyframes(keyframes: Sequence[tuple[int, Path, Path]]) -> None:
    if not keyframes:
        raise ValueError("at least one RGB-D keyframe pair is required")
    if len(keyframes) > MAXIMUM_TRANSPORT_FRAMES:
        raise ValueError("RGB-D transport is limited to 300 keyframe pairs")
    previous_index = -1
    for frame_index, rgb_path, depth_path in keyframes:
        if frame_index < 0 or frame_index <= previous_index:
            raise ValueError("RGB-D keyframes must have increasing non-negative indices")
        if not rgb_path.is_file() or not depth_path.is_file():
            raise ValueError("every RGB-D keyframe pair must contain local files")
        previous_index = frame_index


def _archive_image_name(sequence_number: int, frame_index: int) -> str:
    return f"rgb/{sequence_number:03d}_frame_{frame_index:06d}.jpg"


def _write_zip_entry(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    entry = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = 0o600 << 16
    archive.writestr(entry, content)


__all__ = [
    "build_rgb_contact_sheet_pdf",
    "build_rgb_keyframe_zip",
    "build_rgbd_contact_sheet_pdf",
    "build_rgbd_keyframe_zip",
]
