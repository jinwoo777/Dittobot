import json
import os
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

SKILLS_ROOT = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
STAGES = ["raw", "smooth", "verify", "classify", "generate"]
_TIMESTAMP_RE = re.compile(r"hammering_(\d+)")


def _summarize(stage: str, filename: str) -> dict:
    path = os.path.join(SKILLS_ROOT, stage, filename)
    entry = {"stage": stage, "filename": filename, "path": path}

    if filename.endswith(".json"):
        try:
            with open(path) as f:
                data = json.load(f)
            if "skill" in data:
                entry["skill"] = data["skill"]
            if "frames" in data:
                entry["frame_count"] = len(data["frames"])
            if "segments" in data:
                entry["segment_count"] = len(data["segments"])
        except (OSError, json.JSONDecodeError):
            pass

    m = _TIMESTAMP_RE.search(filename)
    if m:
        entry["recorded_at_epoch"] = int(m.group(1))
    return entry


@router.get("")
def list_skills():
    result = {}
    for stage in STAGES:
        stage_dir = os.path.join(SKILLS_ROOT, stage)
        if not os.path.isdir(stage_dir):
            result[stage] = []
            continue
        result[stage] = [_summarize(stage, f) for f in sorted(os.listdir(stage_dir))]
    return result


@router.get("/{stage}/{filename}")
def get_skill_file(stage: str, filename: str):
    if stage not in STAGES:
        raise HTTPException(404, "알 수 없는 stage입니다.")
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "잘못된 파일명입니다.")

    path = os.path.join(SKILLS_ROOT, stage, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "파일을 찾을 수 없습니다.")

    if filename.endswith(".json"):
        with open(path) as f:
            return JSONResponse(json.load(f))
    return FileResponse(path)
