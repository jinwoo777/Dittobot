"""
routes/coords.py

"좌표 생성" 페이지: record_trajectory를 시작/정지하고, 가장 최근 녹화 결과
(raw json은 record_trajectory.py가 저장 직후 자체적으로 smooth_trajectory/
verify_trajectory까지 이어서 실행해두므로) smooth json/그래프, verify 그래프를
한 번에 찾아서 돌려준다. 실제 파일 내용/이미지는 routes/skills.py의 기존
/api/skills/{stage}/{filename} 경로를 그대로 재사용한다 (중복 구현 안 함).
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..process_manager import manager
from .skills import SKILLS_ROOT

router = APIRouter()

RUNTIME_DIR = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/.runtime")
FRAME_PATH = os.path.join(RUNTIME_DIR, "coords_frame.jpg")


@router.post("/start")
def start_coords():
    if manager.tool.is_running():
        raise HTTPException(409, f"로봇이 이미 다른 작업 중입니다: {manager.tool.exe}")
    manager.start_tool("record_trajectory")
    return manager.tool.status()


@router.post("/stop")
def stop_coords():
    manager.stop_tool()
    return manager.tool.status()


@router.get("/status")
def coords_status():
    return manager.tool.status()


@router.get("/frame.jpg")
def coords_frame():
    if not os.path.exists(FRAME_PATH):
        raise HTTPException(404, "아직 저장된 프레임이 없습니다.")
    return FileResponse(FRAME_PATH, media_type="image/jpeg")


@router.get("/latest")
def latest_coords():
    raw_dir = os.path.join(SKILLS_ROOT, "raw")
    if not os.path.isdir(raw_dir):
        return {"raw": None}

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    if not raw_files:
        return {"raw": None}

    latest_raw = max(raw_files, key=lambda f: os.path.getmtime(os.path.join(raw_dir, f)))
    raw_base = latest_raw[:-len(".json")]  # "hammering_<ts>"

    smoothed_filename = f"{raw_base}_smoothed.json"
    compare_filename = f"{raw_base}_compare.png"
    verify_filename = f"{raw_base}_smoothed_verify.png"

    def _entry(stage, filename):
        path = os.path.join(SKILLS_ROOT, stage, filename)
        return {"stage": stage, "filename": filename, "exists": os.path.isfile(path)}

    return {
        "raw": _entry("raw", latest_raw),
        "smooth_json": _entry("smooth", smoothed_filename),
        "compare_png": _entry("smooth", compare_filename),
        "verify_png": _entry("verify", verify_filename),
    }
