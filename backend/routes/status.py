import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

RUNTIME_DIR = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/.runtime")
STATUS_PATH = os.path.join(RUNTIME_DIR, "status.json")
FRAME_PATH = os.path.join(RUNTIME_DIR, "frame.jpg")


@router.get("")
def get_status():
    if not os.path.exists(STATUS_PATH):
        return {
            "phase": "IDLE",
            "status_text": "robot_replay가 실행 중이 아닙니다.",
            "updated_at": None,
        }
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"phase": "UNKNOWN", "status_text": "상태 파일을 읽지 못했습니다.", "updated_at": None}


@router.get("/frame.jpg")
def get_frame():
    if not os.path.exists(FRAME_PATH):
        raise HTTPException(404, "아직 저장된 프레임이 없습니다.")
    return FileResponse(FRAME_PATH, media_type="image/jpeg")
