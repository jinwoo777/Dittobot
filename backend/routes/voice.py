import json
import os

from fastapi import APIRouter

router = APIRouter()

RUNTIME_DIR = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/.runtime")
VOICE_STATUS_PATH = os.path.join(RUNTIME_DIR, "voice_status.json")


@router.get("/status")
def get_voice_status():
    if not os.path.exists(VOICE_STATUS_PATH):
        return {"state": "unknown", "updated_at": None}
    try:
        with open(VOICE_STATUS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown", "updated_at": None}
