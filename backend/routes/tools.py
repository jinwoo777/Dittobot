from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..process_manager import manager

router = APIRouter()


class StartToolRequest(BaseModel):
    mode: str  # "voice" | "replay"
    skill_path: Optional[str] = None


@router.post("/start")
def start_tool(req: StartToolRequest):
    if manager.tool.is_running():
        raise HTTPException(409, f"이미 실행 중입니다: {manager.tool.exe} {manager.tool.args}")

    if req.mode == "voice":
        manager.start_tool("robot_replay")
    elif req.mode == "replay":
        if not req.skill_path:
            raise HTTPException(400, "replay 모드는 skill_path가 필요합니다.")
        manager.start_tool("robot_replay", [req.skill_path])
    else:
        raise HTTPException(400, "mode는 'voice' 또는 'replay'만 가능합니다.")

    return manager.tool.status()


@router.post("/stop")
def stop_tool():
    manager.stop_tool()
    return manager.tool.status()


@router.get("/status")
def tool_status():
    return {
        "tool": manager.tool.status(),
        "get_keyword": manager.get_keyword.status(),
    }
