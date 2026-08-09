from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..process_manager import manager

router = APIRouter()


class RecordActionRequest(BaseModel):
    action: str = "start"  # "start" | "stop"


class InputPathRequest(BaseModel):
    input_path: str


@router.post("/record")
def record(req: RecordActionRequest):
    """record_trajectory는 ESC로 종료하는 대화형 창이라, API로는 시작만 트리거하고
    종료는 이 엔드포인트를 action=stop으로 다시 호출해 프로세스에 SIGINT를 보낸다."""
    if req.action == "start":
        if manager.tool.is_running():
            raise HTTPException(409, f"이미 실행 중입니다: {manager.tool.exe} {manager.tool.args}")
        manager.start_tool("record_trajectory")
    elif req.action == "stop":
        manager.stop_tool()
    else:
        raise HTTPException(400, "action은 'start' 또는 'stop'만 가능합니다.")
    return manager.tool.status()


def _run_sync(exe: str, req: InputPathRequest) -> dict:
    try:
        return manager.run_pipeline_sync(exe, [req.input_path])
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/smooth")
def smooth(req: InputPathRequest):
    return _run_sync("smooth_trajectory", req)


@router.post("/verify")
def verify(req: InputPathRequest):
    return _run_sync("verify_trajectory", req)


@router.post("/classify")
def classify(req: InputPathRequest):
    return _run_sync("classify_trajectory", req)


@router.post("/generate")
def generate(req: InputPathRequest):
    return _run_sync("generate_skill_code", req)
