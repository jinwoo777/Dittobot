"""
app.py

ditto_system(ros2 ament_python 패키지)을 감싸는 FastAPI 백엔드.
frontend/(ditto_system 전용 신규 UI)가 이 API를 호출한다. robot_skill_system의
FastAPI 백엔드를 대체하며, 로봇 제어는 subprocess로 `ros2 run ditto_system <exe>`를
실행하는 방식이다 (process_manager.py 참고).
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .process_manager import manager
from .routes import coords, jog, pipeline, skillgen, skills, status, tools, voice

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # get_keyword(웨이크워드+STT+도구명 추출)는 robot_replay의 음성 모드가 의존하는
    # 상시 서비스라 백엔드 기동 시 같이 띄운다. 로봇 브링업(roboton)은 실제 하드웨어를
    # 움직이기 시작하는 단계라 여기서 자동 실행하지 않는다 - 사람이 직접 확인 후 켠다.
    manager.ensure_get_keyword_running()
    yield
    manager.stop_tool()
    manager.get_keyword.stop()


app = FastAPI(title="Dittobot ditto_system API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tools.router, prefix="/api/tools", tags=["tools"])
app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(status.router, prefix="/api/status", tags=["status"])
app.include_router(coords.router, prefix="/api/coords", tags=["coords"])
app.include_router(skillgen.router, prefix="/api/skillgen", tags=["skillgen"])
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
# dittobot-design/api-client.js가 이미 이 경로들(접두사 없음)로 호출하고 있어서
# 프론트를 고치지 않기 위해 그대로 맞춘다.
app.include_router(jog.router, prefix="/jog", tags=["jog"])


@app.get("/api/health")
def health():
    return {"ok": True}


# /api/* 라우터들을 먼저 등록한 뒤에 정적 프론트를 "/"에 마운트해야 경로가 안 겹친다.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
