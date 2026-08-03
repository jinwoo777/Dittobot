# Robot Skill System MVP

작업자의 RGB-D 시연을 로컬 궤적 분석과 제한된 의미 분석으로 분해하고, 검증된
`SkillGraph`를 결정론적으로 컴파일하는 Python 3.10 프로젝트입니다. 현재 실행 가능한
End-to-End 경로는 완전한 오프라인 `MockRobotAdapter`이며 로봇·카메라·OpenAI API가 없어도
Teaching → Registry → Runtime → Update 흐름을 실행할 수 있습니다. `simulation` 모드는 아직
별도 물리 시뮬레이터가 아니라 같은 Mock adapter의 별칭입니다.

> RealSense D435i, Doosan M0609, OnRobot RG2, ROS 2, MoveIt 연동은 실제 장치에서 검증하지
> 않았습니다. Doosan/RG2 adapter는 인터페이스 자리만 제공하며 gate가 닫혀 있으면 authorization
> 오류, gate가 열려도 `NotConfiguredError`로 거부합니다. 환경 플래그만으로 실제 로봇을 움직일
> 수 없습니다.

## 설치

Ubuntu 22.04 / Python 3.10 기준입니다. 시스템 패키지를 자동 변경하지 않습니다.

```bash
cd <this-repository>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,api]'
cp .env.example .env
```

Ubuntu 22.04에서 `ensurepip is not available` 또는 `python3.10-venv` 누락 오류가 나면 운영자가
시스템 정책을 확인한 뒤 다음을 직접 실행해야 합니다. 프로젝트와 설치 스크립트는 `sudo`를
자동 실행하지 않습니다.

```bash
sudo apt-get update
sudo apt-get install python3.10-venv
```

Mock 실행에는 API 키가 필요 없습니다. Live 의미 분석 또는 STT를 명시적으로 시험할 때만
`.env`의 `OPENAI_API_KEY`를 채우고 `OPENAI_MODE=live`로 바꿉니다. `.env`는 Git에서 제외됩니다.
이 프로젝트는 `.env`를 암묵적으로 읽지 않으므로 셸에서 다음처럼 로드합니다.

```bash
set -a
source .env
set +a
```

현재 검증에서는 실제 API key를 사용한 OpenAI live 호출을 수행하지 않았습니다. `.env.example`의
모델 ID는 설정 예시이며 해당 계정에서 사용할 수 있는지는 live 실행 전에 별도로 확인해야 합니다.

## Mock 실행

저장소 루트에서:

```bash
source .venv/bin/activate
robot-skill inspect
robot-skill capture-scene --backend mock --mode mock
robot-skill demo-e2e
robot-skill execute --text "파란 걸레로 오른쪽 테이블을 숙련자 방식으로 닦아줘" --mode mock
```

음성 파일 STT도 기본 mock 모드에서 네트워크 없이 실행됩니다. Mock은 같은 경로의 `.txt`
sidecar가 있으면 그 내용을 사용합니다. Audio 경로 자체는 존재해야 하며, transcription 결과를
runtime 명령으로 자동 전달하지는 않습니다.

```bash
robot-skill transcribe /path/to/command.wav
```

모듈 방식도 동일합니다.

```bash
PYTHONPATH=src python3 -m robot_skill_system.cli demo-e2e
```

보조 명령 이름도 지원합니다. `execute-skill`은 의도적으로 `--dry-run`만 허용하며, 먼저
`demo-offline` 또는 `induce-skill`로 활성 스킬을 등록해야 합니다.

```bash
PYTHONPATH=src python3 -m robot_skill_system.cli demo-offline
PYTHONPATH=src python3 -m robot_skill_system.cli capture-scene --backend mock
PYTHONPATH=src python3 -m robot_skill_system.cli validate-skill --skill wipe_surface
PYTHONPATH=src python3 -m robot_skill_system.cli execute-skill --skill wipe_surface --dry-run
```

Teaching/Update 예제:

```bash
robot-skill induce-skill --demo tests/fixtures/demonstrations/novice_wipe.json
robot-skill update-skill --skill-id wipe_surface --demo tests/fixtures/demonstrations/expert_wipe.json
robot-skill list-skills
robot-skill rollback --skill-id wipe_surface --version 1.0.0
```

실행 가능한 예제 스크립트도 모두 Mock 전용입니다.

```bash
PYTHONPATH=src python3 examples/offline_teaching_demo.py
PYTHONPATH=src python3 examples/offline_runtime_demo.py
PYTHONPATH=src python3 examples/expert_update_demo.py
```

기본 SQLite 경로는 `data/robot_skills.db`, 기본 Artifact Store root는 `data/`입니다.
`DATABASE_URL`과 `ARTIFACT_ROOT`로 변경할 수 있습니다. 현재 애플리케이션이 실제로 쓰는 주요
경로는 다음과 같습니다.

- `data/demonstrations/<session_id>/`: teaching metadata, capture/final JSON
- `data/scenes/<scene_id>.json`: SceneSnapshot JSON
- `data/skills/<skill_id>/<version>/`: graph, compiled module, validation report, manifest
- SQLite `execution_runs`/`execution_events`: 실행 상태와 구조화 이벤트

현재 CLI/API teaching 흐름은 원본 RGB, depth, audio, point cloud를 자동 녹화하지 않습니다.
그 데이터용 URI/checksum 스키마와 capture/recorder 인터페이스는 있지만 실제 recorder를 연결해야
합니다. SQLite에는 생성된 Python이나 bulk camera payload가 아니라 메타데이터와 URI/checksum이
저장됩니다.

## FastAPI

```bash
source .venv/bin/activate
uvicorn robot_skill_system.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

`/docs`에서 Teaching, Scene, Skills, Runtime API를 확인할 수 있습니다. Runtime 요청 기본 mode는
`dry_run`이지만 현재 구현에서는 이것도 MockRobot 명령을 실행·기록하는 오프라인 경로입니다.
Hardware 요청은 아래 다섯 환경 gate를 만족해도 구성된 Doosan adapter가 없어 거부됩니다.

## 검사

```bash
python3 -m compileall src tests
python3 -m pytest -q
python3 -m ruff check .
python3 -m mypy src
robot-skill demo-e2e
```

시스템 전역 pytest/plugin 버전이 충돌하는 환경에서는 프로젝트 가상환경 사용을 권장합니다.
격리 진단에는 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`를 사용할 수 있습니다.

## RealSense 경계

Core/Mock 모드는 `pyrealsense2` 없이 import됩니다. `RealSenseCapture`에는 lazy import, color-depth
alignment, burst median, timestamp-domain 보존 인터페이스가 구현되어 있고 fake module로만
테스트했습니다. D435i 실장치와 vendor SDK에서는 검증하지 않았으며 CLI/API의
`capture-scene --backend`는 현재 `mock`만 허용합니다. `RealSenseCapture.stream()`이라는 저수준
iterator는 있지만 continuous recording/Scene 갱신 파이프라인에는 연결되어 있지 않습니다.

실제 카메라를 연결하려면 대상 장비에 맞는 Intel SDK/`pyrealsense2`를 운영자가 설치하고
serial, intrinsics/extrinsics, depth scale, alignment와 clock domain을 직접 검증해야 합니다.
ROS의 `realsense2_camera` topic 또는 rosbag adapter는 이 저장소에 구현되어 있지 않습니다.

ROS 2 순서:

```bash
source /opt/ros/humble/setup.bash
source <your-ros-workspace>/install/setup.bash
```

상세 항목은 [Hardware setup](docs/HARDWARE_SETUP.md)을 참고하십시오.

## Doosan/RG2 Hardware Mode

초기 저장소와 실행 환경에는 `DSR_ROBOT2`, 검증된 Doosan ROS 2 서비스/Action, RG2 드라이버가
발견되지 않았습니다. 실제 패키지 버전, import, namespace, 함수 시그니처, TCP/load/force 단위를
셀에서 확인하고 adapter와 hardware validator를 구현한 뒤에만 다음 다섯 gate를 모두 엽니다.

```bash
export ROBOT_EXECUTION_MODE=hardware
export ENABLE_HARDWARE_EXECUTION=true
export ROBOT_BACKEND=doosan
export ENABLE_REAL_ROBOT=true
export DRY_RUN=false
```

`ENABLE_HARDWARE_EXECUTION`은 legacy enable gate이며 생략할 수 없습니다. 다섯 gate 뒤에도 연결,
E-stop, 최신 Scene, 검증된 Graph/Tool, 실제 IK·관절·자체/환경 충돌·특이점 검사, hardware-verified
workspace/scene monitor와 force supervisor가 모두 필요합니다. 현재 validator는 Mock 표시가 붙은
TCP/명시 경로 geometry만 제공하므로 hardware 승인을 만들 수 없습니다.

현재 adapter protocol의 motion/force vendor 호출은 blocking입니다. 장애물 hook과 힘 측정은 호출
전후에만 실행되며 호출 중 연속 polling이나 즉시 중단을 보장하지 않습니다. 별도의 safety-rated
monitor 또는 interruptible/streaming vendor API를 연결하기 전에는 실제 로봇에 사용하면 안 됩니다.
테스트는 실제 vendor driver나 로봇 하드웨어에 명령을 보내지 않았습니다.

## OpenAI 구현 범위

Live 경로 코드는 공식 Python SDK의 Responses structured output
(`responses.parse` + Pydantic), Audio Transcriptions, Embeddings API를 사용합니다. 모든 실제 좌표,
profile 수치, graph 검증, compilation과 실행 권한은 로컬 코드에 남습니다. Strict function-tool
descriptor와 6개 read-only 로컬 dispatcher, bounded Responses tool-call loop도 구현되어 있습니다.
Application의 live `/runtime/resolve`는 현재 registry와 Scene의 검증된 read-only snapshot을
dispatcher로 주입합니다. Demonstration analyzer와 graph composer는 strict structured output만
사용합니다. 자세한 내용은
[OpenAI integration](docs/OPENAI_INTEGRATION.md)을 참고하십시오.

## 설계 문서

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [OpenAI integration](docs/OPENAI_INTEGRATION.md)
- [Scene/Skill schema](docs/SKILL_SCHEMA.md)
- [Data schema and version rules](docs/DATA_SCHEMA.md)
- [Safety](docs/SAFETY.md)
- [Hardware setup](docs/HARDWARE_SETUP.md)
