"""
routes/skillgen.py

"스킬 생성" 탭: classify_trajectory + generate_skill_code를 robot_replay 실행과는
독립적으로 돌려서 만들어진 스킬(move plan) 목록을 보여주고, 세그먼트별 속도/블렌드를
수정하면 바로 파일에 덮어쓴다. robot_replay.py가 실행 시점에 여기서 만든
skills/generate/<smoothed 파일명>_move_plan.json 경로를 그대로 인자로 받아서, 이 파일에
있는 좌표+실행방식 그대로 재생한다(smoothed json은 다시 안 읽음).

classify_trajectory/generate_skill_code는 rclpy/하드웨어를 안 쓰는 순수 계산+GPT 호출
모듈이라 서브프로세스 없이 이 프로세스 안에서 바로 import해서 쓴다. 단,
generate_skill_code.call_gpt()가 get_package_share_directory()를 쓰므로, 이 백엔드
프로세스 자체가 ROS2 환경(ditto_ws 포함)을 소싱한 상태로 떠 있어야 한다(실행명령어 참고).

파일 포맷 2가지:
  v1 (과거 포맷, 더 이상 새로 안 만듦): {"<segment_index>": {move_type, vel_mm_s, ...}, ...}
      좌표가 없어서 robot_replay.py가 더 이상 실행할 수 없다 - "스킬 생성"에서 다시 생성하면
      v2로 갱신된다. 조회/편집(GET/PUT)은 과거 파일 호환을 위해 계속 지원한다.
  v2 (지금은 "스킬 생성"/"순차 블록" 둘 다 이 포맷으로 저장): {"format": "v2",
      "sequence": [{move_type, vel_mm_s, ..., end_pose, via_pose}, ...]}
      좌표(yaw는 robot_replay.py의 rz 보정까지 미리 구운 최종값, generate_skill_code.
      build_v2_sequence 참고)가 통째로 들어있어 robot_replay가 재분류 없이 그대로 실행한다.
  이 파일의 조회 엔드포인트들은 둘 다 읽을 수 있어서, 아래 헬퍼들이 포맷을 감지해서
  일관된 모양(basename/segments/plan)으로 돌려준다 - "스킬 생성" 탭 프론트 코드는
  포맷 신경 안 쓰고 그대로 재사용 가능하다.
"""

import json
import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .skills import SKILLS_ROOT

router = APIRouter()

_DITTO_SYSTEM_SRC = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system")
if _DITTO_SYSTEM_SRC not in sys.path:
    sys.path.insert(0, _DITTO_SYSTEM_SRC)

from ditto_system import classify_trajectory  # noqa: E402
from ditto_system import generate_skill_code as skillgen  # noqa: E402

GENERATE_DIR = os.path.join(SKILLS_ROOT, "generate")
CLASSIFY_DIR = os.path.join(SKILLS_ROOT, "classify")


def _plan_path(basename: str) -> str:
    return os.path.join(GENERATE_DIR, f"{basename}_move_plan.json")


def _segments_path(basename: str) -> str:
    return os.path.join(CLASSIFY_DIR, f"{basename}_segments.json")


def _is_v2(raw) -> bool:
    return isinstance(raw, dict) and raw.get("format") == "v2" and isinstance(raw.get("sequence"), list)


def _items_of(raw) -> list:
    """v1/v2 상관없이 세그먼트(계획) 항목들의 리스트만 뽑는다."""
    return raw["sequence"] if _is_v2(raw) else list(raw.values())


class GenerateRequest(BaseModel):
    smoothed_path: str


@router.post("/generate")
def generate(req: GenerateRequest):
    if not os.path.isfile(req.smoothed_path):
        raise HTTPException(404, f"파일을 찾을 수 없습니다: {req.smoothed_path}")
    with open(req.smoothed_path) as f:
        frames = json.load(f)["frames"]

    basename = os.path.basename(req.smoothed_path)
    if basename.endswith(".json"):
        basename = basename[: -len(".json")]

    segments, _radius = classify_trajectory.build_segments(frames)

    prompt = skillgen.build_prompt(segments)
    gpt_error = None
    try:
        raw = skillgen.call_gpt(prompt)
    except Exception as e:
        raw = "[]"
        gpt_error = str(e)

    plan = skillgen.parse_move_plan(raw, segments)

    first_yaw = frames[0]["yaw"]
    sequence = skillgen.build_v2_sequence(segments, plan, first_yaw)

    os.makedirs(GENERATE_DIR, exist_ok=True)
    with open(_plan_path(basename), "w") as f:
        json.dump({"format": "v2", "sequence": sequence}, f, indent=2, ensure_ascii=False)

    return {
        "basename": basename,
        "segment_count": len(segments),
        "gpt_error": gpt_error,
    }


def _summarize_plan_file(filename: str) -> dict:
    basename = filename[: -len("_move_plan.json")]
    with open(os.path.join(GENERATE_DIR, filename)) as f:
        raw = json.load(f)
    items = _items_of(raw)
    movel = sum(1 for p in items if p.get("move_type") == "movel")
    movec = sum(1 for p in items if p.get("move_type") == "movec")
    order = "→".join("movel" if p.get("move_type") == "movel" else "movec" for p in items)
    return {
        "basename": basename,
        "path": _plan_path(basename),
        "segment_count": len(items),
        "movel_count": movel,
        "movec_count": movec,
        "order": order,
        "format": "v2" if _is_v2(raw) else "v1",
    }


@router.get("")
def list_plans():
    if not os.path.isdir(GENERATE_DIR):
        return []
    files = sorted(f for f in os.listdir(GENERATE_DIR) if f.endswith("_move_plan.json"))
    return [_summarize_plan_file(f) for f in files]


@router.get("/segments-pool")
def segments_pool():
    """"순차 블록"에서 "다른 스킬에서 가져오기"로 보여줄 후보 - 지금까지 생성된 모든
    스킬(v2, 좌표 포함)의 세그먼트를 나열한다. v1(좌표 없음)은 후보에서 제외한다."""
    if not os.path.isdir(GENERATE_DIR):
        return []
    pool = []
    for filename in sorted(os.listdir(GENERATE_DIR)):
        if not filename.endswith("_move_plan.json"):
            continue
        source_basename = filename[: -len("_move_plan.json")]
        try:
            with open(os.path.join(GENERATE_DIR, filename)) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_v2(raw):
            continue
        for i, item in enumerate(raw["sequence"]):
            pool.append({
                "source_basename": source_basename,
                "segment_index": i,
                "type": "arc" if item["move_type"] == "movec" else "line",
                "path_length_mm": None,
                "end_pose": item.get("end_pose"),
                "via_pose": item.get("via_pose"),
            })
    return pool


@router.get("/{basename}")
def get_plan_detail(basename: str):
    plan_path = _plan_path(basename)
    if not os.path.isfile(plan_path):
        raise HTTPException(404, "스킬을 찾을 수 없습니다.")
    with open(plan_path) as f:
        raw = json.load(f)

    if _is_v2(raw):
        sequence = raw["sequence"]
        segments = [
            {
                "index": i,
                "type": "arc" if item["move_type"] == "movec" else "line",
                "end_pose": item.get("end_pose"),
                "via_pose": item.get("via_pose"),
                "path_length_mm": None,
            }
            for i, item in enumerate(sequence)
        ]
        plan = {
            str(i): {
                "move_type": item["move_type"],
                "vel_mm_s": item["vel_mm_s"],
                "acc_mm_s2": item["acc_mm_s2"],
                "blend_radius_mm": item["blend_radius_mm"],
                "reasoning": item.get("reasoning", ""),
                "source": item.get("source", "manual_edit"),
            }
            for i, item in enumerate(sequence)
        }
        return {"basename": basename, "format": "v2", "segments": segments, "plan": plan, "sequence": sequence}

    segments = []
    seg_path = _segments_path(basename)
    if os.path.isfile(seg_path):
        with open(seg_path) as f:
            segments = json.load(f).get("segments", [])
    return {"basename": basename, "format": "v1", "segments": segments, "plan": raw, "sequence": None}


class UpdateSegmentRequest(BaseModel):
    segment_index: int
    vel_mm_s: float | None = None
    acc_mm_s2: float | None = None
    blend_radius_mm: float | None = None


@router.put("/{basename}")
def update_segment(basename: str, req: UpdateSegmentRequest):
    plan_path = _plan_path(basename)
    if not os.path.isfile(plan_path):
        raise HTTPException(404, "스킬을 찾을 수 없습니다.")
    with open(plan_path) as f:
        raw = json.load(f)

    if _is_v2(raw):
        sequence = raw["sequence"]
        if not (0 <= req.segment_index < len(sequence)):
            raise HTTPException(404, f"세그먼트 {req.segment_index}를 찾을 수 없습니다.")
        item = sequence[req.segment_index]
        if req.vel_mm_s is not None:
            item["vel_mm_s"] = req.vel_mm_s
        if req.acc_mm_s2 is not None:
            item["acc_mm_s2"] = req.acc_mm_s2
        if req.blend_radius_mm is not None:
            item["blend_radius_mm"] = req.blend_radius_mm
        item["source"] = "manual_edit"
        with open(plan_path, "w") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        return item

    plan = raw
    key = str(req.segment_index)
    if key not in plan:
        raise HTTPException(404, f"세그먼트 {req.segment_index}를 찾을 수 없습니다.")

    if req.vel_mm_s is not None:
        plan[key]["vel_mm_s"] = req.vel_mm_s
    if req.acc_mm_s2 is not None:
        plan[key]["acc_mm_s2"] = req.acc_mm_s2
    if req.blend_radius_mm is not None:
        plan[key]["blend_radius_mm"] = req.blend_radius_mm
    plan[key]["source"] = "manual_edit"

    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    return plan[key]


class SequenceItem(BaseModel):
    move_type: str
    vel_mm_s: float
    acc_mm_s2: float
    blend_radius_mm: float
    end_pose: dict
    via_pose: dict | None = None
    source: str | None = None
    reasoning: str | None = None


class SequenceRequest(BaseModel):
    sequence: list[SequenceItem]


@router.put("/{basename}/sequence")
def put_sequence(basename: str, req: SequenceRequest):
    """"순차 블록"에서 순서 변경/삭제/다른 스킬 세그먼트 삽입 결과를 통째로 받아
    v2 포맷으로 저장한다. robot_replay가 다음 실행부터 이 순서/좌표 그대로 쓴다."""
    if not req.sequence:
        raise HTTPException(400, "시퀀스가 비어 있습니다.")

    items = [item.model_dump() for item in req.sequence]
    for item in items:
        if not item.get("source"):
            item["source"] = "manual_edit"
    items[-1]["blend_radius_mm"] = 0.0  # 마지막 세그먼트는 항상 완전 정지

    payload = {"format": "v2", "sequence": items}
    os.makedirs(GENERATE_DIR, exist_ok=True)
    with open(_plan_path(basename), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload
