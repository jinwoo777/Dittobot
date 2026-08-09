"""
generate_skill_code.py

2단계: 1단계에서 만든 세그먼트 리스트(hammering_..._segments.json)를 GPT에게
보내서 "각 세그먼트를 어떤 move 명령으로, 어떤 속도/블렌딩으로 실행할지"를
결정하게 한다.

핵심 설계 원칙 (지난 논의 그대로): GPT는 좌표를 만들지 않는다.
  - 좌표(x,y,z,yaw)는 전부 1단계 segments.json에 이미 들어있는 실측값을 그대로 쓴다.
    프롬프트에도 좌표는 아예 안 넣는다 (숫자를 새로 만들 여지 자체를 차단).
  - GPT가 실제로 결정하는 건: move_type(movel/movec 중 어느 쪽이 나은지 - 1단계
    분류를 뒤집을 수도 있음), vel/acc, blend_radius, 판단 근거(reasoning)뿐이다.
  - GPT 출력은 반드시 JSON. 스키마 검증을 통과 못하거나 세그먼트가 누락되면
    그 세그먼트는 안전한 기본값(movel, DEFAULT_VEL/ACC, blend 없음)으로 자동
    대체한다 - GPT 출력을 맹신하지 않는다.

처리 순서:
  1) segments.json 로드, 좌표 없이 요약 정보만 프롬프트에 넣음
     (분류 타입, 점 개수, 길이, 소요시간, arc면 fit_radius/residual)
  2) GPT 호출 (get_keyword.py와 동일하게 langchain ChatOpenAI, gpt-4o, .env의
     OPENAI_API_KEY 사용)
  3) GPT 응답을 JSON으로 파싱 + 스키마 검증, 실패/누락된 세그먼트는 기본값 대체
  4) move plan + 실제 좌표(1단계 실측값)를 합쳐서 movel/movec 호출로 이루어진
     Python 함수 코드를 문자열로 생성 -> .py 파일로 저장

주의 (실행 전 반드시 확인):
  - movec()의 실제 파라미터 이름/순서는 아직 실기로 확인 안 됨. robot_replay.py가
    movel/movej에 대해 했던 것과 동일하게:
        import DSR_ROBOT2, inspect; print(inspect.signature(DSR_ROBOT2.movec))
    로 실제 시그니처를 확인하고, 생성된 move_and_verify_arc()의 movec() 호출부를
    맞춰야 한다. 지금은 (via_point, target_point, vel, acc, ref, mod) 순서로
    "가정"하고 스켈레톤만 만들어 둔다.
  - 생성된 코드는 사람이 검토하고 verify_trajectory.py류 검증을 거친 뒤
    robot_replay.py에 수동으로 병합하는 걸 전제로 한다. 절대 자동 실행 안 함.

사용:
    ros2 run ditto_system generate_skill_code skills/classify/hammering_1786155386_smoothed_segments.json
    (출력: <SKILLS_ROOT>/generate/hammering_..._move_plan.json + hammering_..._skill.py,
     입력 파일이 어느 폴더에 있든 출력은 항상 SKILLS_ROOT/generate/ 밑에 파일명만 써서 저장)
"""

import json
import os
import sys

DEFAULT_VEL_MM_S = 15.0     # robot_replay.py의 VEL_MM_S와 동일한 기본값
DEFAULT_ACC_MM_S2 = 15.0
DEFAULT_BLEND_RADIUS_MM = 0.0
MAX_VEL_MM_S = 100.0        # GPT가 이보다 큰 값을 제안해도 여기서 강제로 clamp
MAX_ACC_MM_S2 = 150.0
ARC_RESIDUAL_RATIO_WARN = 0.15  # fit_residual/fit_radius가 이보다 크면 GPT에게 힌트 제공

ORIENTATION_RZ_DEG = 156.20  # robot_replay.py의 ORIENTATION_RZ_DEG와 동일한 값(반드시 동기화)

# 녹화/가공된 궤적 데이터는 colcon install 산출물(share/)이 아니라 이 고정 경로에
# 둔다 - install share는 재빌드 시 덮어써질 수 있는 빌드 산출물이라 계속 쌓이는
# 실행 데이터를 두기엔 안 맞고, ros2 run은 임의의 작업 디렉터리에서 실행될 수
# 있어 상대경로("skills/generate")도 못 믿는다.
SKILLS_ROOT = os.path.expanduser("~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills")
OUTPUT_DIR = os.path.join(SKILLS_ROOT, "generate")


def load_segments(path):
    with open(path) as f:
        data = json.load(f)
    return data["segments"], data.get("source")


def summarize_segment_for_prompt(seg):
    """GPT 프롬프트용 요약 - 좌표는 절대 넣지 않는다."""
    start_t = seg["start_pose"]["time"]
    end_t = seg["end_pose"]["time"]
    summary = {
        "segment_index": seg["index"],
        "classified_type": seg["type"],
        "point_count": seg["point_count"],
        "duration_sec": round(end_t - start_t, 3),
        "path_length_mm": seg["path_length_mm"],
    }
    if seg["type"] == "arc":
        r = seg["fit_radius_mm"]
        resid = seg["fit_residual_mm"]
        summary["fit_radius_mm"] = r
        summary["fit_residual_mm"] = resid
        if r:
            summary["residual_ratio"] = round(resid / r, 3)
    return summary


def build_prompt(segments):
    summaries = [summarize_segment_for_prompt(s) for s in segments]
    max_index = max(s["index"] for s in segments)
    return f"""
당신은 협동로봇(두산 M0609) 궤적 실행 계획을 세우는 어시스턴트입니다.
아래는 기록된 궤적을 line/arc 구간으로 분류한 결과입니다. 좌표는 일부러
주지 않습니다 - 좌표는 이미 실측으로 확정되어 있고, 당신이 새로 만들 필요가
없습니다. 당신은 오직 "어떻게 움직일지"만 결정합니다.

<세그먼트 요약>
{json.dumps(summaries, ensure_ascii=False, indent=2)}

<결정할 항목> (세그먼트마다)
- move_type: "movel"(직선) 또는 "movec"(원호).
  classified_type이 "arc"라도 residual_ratio가 {ARC_RESIDUAL_RATIO_WARN} 이상이면
  실제로는 원이 아닐 수 있으니 movel로 바꾸는 걸 권장합니다.
  classified_type이 "line"인 세그먼트는 via point가 없어서 movec을 쓸 수 없으니
  반드시 movel이어야 합니다.
- vel_mm_s: duration_sec과 path_length_mm으로 평균 속도를 가늠하되,
  {DEFAULT_VEL_MM_S}~{MAX_VEL_MM_S} 범위 안에서 정하세요.
- acc_mm_s2: {DEFAULT_ACC_MM_S2}~{MAX_ACC_MM_S2} 범위 안에서 정하세요.
- blend_radius_mm: 다음 세그먼트로 자연스럽게 이어지도록 하는 블렌딩 반경.
  segment_index {max_index}(마지막)는 반드시 0으로 하세요 (완전히 멈춰야 함).
- reasoning: 왜 이렇게 정했는지 한 줄 설명.

<출력 형식>
다른 설명 없이 아래 스키마의 JSON 배열만 출력하세요:
[
  {{
    "segment_index": 0,
    "move_type": "movel",
    "vel_mm_s": 15.0,
    "acc_mm_s2": 15.0,
    "blend_radius_mm": 20.0,
    "reasoning": "..."
  }}
]
"""


def call_gpt(prompt: str) -> str:
    """get_keyword.py와 동일한 방식(langchain ChatOpenAI, gpt-4o, .env)으로 호출."""
    from ament_index_python.packages import get_package_share_directory
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    # colcon으로 설치된 패키지 share 디렉터리(resource/.env) 기준으로 찾는다.
    # (get_keyword.py도 동일한 방식)
    env_path = os.path.join(get_package_share_directory("ditto_system"), "resource", ".env")
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY가 없습니다. {env_path} 파일을 확인하세요.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=api_key)
    response = llm.invoke(prompt)
    return response.content


def parse_move_plan(raw_text: str, segments: list) -> dict:
    """GPT 응답 파싱 + 검증. 유효하지 않은 세그먼트는 안전한 기본값으로 대체.
    반환: {segment_index: {move_type, vel_mm_s, acc_mm_s2, blend_radius_mm, reasoning, source}}"""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        bracket = text.find("[")
        if bracket >= 0:
            text = text[bracket:]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[경고] GPT 응답이 JSON으로 파싱되지 않습니다 ({e}). 전부 기본값으로 대체합니다.")
        parsed = []

    by_index = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "segment_index" in item:
                by_index[item["segment_index"]] = item

    plan = {}
    max_index = max(s["index"] for s in segments) if segments else -1
    for seg in segments:
        idx = seg["index"]
        item = by_index.get(idx, {})
        via_available = seg["via_idx"] is not None

        move_type = item.get("move_type")
        if move_type not in ("movel", "movec"):
            move_type = "movel"
        if move_type == "movec" and not via_available:
            move_type = "movel"  # via point 없으면 movec 물리적으로 불가

        try:
            vel = float(item.get("vel_mm_s", DEFAULT_VEL_MM_S))
            acc = float(item.get("acc_mm_s2", DEFAULT_ACC_MM_S2))
            blend = float(item.get("blend_radius_mm", DEFAULT_BLEND_RADIUS_MM))
        except (TypeError, ValueError):
            vel, acc, blend = DEFAULT_VEL_MM_S, DEFAULT_ACC_MM_S2, DEFAULT_BLEND_RADIUS_MM

        vel = max(1.0, min(vel, MAX_VEL_MM_S))
        acc = max(1.0, min(acc, MAX_ACC_MM_S2))
        blend = max(0.0, blend)
        if idx == max_index:
            blend = 0.0  # 마지막 세그먼트는 항상 완전 정지

        plan[idx] = {
            "move_type": move_type,
            "vel_mm_s": vel,
            "acc_mm_s2": acc,
            "blend_radius_mm": blend,
            "reasoning": item.get("reasoning", "(GPT 응답 없음/누락 - 기본값 사용)"),
            "source": "gpt" if idx in by_index else "default_fallback",
        }
    return plan


def build_v2_sequence(segments: list, plan: dict, first_yaw: float) -> list:
    """segments(좌표, classify_trajectory 출력)와 plan(속도/가감속/blend/move_type,
    parse_move_plan 출력)을 인덱스로 합쳐서 좌표까지 포함한 v2 sequence를 만든다.
    robot_replay.py의 rz = ORIENTATION_RZ_DEG + (yaw - first_yaw) 보정을 여기서 미리
    구워 넣어서, 실행 시점에 smoothed json/first_yaw 없이 이 파일만으로 재생 가능하게 한다."""
    def _baked_pose(pose):
        if pose is None:
            return None
        baked = dict(pose)
        baked["yaw"] = ORIENTATION_RZ_DEG + (pose["yaw"] - first_yaw)
        return baked

    sequence = []
    for seg in segments:
        idx = seg["index"]
        p = plan[idx]
        sequence.append({
            "move_type": p["move_type"],
            "vel_mm_s": p["vel_mm_s"],
            "acc_mm_s2": p["acc_mm_s2"],
            "blend_radius_mm": p["blend_radius_mm"],
            "reasoning": p.get("reasoning", ""),
            "source": p.get("source", "gpt"),
            "end_pose": _baked_pose(seg["end_pose"]),
            "via_pose": _baked_pose(seg.get("via_pose")),
        })
    if sequence:
        sequence[-1]["blend_radius_mm"] = 0.0  # 마지막 세그먼트는 항상 완전 정지
    return sequence


def render_posx(pose):
    """frame pose(x,y,z,yaw) -> DRL posx 리스트 코드 문자열.
    robot_replay.py replay_trajectory()와 동일한 변환식:
      rz = base_rz + (frame_yaw - first_yaw)
      z  = clamp_z_safe(frame_z + Z_GRASP_OFFSET_MM)
    좌표 숫자는 1단계 실측값을 그대로 박아 넣는다 (GPT가 만든 숫자가 아님).
    base_rz/first_yaw는 실행 시점(도구 검출 결과)에 정해지는 값이라 변수명 그대로 둔다."""
    x, y, z, yaw = pose["x"], pose["y"], pose["z"], pose["yaw"]
    return (
        f"[{x:.2f}, {y:.2f}, "
        f"clamp_z_safe({z:.2f} + Z_GRASP_OFFSET_MM, 'skill segment'), "
        f"ORIENTATION_RX_DEG, ORIENTATION_RY_DEG, "
        f"base_rz + ({yaw:.4f} - first_yaw)]"
    )


ARC_STUB = '''
def move_and_verify_arc(via_posx, target_posx, vel, acc, blend_radius, label):
    """movec()로 via_posx를 지나 target_posx까지 원호로 이동.
    [!] movec 실제 시그니처 미확인 상태의 가정 코드. 실행 전 반드시
        import DSR_ROBOT2, inspect; print(inspect.signature(DSR_ROBOT2.movec))
    로 확인하고 아래 인자 순서/이름을 맞출 것 (특히 blend_radius를 movec가
    받는지, 아니면 amovec/DR_MV_MOD 쪽 별도 파라미터인지 확인 필요)."""
    movec(via_posx, target_posx, vel=vel, acc=acc, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    mwait()

    actual, _ = get_current_posx()
    dist = sum((a - b) ** 2 for a, b in zip(actual[:3], target_posx[:3])) ** 0.5
    if dist > POSITION_TOLERANCE_MM:
        raise RuntimeError(
            f"{label}: 목표 근처로 못 갔습니다 (차이 {dist:.1f}mm). "
            f"movec 시그니처 가정이 실제와 다를 가능성이 있습니다."
        )
'''


def render_code(segments, plan, source_name):
    needs_arc_stub = any(p["move_type"] == "movec" for p in plan.values())

    lines = []
    lines.append('"""')
    lines.append(f"자동 생성됨: generate_skill_code.py  (원본: {source_name})")
    lines.append("사람이 검토 후 robot_replay.py에 병합할 것. 자동 실행 금지.")
    if needs_arc_stub:
        lines.append("실행 전 확인: DSR_ROBOT2.movec 실제 시그니처를 inspect로 검증했는가? (move_and_verify_arc 참고)")
    lines.append('"""')
    lines.append("")

    if needs_arc_stub:
        lines.append(ARC_STUB.strip("\n"))
        lines.append("")

    lines.append("def run_generated_skill(base_rz, first_yaw):")
    lines.append('    """1단계 세그먼트 분류 + 2단계 GPT 결정 결과로 생성된 스킬 실행 함수."""')
    lines.append('    set_phase("MOVING", "[skill] Running generated trajectory...", (0, 255, 255))')

    for seg in segments:
        idx = seg["index"]
        p = plan[idx]
        label = f"segment {idx} ({p['move_type']}, {p['source']})"

        lines.append("")
        lines.append(f"    # --- segment {idx}: {seg['type']} 분류 -> {p['move_type']} 실행 ---")
        lines.append(f"    # GPT reasoning: {p['reasoning']}")

        if p["move_type"] == "movec":
            via_posx = render_posx(seg["via_pose"])
            end_posx = render_posx(seg["end_pose"])
            lines.append(f"    via_posx = {via_posx}")
            lines.append(f"    end_posx = {end_posx}")
            lines.append(
                f"    move_and_verify_arc(via_posx, end_posx, vel={p['vel_mm_s']:.1f}, "
                f"acc={p['acc_mm_s2']:.1f}, blend_radius={p['blend_radius_mm']:.1f}, label={label!r})"
            )
        else:
            end_posx = render_posx(seg["end_pose"])
            lines.append(f"    posx = {end_posx}")
            lines.append(
                f"    move_and_verify(posx, {label!r}, vel={p['vel_mm_s']:.1f}, acc={p['acc_mm_s2']:.1f})"
            )

    lines.append("")
    return "\n".join(lines)


def print_plan_summary(segments, plan):
    print(f"\n생성된 move plan ({len(segments)}개 세그먼트)\n")
    header = f"{'#':>3} {'분류':>5}    {'실행':>6} {'vel':>6} {'acc':>6} {'blend':>6}  출처            reasoning"
    print(header)
    print("-" * len(header))
    for seg in segments:
        idx = seg["index"]
        p = plan[idx]
        print(f"{idx:>3} {seg['type']:>5} -> {p['move_type']:>6} "
              f"{p['vel_mm_s']:>6.1f} {p['acc_mm_s2']:>6.1f} {p['blend_radius_mm']:>6.1f}  "
              f"{p['source']:<15} {p['reasoning']}")
    print()


def process(path: str):
    segments, source_name = load_segments(path)
    if not segments:
        print("[경고] 세그먼트가 없습니다.")
        return

    prompt = build_prompt(segments)
    try:
        raw = call_gpt(prompt)
    except Exception as e:
        print(f"[경고] GPT 호출 실패 ({e}). 전체 세그먼트를 기본값(movel)으로 대체합니다.")
        raw = "[]"

    plan = parse_move_plan(raw, segments)
    print_plan_summary(segments, plan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(path)

    out_plan_path = os.path.join(OUTPUT_DIR, base.replace(".json", "_move_plan.json"))
    with open(out_plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"[저장] {out_plan_path}")

    code = render_code(segments, plan, source_name)
    out_code_path = os.path.join(OUTPUT_DIR, base.replace(".json", "_skill.py"))
    with open(out_code_path, "w") as f:
        f.write(code)
    print(f"[저장] {out_code_path}  (검토 후 robot_replay.py에 수동 병합할 것)")


def main():
    if len(sys.argv) < 2:
        print("사용법: ros2 run ditto_system generate_skill_code <hammering_..._smoothed_segments.json>")
        sys.exit(1)
    process(sys.argv[1])


if __name__ == "__main__":
    main()
