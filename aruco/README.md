# 물체 폭 기반 fixed ArUco workspace

이 폴더의 두 모듈은 기준 자세에서 측정한 ArUco plane을 한 번 고정하고, 물체 폭에 따라
TCP의 plane-Z 하한만 물체별로 계산합니다. +Z 상한은 frozen reference-camera 원점의 plane-Z로
고정합니다. 일반 workspace 점의 Z 범위는 `[plane z=0, frozen camera z]`이고, TCP는 gripper
clearance 때문에 더 엄격한 `[z_min(width), frozen camera z]`를 사용합니다. 두 범위를 섞으면
안 됩니다. 로봇·ROS·카메라·네트워크를 호출하지 않으며 내부 길이와 translation은 모두 metre입니다.

## 중요한 현재 데이터 상태

`results/d435i_plane_scans/scan_20260807_112154/plane_workspace_result.json`은 지금 바로
freeze하면 안 됩니다. 유효 workspace marker가 1, 3, 4번뿐이고 거의 일직선이어서 polygon의
최소 폭이 약 `1.062 mm`입니다. 9번 marker는 영상에서 검출됐지만 유효 depth center가 결과에
남지 않았습니다. 코드는 이 경계를 임의로 확장하지 않고 `5 mm` data-quality floor에서
거부합니다.

고정 `T_camera_plane`을 그대로 유지해야 한다면, 9번 등 비공선 marker의 유효 depth 점을 다시
얻어 기존 `T_plane_camera`로 투영한 승인 polygon을 만들어야 합니다. 아직 reference를 freeze하지
않았다면 기준 joint를 유지한 재측정도 가능합니다. 어느 경우에도 현재의 얇은 삼각형을 수동으로
늘려서 사용하면 안 됩니다.

후속 `scan_20260807_123803`은 1, 3, 4, 6, 8번 marker로 만든 4-vertex polygon이며 최소 폭이
약 `151.988 mm`라 공선성 검사를 통과합니다. 그래도 calibration/TCP identity, 기준 joint,
reachability와 충돌 검증을 별도로 마치기 전에는 hardware 승인 자료가 아닙니다.

## 1. fixed reference 생성

반드시 Doosan M0609 joint `[0, 0, 90, 0, 90, -90] deg`에서 얻은 결과를 사용합니다. 스크립트는
joint를 읽지 않으므로 이 조건은 작업자가 확인해야 합니다. 또한 이 저장소의
`T_gripper2camera_orig.npy`는 현재 문서상 active TCP 불일치와 held-out 검증 실패가 확인된
candidate이므로 영구 fixed reference의 calibration authority로 사용하면 안 됩니다
([근거](../docs/HARDWARE_SETUP.md#legacy-t_gripper2cameranpy)). 아래에는
동일한 `T_tcp_camera`, mm convention으로 별도 승인된 NPY 경로를 넣어야 합니다.

```bash
cd /home/rokey/Dittobot
python3 aruco/freeze_fixed_aruco_workspace.py \
  --plane-result-json /path/to/approved/plane_workspace_result.json \
  --tcp-camera-npy /path/to/approved_T_tcp_camera.npy \
  --tcp-camera-direction camera-to-tcp \
  --tcp-camera-translation-unit mm \
  --output-npz aruco/fixed_workspace_reference.npz
```

`fixed_workspace_reference.npz`는 create-only입니다. 이미 있으면 덮어쓰지 않습니다. 원본 JSON과
calibration NPY도 수정하지 않으며 SHA-256 provenance만 NPZ에 저장합니다.

freeze 단계에서 다음을 거부합니다.

- `+Z toward camera`가 불명확하거나 plane normal/transform과 모순되는 좌표계
- rigid transform이 아니거나 inverse가 맞지 않는 행렬
- NaN/Inf, 중복점, self-intersection, concave 또는 거의 공선인 XY polygon
- raw `boundary_xy_m`만 있고 `safe_boundary_xy_m`가 없는 결과
- 기존 fixed NPZ 또는 calibration 원본을 덮어쓰려는 경로

## 2. 물체 폭으로 runtime workspace 생성

기본 모델은 `w = R*sin(theta)`, `R=110 mm`입니다.

```bash
python3 aruco/object_width_workspace.py \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --width-mm 80
```

cm 입력과 legacy 모델도 지원합니다.

```bash
python3 aruco/object_width_workspace.py \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --width-cm 4 \
  --width-model legacy-half-factor
```

출력은 기본적으로 `aruco/runtime/runtime_workspace.npz`에 같은 디렉터리의 임시파일을 완전히
기록한 뒤 atomic replace됩니다. XY polygon과 모든 transform은 fixed reference와 같고,
`z_min_plane_m`과 폭 관련 metadata만 물체별로 계산됩니다. TCP 하한은 reference TCP Z에서
계산하고 상한은 frozen camera Z입니다.

80 mm, 기본 모델의 수치는 다음과 같습니다.

- `sin(theta) = (80 / 2) / 110`, `theta = 21.324 deg`
- `opening_offset = 7.530 mm`
- `geometric_allowed_down = 184 - 7.530 = 176.470 mm`
- `allowed_down = 176.470 - 5 = 171.470 mm`
- `z_min = z_tcp_reference - 171.470 mm`
- `z_max = z_frozen_reference_camera`

legacy 모델은 `w = R*sin(theta)/2`이므로 허용 최대 폭은 `55 mm`입니다. 기본 모델 최대 폭은
`220 mm`이지만, RG2가 실제로 명령할 수 있는 총 stroke는 110 mm이므로 API 입력은 110 mm에서
fail-closed합니다.

폭이 커질수록 허용 하강량은 작아지고 `z_min`은 위로 올라갑니다. live detector의 폭과 ArUco
세션에 입력한 폭이 다르면 더 큰 값을 사용해 더 깊은 하강을 허용하지 않습니다. 실제 gripper
geometry가 이 모델과 다르면 runtime을 사용하면 안 됩니다.

### 다른 컴퓨터에서 실행

저장소에는 최신 유효 스캔의 JSON/NPY와 create-only
`aruco/fixed_workspace_reference.npz`를 함께 보관합니다. Clone/pull 후 저장소 루트에서 먼저
전송 무결성을 확인합니다.

```bash
sha256sum -c aruco/PORTABLE_ARTIFACTS.sha256
```

`T_gripper2camera_orig.npy`도 이미 저장소에서 추적됩니다. 물체별
`runtime/runtime_workspace.npz`는 폭을 잘못 재사용하지 않도록 커밋하지 않고 각 컴퓨터에서
실제 물체 폭으로 다시 생성합니다.

```bash
python3 aruco/object_width_workspace.py \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --width-mm 80
```

fixed NPZ는 geometry reference이며 실제 로봇용 hardware 승인이나 현재 TCP 일치 검증을
대신하지 않습니다.

## 3. URDF로 base 좌표 후보 생성

`urdf_reference_candidate.py`는 로봇·ROS·카메라를 호출하지 않고, 전개된 URDF의
forward kinematics와 frozen NPZ의 raw `T_camera_plane`만 합성합니다. 행렬 규약은
`T_A_B @ p_B = p_A`입니다. URDF에 `camera_color_optical_frame`으로 가는 체인이 있으면
다음을 사용합니다.

```text
T_base_camera = FK_URDF(base_link, camera_color_optical_frame, q)
T_base_plane  = T_base_camera * T_camera_plane
```

이 모드는 legacy `T_tcp_camera`/`T_tcp_plane`을 base 변환에 사용하지 않습니다. URDF에
카메라 체인이 없으면 `flange == legacy TCP`라는 미검증 가정이 포함된 조건부
fallback만 생성합니다. 두 모드 모두 `status=candidate`, `usable_for_motion=false`,
`hardware_approved=false`입니다.

이 워크스테이션의 결합 모델은 다음 경로에 있습니다. 이는 배포 환경 의존
경로이며 Dittobot에 vendor URDF를 복사한 것은 아닙니다.

```text
/home/rokey/wok_wark/ws_cobot_pjt/ws_dsr/src/rg2/m0609_rg2_bringup/urdf/m0609_with_rg2_camera.urdf.xacro
```

현재 설치 본을 전개하고 기준 joint `[0, 0, 90, 0, 90, -90] deg`를 적용하는 재현
명령은 다음과 같습니다. 출력 JSON은 create-only이므로 기존 파일을 덮어쓰지
않습니다.

```bash
source /opt/ros/humble/setup.bash
source /home/rokey/wok_wark/ws_cobot_pjt/ws_dsr/install/setup.bash
xacro \
  /home/rokey/wok_wark/ws_cobot_pjt/ws_dsr/src/rg2/m0609_rg2_bringup/urdf/m0609_with_rg2_camera.urdf.xacro \
  -o /tmp/dittobot_m0609_rg2_d435_expanded.urdf

python3 aruco/urdf_reference_candidate.py \
  --urdf /tmp/dittobot_m0609_rg2_d435_expanded.urdf \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --base-link base_link \
  --flange-link link_6 \
  --camera-link camera_color_optical_frame \
  --joint-deg 0 0 90 0 90 -90 \
  --output-json \
    aruco/results/d435i_plane_scans/scan_20260807_123803/base_workspace_urdf_candidate.json
```

현재 expanded URDF SHA-256은
`827517126f4d21a2246d0b9862b746357437a0c7e8f0f7d3fb108b4eb853e66c`입니다. 원본
combined Xacro는 `0a60a23987295374d52aaa872ace359d21405d8356aa747217d51589b12f677c`,
bracket Xacro는 `3a74eabaa16c93a2dd76ad2c92472ec87dac7ecb710e7ac3c86200af7f60498b`,
M0609 URDF는 `05fa5a1ae173e1bf7a19a3bdaf2365b7c3d87b3e6289885b6e7eadb65c0d6b1b`입니다.

생성된 `T_base_link_plane`은 다음과 같습니다. translation은 metre입니다.

```text
[[ 0.000358036, -0.999622542, -0.027470797,  0.403021311],
 [ 0.999984096,  0.000203278,  0.005636138, -0.088195103],
 [-0.005628426, -0.027472378,  0.999606717, -0.179744032],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]
```

plane normal은 base 좌표에서 `[-0.027470797, 0.005636138, 0.999606717]`, base XY
plane 대비 기울기는 약 `1.606957 deg`입니다. 전체 행렬, `T_base_flange`,
`T_base_camera`, 4개 workspace 경계점과 source checksum은
[`base_workspace_urdf_candidate.json`](results/d435i_plane_scans/scan_20260807_123803/base_workspace_urdf_candidate.json)에
보존했습니다.

이 값은 bracket 설계치와 RealSense nominal extrinsics에 기대며 hand-eye 실측값이 아닙니다.
또한 ROS `base_link`와 controller `DR_BASE`의 일치를 실기로 검증하지 않았습니다.
따라서 시각화·재현성 확인에만 사용하고 target 생성, preflight 통과, 로봇 동작
허가의 근거로 사용하면 안 됩니다.

## 4. target 검사와 reject

아래 TCP 옵션은 점을 보정하지 않습니다. XY 또는 동적 TCP Z가 범위를 벗어나면 usable runtime을
publish하지 않고 종료 코드 `2`를 반환합니다. 기존 runtime을 잘못 재사용하지 않도록 표준 출력
경로는 Z bound가 없는 `unsafe_target_rejected` NPZ marker로 atomic replace됩니다.
폭 범위 오류나 reference 손상처럼 runtime 생성 자체가 실패한 경우도 가능한 한 같은 경로를
`runtime_generation_failed` marker로 바꿉니다. 소비자는 `artifact_kind`, `status`,
`usable_for_target_validation`, reference checksum/generation을 확인한 뒤에만 Z bound를 사용해야 합니다.
숫자가 아닌 폭, 필수 폭 누락 등 CLI parsing 실패도 같은 marker로 기존 정상 runtime을
무효화합니다. 단, `--help`는 읽기 전용이라 runtime을 바꾸지 않습니다.

기존 출력 파일은 이 모듈이 만든 schema 2.0 runtime 또는 rejection marker로 식별될 때만
교체합니다. 손상됐거나 정체를 확인할 수 없는 NPZ는 fixed reference일 가능성을 배제할 수 없어
자동으로 덮어쓰지 않습니다. 이 경우 파일을 보존한 채 nonzero로 실패하므로 운영자가 출처를
확인한 후 별도 경로를 사용하거나 수동으로 격리해야 합니다.

```bash
python3 aruco/object_width_workspace.py \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --width-mm 80 \
  --check-plane-xyz X_M Y_M Z_M
```

TCP/gripper clearance가 아닌 일반 workspace 점을 `plane z=0`부터 frozen camera Z까지 검사하려면
별도 옵션을 사용합니다.

```bash
python3 aruco/object_width_workspace.py \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --width-mm 80 \
  --check-workspace-plane-xyz X_M Y_M Z_M
```

`check_workspace_point_plane()`은 위 일반 기하 범위만 검사하며 TCP 실행 허가로 사용하면 안 됩니다.
TCP에는 반드시 `check_tcp_point_plane()` 또는 raising API를 사용합니다.

frozen reference-camera 좌표의 점은 다음과 같이 검사합니다.

```bash
python3 aruco/object_width_workspace.py \
  --reference-npz aruco/fixed_workspace_reference.npz \
  --width-mm 80 \
  --check-camera-ref-xyz X_M Y_M Z_M
```

이 camera 좌표는 기준 자세에서 고정한 가상 reference-camera frame입니다. eye-in-hand 카메라가
이동한 뒤의 live camera XYZ를 외부 TF 없이 직접 넣으면 안 됩니다. Python 호출부에서는
`require_tcp_point_plane()` 또는 `require_tcp_point_camera_ref()`를 사용하면 unsafe 입력에서
`UnsafeWorkspaceTargetError`가 발생합니다.

## 검사

```bash
python3 -m py_compile \
  aruco/freeze_fixed_aruco_workspace.py \
  aruco/object_width_workspace.py \
  aruco/urdf_reference_candidate.py \
  aruco/test_object_width_workspace.py

PYTHONDONTWRITEBYTECODE=1 python3 aruco/test_object_width_workspace.py

# 시스템 pytest plugin 충돌이 있을 때
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -m pytest -q aruco/test_object_width_workspace.py
```

## 안전 경계

이 모듈은 고정 gripper orientation/TCP 정의에서 endpoint TCP의 fixed-plane XYZ만 검사합니다.
다음 항목을 승인하지 않습니다.

- 실제 Doosan hardware 실행, joint/reachability/특이점/self-collision
- tool body와 swept path, MoveJ/MoveC 중간 경로, fixture/obstacle collision
- 실제 gripper opening/state가 입력 폭과 일치하는지 여부
- `DR_BASE <-> fixed plane`의 hardware-verified TF

`src/robot_skill_system/aruco_experiment/controller.py`와 `/ui/`의 `ArUco 실험` 화면은 첫 실험용
좁은 실행 경계를 제공합니다. 정확한 기준 joint에서 현재 `T_base_tcp`와 frozen
`T_tcp_plane`을 합성하여 `T_base_plane`을 한 세션 동안 고정하고, active TCP 일치, fixed NPZ
checksum/schema, 폭별 TCP Z, plane-to-camera Z 상한, XY polygon, Joint 1 기준 1 m TCP 반경,
IK/관절 한계와 2 mm 간격의 전체 +Z 20 mm 선분을 검사합니다. 실패한 target은 보정하지 않고
MoveL 전에 거절합니다.

이 전용 경계도 runtime NPZ 하나만으로 hardware를 승인하지 않습니다. 실제 모드에는 저장소의
5개 공통 hardware gate와 `ENABLE_ARUCO_EXPERIMENT=true`,
`ARUCO_EXPERIMENT_CELL_SAFETY_VERIFIED=true`, UI의 작업공간/E-stop/직접 이동 확인이 모두
필요합니다. 실제 gripper 폭 feedback과 일반 SkillGraph hardware 실행은 이 첫 실험 범위에
포함하지 않습니다.

Doosan bringup과 같은 ROS 2/DSR workspace가 source된 별도 터미널에서 다음처럼 API를
시작합니다. 현재 장치의 실제 TCP 이름이 다르면 임의로 맞추지 말고, frozen calibration을 만든
TCP와 일치하는 검증된 이름으로 `ARUCO_EXPERIMENT_EXPECTED_TCP`를 설정해야 합니다.

```bash
cd /home/rokey/Dittobot
source /opt/ros/humble/setup.bash
source /path/to/built_doosan_workspace/install/setup.bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export ROBOT_EXECUTION_MODE=hardware
export ENABLE_HARDWARE_EXECUTION=true
export ROBOT_BACKEND=doosan
export ENABLE_REAL_ROBOT=true
export DRY_RUN=false
export ENABLE_ARUCO_EXPERIMENT=true
export ARUCO_EXPERIMENT_CELL_SAFETY_VERIFIED=true
export ARUCO_EXPERIMENT_EXPECTED_TCP=GripperDA_v1
export ARUCO_FIXED_REFERENCE_NPZ="$PWD/aruco/fixed_workspace_reference.npz"
export ARUCO_RUNTIME_WORKSPACE_NPZ="$PWD/aruco/runtime/runtime_workspace.npz"
python3 -m uvicorn robot_skill_system.api.app:create_app \
  --factory --host 127.0.0.1 --port 8001
```

브라우저에서 `http://127.0.0.1:8001/ui/`의 `ArUco 실험`을 엽니다. gate 또는 DSR 서비스가
하나라도 준비되지 않으면 hardware 동작을 시도하지 않고 상태/API 오류로 거절합니다.
