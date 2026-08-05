# RG2 파지까지 Mock flow

이 결과는 물체 이름을 스킬 정의에 고정하지 않고 런타임 anchor로만 바인딩합니다. 검증 이미지의
첫 RGB-D 프레임에서 얻은 파지점과 이미지 jaw 축을 사용했으며, LLM·DB·로봇·그리퍼 호출은
하지 않았습니다.

| 물체 | fingertip 표본 mm | 평균 폭 mm | image jaw ° | 184−5 mm | 수정식 적용 mm |
|---|---:|---:|---:|---:|---:|
| hammer | 24.4, 18.8 | 21.6 | +32.1 | 179.0 | 179.5 |
| jetty | 14.1 | 14.1 | -44.7 | 179.0 | 179.2 |
| knife | 34.7 | 34.7 | +86.3 | 179.0 | 180.4 |
| spanner | 25.0 | 25.0 | +82.1 | 179.0 | 179.7 |

마지막 열은 `R=110 mm`, `sin(theta)=(평균 폭/2)/R`라고 가정해
`184−5+R(1−cos(theta))`를 계산한 비교값입니다. 공식 사양의 110 mm는 전체 스트로크이므로
이 값은 실제 Z 명령으로 선택되지 않았습니다.

## hammer

- 동작 순서: `runtime binding → RGB-D 파지점 → gripper.open → pregrasp → yaw 정렬 → MoveL 하강 → gripper.close → holding 확인`
- 학습 폭 출처: `hammer1, hammer2`
- 예시 파지 폭: `21.58 mm`
- 최종 상태: Mock close까지 완료, hardware preflight 차단

## jetty

- 동작 순서: `runtime binding → RGB-D 파지점 → gripper.open → pregrasp → yaw 정렬 → MoveL 하강 → gripper.close → holding 확인`
- 학습 폭 출처: `jetty2`
- 예시 파지 폭: `14.10 mm`
- 최종 상태: Mock close까지 완료, hardware preflight 차단

## knife

- 동작 순서: `runtime binding → RGB-D 파지점 → gripper.open → pregrasp → yaw 정렬 → MoveL 하강 → gripper.close → holding 확인`
- 학습 폭 출처: `knife1`
- 예시 파지 폭: `34.72 mm`
- 최종 상태: Mock close까지 완료, hardware preflight 차단

## spanner

- 동작 순서: `runtime binding → RGB-D 파지점 → gripper.open → pregrasp → yaw 정렬 → MoveL 하강 → gripper.close → holding 확인`
- 학습 폭 출처: `spanner1`
- 예시 파지 폭: `25.02 mm`
- 최종 상태: Mock close까지 완료, hardware preflight 차단

## 실제 실행 전에 필요한 것

- 동일 active TCP·fingertip·task-plane revision에서 폭별 Z 보정 실측표
- validation 첫 프레임의 camera point/축을 task-plane으로 옮길 TF
- local IK와 collision/preflight 결과로 정한 TCP yaw; image angle을 joint 6에 직접 대입하지 않음
- Doosan/RG2 adapter 및 force/velocity profile의 hardware verification
