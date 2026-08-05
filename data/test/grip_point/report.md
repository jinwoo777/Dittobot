# Grip point train/validation 오프라인 실험

## 결론

각 물체의 독립 녹화 마지막 1개를 validation으로 분리했습니다. Validation 파지점은
`rgb/000000.jpg`와 `depth/000000.npz`만으로 먼저 확정했고, 이후 프레임의 MediaPipe 접촉점은
예측에 사용하지 않고 held-out reference로만 사용했습니다. 이 실험은 로봇 실행이나 DB 승격을
수행하지 않은 camera-relative 진단입니다.

## 결과

| 물체 | Train | Validation | point px / point metric / jaw angle |
|---|---|---|---|
| hammer | hammer1, hammer2 (2 usable) | hammer3 | 23.1 px / 14.6 mm / 22.6° |
| jetty | jetty1, jetty2 (1 usable) | jetty3 | 95.0 px / 52.2 mm / 5.6° |
| knife | knife1 (1 usable) | knife2 | 15.0 px / 9.9 mm / 31.6° |
| spanner | spanner1 (1 usable) | spanner2 | 25.1 px / 16.0 mm / 5.6° |

- 평균 pixel error: `39.6 px`
- 평균 metric error: `23.2 mm`
- 평균 jaw-axis error: `16.3°`
- 전체 이미지: `all_objects_summary.jpg`

노란색은 train에서 전이한 예측이며, 빨간색은 validation 접촉을 물체 상대좌표로 정규화한 뒤
첫 프레임에 다시 투영한 reference입니다. 파란 화살표는 물체의 canonical 장축입니다.

## hammer

- train: `hammer1, hammer2`
- validation: `hammer3` — 예측 입력은 첫 RGB-D 프레임 한 쌍뿐
- 학습 파지점: longitudinal `-0.718`,
  lateral `-0.021`
- 학습 jaw 상대각: `+66.5°`
- 손가락 3D 거리 표본: `hammer1=24.4 mm, hammer2=18.8 mm`
- 예시 파지 폭(산술평균): `21.6 mm`
- 이미지: `hammer/summary.jpg`

경고:

- fewer than three usable independent training demonstrations
- validation contact uses minimum-distance fallback and needs operator confirmation

## jetty

- train: `jetty1, jetty2`
- validation: `jetty3` — 예측 입력은 첫 RGB-D 프레임 한 쌍뿐
- 학습 파지점: longitudinal `-0.123`,
  lateral `-0.694`
- 학습 jaw 상대각: `-38.9°`
- 손가락 3D 거리 표본: `jetty2=14.1 mm`
- 예시 파지 폭(산술평균): `14.1 mm`
- 이미지: `jetty/summary.jpg`

경고:

- fewer than three usable independent training demonstrations
- one or more assigned training recordings were unusable

## knife

- train: `knife1`
- validation: `knife2` — 예측 입력은 첫 RGB-D 프레임 한 쌍뿐
- 학습 파지점: longitudinal `-0.289`,
  lateral `+0.131`
- 학습 jaw 상대각: `+78.6°`
- 손가락 3D 거리 표본: `knife1=34.7 mm`
- 예시 파지 폭(산술평균): `34.7 mm`
- 이미지: `knife/summary.jpg`

경고:

- fewer than three usable independent training demonstrations
- validation contact uses minimum-distance fallback and needs operator confirmation

## spanner

- train: `spanner1`
- validation: `spanner2` — 예측 입력은 첫 RGB-D 프레임 한 쌍뿐
- 학습 파지점: longitudinal `-0.272`,
  lateral `+0.003`
- 학습 jaw 상대각: `-79.7°`
- 손가락 3D 거리 표본: `spanner1=25.0 mm`
- 예시 파지 폭(산술평균): `25.0 mm`
- 이미지: `spanner/summary.jpg`

경고:

- fewer than three usable independent training demonstrations

## 해석 제한

- 파지 성공을 증명하는 object-following/post-lift label이 없어 접촉 후보는 운영자 미확인입니다.
- knife는 3 cm closed 상태가 없어서 최소 fingertip 거리 프레임을 사용했습니다.
- 물체 검출은 이번 데이터에 맞춘 색상 기반 local CV이며 LLM을 호출하지 않았습니다.
- 생성된 3D 점은 camera frame 진단값이며 task-plane/hand-eye TF가 없어 실행할 수 없습니다.
- 손가락 거리 평균은 사람의 접촉 폭이며 RG2 명령 폭과 동일하다고 보장되지 않습니다.
- 모든 물체의 usable train 시연이 3개 미만이므로 일반화 성능 수치로 해석하면 안 됩니다.
