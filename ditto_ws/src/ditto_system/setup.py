import os

from setuptools import find_packages, setup

package_name = 'ditto_system'

# 정적 리소스(.env, mobile_sam.pt, T_gripper2camera.npy)만 share에 설치한다.
# skills/(녹화 데이터)는 계속 쌓이는 실행 데이터라 여기 안 넣는다 - 각 스크립트가
# SKILLS_ROOT 절대경로(~/Desktop/Dittobot/ditto_ws/src/ditto_system/skills)로 직접 읽고 쓴다.
# glob('resource/*')는 점(.)으로 시작하는 .env를 못 잡으므로(셸 글롭 방식) 명시한다.
RESOURCE_FILES = [
    'resource/.env',
    'resource/mobile_sam.pt',
    'resource/T_gripper2camera.npy',
]

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), RESOURCE_FILES),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey4090',
    maintainer_email='rokey4090@todo.todo',
    description=(
        'Desktop/test2에서 옮겨온 도구 픽업 + 궤적 재생(실시간 스킬화) + 손 핸드오버 파이프라인. '
        'Dittobot의 robot_skill_system과는 별도의 독립 실행 패키지.'
    ),
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_replay = ditto_system.robot_replay:main',
            'get_keyword = ditto_system.get_keyword:main',
            'record_trajectory = ditto_system.record_trajectory:main',
            'smooth_trajectory = ditto_system.smooth_trajectory:main',
            'verify_trajectory = ditto_system.verify_trajectory:main',
            'classify_trajectory = ditto_system.classify_trajectory:main',
            'generate_skill_code = ditto_system.generate_skill_code:main',
            'jog_server = ditto_system.jog_server:main',
        ],
    },
)
