import asyncio
from typing import AsyncIterator, Optional

from app.models.base import MotionGenerator
from app.schemas import MotionFrame

JOINT_NAMES = [
    "pelvis",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]
ZERO_VEC3 = [0.0, 0.0, 0.0]


class MockMotionGenerator(MotionGenerator):
    name = "mock-motion-gen"

    async def generate(
        self,
        text_prompt: str,
        duration_seconds: Optional[float],
        fps: float,
        initial_frames: Optional[list[dict]] = None,
    ) -> AsyncIterator[MotionFrame]:
        interval = 1.0 / fps
        total_frames = None
        if duration_seconds is not None:
            total_frames = max(1, int(duration_seconds * fps))

        index = 0
        while total_frames is None or index < total_frames:
            frame = MotionFrame(
                timestamp=index * interval,
                root_position=ZERO_VEC3,
                root_rotation=IDENTITY_QUAT,
                joint_rotations={name: IDENTITY_QUAT for name in JOINT_NAMES},
            )
            yield frame
            index += 1
            await asyncio.sleep(interval)
