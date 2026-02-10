import os
from typing import Optional

from app.models.base import MotionGenerator
from app.models.mock import MockMotionGenerator


def get_generator(model_name: Optional[str] = None) -> MotionGenerator:
    selected = model_name or os.getenv("MOTION_MODEL", "mock")
    if selected == "mock":
        return MockMotionGenerator()
    if selected == "dart":
        from app.models.dart import DartMotionGenerator

        return DartMotionGenerator()
    raise ValueError(f"unknown model: {selected}")
