from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from app.schemas import MotionFrame


class MotionGenerator(ABC):
    name: str

    @abstractmethod
    async def generate(
        self,
        text_prompt: str,
        duration_seconds: Optional[float],
        fps: float,
    ) -> AsyncIterator[MotionFrame]:
        raise NotImplementedError
