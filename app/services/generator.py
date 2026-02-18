import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import WebSocket

from app.schemas import DoneMessage, DoneMetadata, ErrorMessage, FrameMessage
from app.services.registry import get_generator


@dataclass
class GenerationState:
    cancel_event: asyncio.Event
    task: asyncio.Task


def _now_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


async def _send_error(ws: WebSocket, req_id: str, message: str) -> None:
    await ws.send_json(ErrorMessage(type="error", id=req_id, error=message).model_dump())


async def stream_generation(
    ws: WebSocket,
    req_id: str,
    text_prompt: str,
    duration_seconds: Optional[float],
    fps: float,
    cancel_event: asyncio.Event,
    model_name: Optional[str] = None,
    current_frames: Optional[List[Dict]] = None,
) -> None:
    generator = get_generator(model_name)
    start = time.perf_counter()
    frames = 0

    try:
        async for frame in generator.generate(text_prompt, duration_seconds, fps, current_frames):
            if cancel_event.is_set():
                break
            msg = FrameMessage(type="frame", id=req_id, frame=frame)
            await ws.send_json(msg.model_dump())
            frames += 1

        done = DoneMessage(
            type="done",
            id=req_id,
            metadata=DoneMetadata(
                total_frames=frames,
                generation_time_ms=_now_ms(start),
                model_name=generator.name,
            ),
        )
        await ws.send_json(done.model_dump())
    except Exception as exc:
        await _send_error(ws, req_id, f"generation failed: {exc}")


class SessionManager:
    def __init__(self) -> None:
        self._active: Dict[str, GenerationState] = {}

    def cancel(self, req_id: str) -> bool:
        state = self._active.get(req_id)
        if state is None:
            return False
        state.cancel_event.set()
        return True

    def start(
        self,
        ws: WebSocket,
        req_id: str,
        text_prompt: str,
        duration_seconds: Optional[float],
        fps: float,
        model_name: Optional[str] = None,
        current_frame: Optional[Dict] = None,
    ) -> None:
        if req_id in self._active:
            self._active[req_id].cancel_event.set()

        frames_to_use: Optional[List[Dict]] = None
        if current_frame is not None:
            if hasattr(current_frame, "model_dump"):
                current_frame = current_frame.model_dump()
            frames_to_use = [current_frame, current_frame]

        cancel_event = asyncio.Event()
        task = asyncio.create_task(
            stream_generation(
                ws,
                req_id,
                text_prompt,
                duration_seconds,
                fps,
                cancel_event,
                model_name,
                frames_to_use,
            )
        )
        self._active[req_id] = GenerationState(cancel_event=cancel_event, task=task)

    async def shutdown(self) -> None:
        for state in self._active.values():
            state.cancel_event.set()
        await asyncio.gather(
            *(state.task for state in self._active.values()), return_exceptions=True
        )
