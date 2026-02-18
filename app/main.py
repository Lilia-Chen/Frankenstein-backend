import asyncio
import os

from fastapi import FastAPI, WebSocket

from app.ws import handle_motion_ws

app = FastAPI()


@app.on_event("startup")
async def warmup() -> None:
    if os.getenv("MOTION_MODEL") == "motionstreamer":
        from app.models.motionstreamer import MotionStreamerRuntime
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, MotionStreamerRuntime.get)


@app.websocket("/ws/motion")
async def motion_ws(ws: WebSocket) -> None:
    await handle_motion_ws(ws)
