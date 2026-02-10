from fastapi import FastAPI, WebSocket

from app.ws import handle_motion_ws

app = FastAPI()


@app.websocket("/ws/motion")
async def motion_ws(ws: WebSocket) -> None:
    await handle_motion_ws(ws)
