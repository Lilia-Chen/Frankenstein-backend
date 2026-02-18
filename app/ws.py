import json
import os
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.schemas import CancelRequest, GenerateRequest
from app.services.generator import SessionManager


def _invalid(payload: Any) -> str:
    return "invalid message"


async def handle_motion_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json(
        {
            "type": "handshake",
            "capabilities": {
                "supportsText": True,
                "supportsSpatial": False,
                "supportsTrajectory": False,
                "supportsTransition": False,
            },
        }
    )
    manager = SessionManager()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "id": "", "error": "invalid json"})
                continue

            msg_type = msg.get("type")
            if msg_type == "generate":
                try:
                    req = GenerateRequest.model_validate(msg)
                except Exception:
                    await ws.send_json({"type": "error", "id": "", "error": _invalid(msg)})
                    continue

                payload = req.payload
                print(f"generate request {req.id} payload={payload.model_dump()}")
                fps = payload.fps or 30.0
                if fps <= 0:
                    await ws.send_json(
                        {"type": "error", "id": req.id, "error": "fps must be > 0"}
                    )
                    continue

                # Extract current_frame if provided
                current_frame = None
                if payload.current_frame is not None:
                    current_frame = payload.current_frame.model_dump()

                manager.start(
                    ws=ws,
                    req_id=req.id,
                    text_prompt=payload.conditioning.text,
                    duration_seconds=payload.duration_seconds,
                    fps=fps,
                    model_name=os.getenv("MOTION_MODEL", "mock"),
                    current_frame=current_frame,
                )

            elif msg_type == "cancel":
                try:
                    req = CancelRequest.model_validate(msg)
                except Exception:
                    await ws.send_json({"type": "error", "id": "", "error": _invalid(msg)})
                    continue

                if not manager.cancel(req.id):
                    await ws.send_json(
                        {"type": "error", "id": req.id, "error": "unknown request id"}
                    )

            else:
                await ws.send_json(
                    {"type": "error", "id": msg.get("id", ""), "error": "unknown type"}
                )
    except WebSocketDisconnect:
        pass
    finally:
        await manager.shutdown()
