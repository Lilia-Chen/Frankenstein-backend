from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


JointName = Literal[
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


class MotionFrame(BaseModel):
    timestamp: float
    root_position: List[float] = Field(min_length=3, max_length=3)
    root_rotation: List[float] = Field(min_length=4, max_length=4)
    joint_rotations: Dict[JointName, List[float]]
    joint_positions: Optional[Dict[JointName, List[float]]] = None
    root_velocity: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)


class ConditioningSpec(BaseModel):
    text: str


class GeneratePayload(BaseModel):
    conditioning: ConditioningSpec
    duration_seconds: Optional[float] = None
    fps: Optional[float] = 30.0
    current_frame: Optional[MotionFrame] = None  # Optional current pose for continuous generation


class GenerateRequest(BaseModel):
    type: Literal["generate"]
    id: str
    payload: GeneratePayload


class CancelRequest(BaseModel):
    type: Literal["cancel"]
    id: str


class FrameMessage(BaseModel):
    type: Literal["frame"]
    id: str
    frame: MotionFrame


class DoneMetadata(BaseModel):
    total_frames: int
    generation_time_ms: int
    model_name: str


class DoneMessage(BaseModel):
    type: Literal["done"]
    id: str
    metadata: DoneMetadata


class ErrorMessage(BaseModel):
    type: Literal["error"]
    id: str
    error: str
