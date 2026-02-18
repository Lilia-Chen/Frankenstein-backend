import asyncio
import math
import os
import sys
import threading
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.models.base import MotionGenerator
from app.schemas import MotionFrame


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


class MotionStreamerRuntime:
    _instance: Optional["MotionStreamerRuntime"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.ms_root = Path(os.getenv("MS_ROOT", "MotionStreamer")).resolve()
        self.device = os.getenv("MS_DEVICE", "cuda")
        self.tae_ckpt = os.getenv("MS_TAE_CKPT", "")
        self.trans_ckpt = os.getenv("MS_TRANS_CKPT", "")
        self.text_encoder_name = os.getenv("MS_TEXT_ENCODER", "sentencet5-xxl/")
        self.unit_length = int(os.getenv("MS_UNIT_LENGTH", "4"))
        self.block_size = int(os.getenv("MS_BLOCK_SIZE", "78"))
        self.threshold = float(os.getenv("MS_END_THRESHOLD", "0.1"))
        self.cfg = float(os.getenv("MS_CFG_SCALE", "4.0"))
        self.temperature = float(os.getenv("MS_TEMPERATURE", "1.0"))
        self.mean_std_root = Path(
            os.getenv("MS_MEAN_STD_ROOT", str(self.ms_root / "humanml3d_272/mean_std"))
        )
        self.reference_end_latent_path = Path(
            os.getenv(
                "MS_REFERENCE_END_LATENT",
                str(self.ms_root / "reference_end_latent_t2m_272.npy"),
            )
        )

        self._setup_imports()
        self._init_runtime()

    @classmethod
    def get(cls) -> "MotionStreamerRuntime":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _setup_imports(self) -> None:
        if str(self.ms_root) not in sys.path:
            sys.path.insert(0, str(self.ms_root))

    def _init_runtime(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for MotionStreamer. "
                "Install it or update your environment."
            ) from exc

        import options.option_transformer as option_trans
        from models.llama_model import LLaMAHF, LLaMAHFConfig
        import models.tae as tae
        from visualization.recover_visualize import recover_from_local_rotation
        from utils.face_z_align_util import axis_angle_to_quaternion

        self.recover_from_local_rotation = recover_from_local_rotation
        self.axis_angle_to_quaternion = axis_angle_to_quaternion

        import sys as _sys
        _argv, _sys.argv = _sys.argv, _sys.argv[:1]
        args = option_trans.get_args_parser()
        _sys.argv = _argv
        torch.manual_seed(args.seed)

        self.device_t = torch.device(
            self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        )

        self.text_encoder = SentenceTransformer(self.text_encoder_name)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        clip_range = [-30, 20]
        self.tae = tae.Causal_HumanTAE(
            hidden_size=args.hidden_size,
            down_t=args.down_t,
            stride_t=args.stride_t,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation="relu",
            latent_dim=args.latent_dim,
            clip_range=clip_range,
        ).to(self.device_t)

        config = LLaMAHFConfig.from_name("Normal_size")
        config.block_size = self.block_size
        self.trans_encoder = LLaMAHF(
            config, args.num_diffusion_head_layers, args.latent_dim, self.device_t
        ).to(self.device_t)

        if not self.tae_ckpt:
            raise FileNotFoundError(
                "MS_TAE_CKPT is not set. Provide a Causal TAE checkpoint path."
            )
        if not self.trans_ckpt:
            raise FileNotFoundError(
                "MS_TRANS_CKPT is not set. Provide a MotionStreamer transformer checkpoint path."
            )

        tae_ckpt = torch.load(self.tae_ckpt, map_location="cpu")
        self.tae.load_state_dict(tae_ckpt["net"], strict=True)
        self.tae.eval()

        trans_ckpt = torch.load(self.trans_ckpt, map_location="cpu")
        new_ckpt_trans = {}
        for key, value in trans_ckpt["trans"].items():
            new_key = ".".join(key.split(".")[1:]) if key.split(".")[0] == "module" else key
            new_ckpt_trans[new_key] = value
        self.trans_encoder.load_state_dict(new_ckpt_trans, strict=True)
        self.trans_encoder.eval()

        diff_steps = os.getenv("MS_DIFF_STEPS", "10")
        from models.diffusion import create_diffusion
        self.trans_encoder.diff_loss.gen_diffusion = create_diffusion(
            timestep_respacing=diff_steps, noise_schedule="cosine"
        )

        mean_path = self.mean_std_root / "Mean.npy"
        std_path = self.mean_std_root / "Std.npy"
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(
                f"Mean/Std not found at {self.mean_std_root}. "
                "Set MS_MEAN_STD_ROOT to the correct directory."
            )
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

        self.reference_end_latent = None
        if self.reference_end_latent_path.exists():
            ref = np.load(self.reference_end_latent_path)
            self.reference_end_latent = torch.from_numpy(ref).to(self.device_t)

    def _infer_motion_272(self, text: str, length: int) -> np.ndarray:
        with torch.no_grad():
            motion_latents = self.trans_encoder.sample_for_eval_CFG_inference(
                text=text,
                tokenizer=self.text_encoder,
                device=self.device_t,
                reference_end_latent=self.reference_end_latent,
                threshold=self.threshold,
                cfg=self.cfg,
                temperature=self.temperature,
                length=length,
                unit_length=self.unit_length,
            )
            motion_seqs = self.tae.forward_decoder(motion_latents)
        motion = motion_seqs.squeeze(0).detach().cpu().numpy()
        motion = motion * self.std + self.mean
        return motion

    def _motion_to_frames(self, motion_272: np.ndarray, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        smpl_85 = self.recover_from_local_rotation(motion_272, njoint=22)
        smpl_85 = smpl_85[:num_frames]

        axis_angles = smpl_85[:, :66].reshape(-1, 22, 3)
        axis_angle_t = torch.from_numpy(axis_angles).float()
        quat_wxyz = self.axis_angle_to_quaternion(axis_angle_t).detach().cpu().numpy()
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]

        root_positions = smpl_85[:, 72:75]
        return root_positions, quat_xyzw

    def generate_frames_token(
        self,
        text: str,
        num_frames: int,
        token_callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Generate motion token by token, calling token_callback after each token is decoded."""
        max_length = self.block_size * self.unit_length
        length = int(math.ceil(num_frames / float(self.unit_length)) * self.unit_length)
        if length > max_length:
            length = max_length
        max_token_len = length // self.unit_length

        feat_text = torch.from_numpy(self.text_encoder.encode(text)).float().to(self.device_t)
        empty_feat = (
            torch.from_numpy(self.text_encoder.encode("")).float().unsqueeze(0).to(self.device_t)
        )

        xs: Optional[torch.Tensor] = None
        frames_yielded = 0

        with torch.no_grad():
            for k in range(max_token_len):
                x = [] if k == 0 else xs

                conditions = self.trans_encoder.forward_inference(x, feat_text)[:, -1, :]
                empty_cond = self.trans_encoder.forward(x, empty_feat)[:, -1, :]
                mix = torch.cat([conditions, empty_cond], dim=0)
                sampled = self.trans_encoder.diff_loss.sample(
                    mix, temperature=self.temperature, cfg=self.cfg
                )
                token, _ = sampled.chunk(2, dim=0)
                token = token.unsqueeze(0)

                if self.reference_end_latent is not None:
                    d = torch.sqrt(torch.sum((token - self.reference_end_latent) ** 2))
                    if d < self.threshold:
                        break

                xs = token if xs is None else torch.cat((xs, token), dim=1)

                # Decode accumulated tokens; only the last unit_length frames are new
                motion_seqs = self.tae.forward_decoder(xs)
                motion = motion_seqs.squeeze(0).detach().cpu().numpy()
                motion = motion * self.std + self.mean
                new_frames = motion[-self.unit_length :]
                frames_to_emit = min(len(new_frames), num_frames - frames_yielded)
                if frames_to_emit <= 0:
                    break
                root_pos, quats = self._motion_to_frames(new_frames, frames_to_emit)
                token_callback(root_pos, quats)
                frames_yielded += frames_to_emit
                if frames_yielded >= num_frames:
                    break


_NATIVE_FPS = 20  # HumanML3D native frame rate


class MotionStreamerGenerator(MotionGenerator):
    name = "motionstreamer"

    async def generate(
        self,
        text_prompt: str,
        duration_seconds: Optional[float],
        fps: float,
        initial_frames: Optional[List[Dict]] = None,
    ) -> AsyncIterator[MotionFrame]:
        runtime = MotionStreamerRuntime.get()

        duration = 4.0 if duration_seconds is None else float(duration_seconds)
        num_model_frames = max(1, int(duration * _NATIVE_FPS))
        max_length = runtime.block_size * runtime.unit_length
        length = int(math.ceil(num_model_frames / float(runtime.unit_length)) * runtime.unit_length)
        if length > max_length:
            length = max_length

        loop = asyncio.get_running_loop()
        motion_272 = await loop.run_in_executor(
            None, runtime._infer_motion_272, text_prompt, length
        )
        available = min(num_model_frames, motion_272.shape[0])
        root_positions, joint_quats = runtime._motion_to_frames(motion_272, available)

        native_interval = 1.0 / _NATIVE_FPS
        for idx in range(available):
            joint_rotations = {
                name: joint_quats[idx][j_idx].tolist() for j_idx, name in enumerate(JOINT_NAMES)
            }
            yield MotionFrame(
                timestamp=idx * native_interval,
                root_position=root_positions[idx].tolist(),
                root_rotation=joint_rotations["pelvis"],
                joint_rotations=joint_rotations,
            )
