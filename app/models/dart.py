import asyncio
import math
import os
import sys
import threading
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import torch
from pytorch3d import transforms as p3d_transforms

from app.models.base import MotionGenerator
from app.schemas import MotionFrame


DART_JOINT_NAMES = [
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

TARGET_JOINT_NAMES = [
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


class ClassifierFreeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        assert (
            self.model.cond_mask_prob > 0
        ), "Cannot run a guided diffusion on a model that has not been trained with no conditions"

    def forward(self, x, timesteps, y=None):
        y["uncond"] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond["uncond"] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y["scale"] * (out - out_uncond))


class DartRuntime:
    _instance: Optional["DartRuntime"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.dart_root = Path(os.getenv("DART_ROOT", "DART-main")).resolve()
        self.device = os.getenv("DART_DEVICE", "cuda")
        self.denoiser_ckpt = os.getenv(
            "DART_DENOISER_CKPT",
            str(
                self.dart_root
                / "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt"
            ),
        )
        self.guidance_param = float(os.getenv("DART_GUIDANCE", "5.0"))
        self.use_predicted_joints = int(os.getenv("DART_USE_PRED_JOINTS", "1"))
        self.batch_size = 1

        self._setup_dart_imports()
        self._init_runtime()

    @classmethod
    def get(cls) -> "DartRuntime":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _setup_dart_imports(self) -> None:
        if str(self.dart_root) not in sys.path:
            sys.path.insert(0, str(self.dart_root))

        import config_files.data_paths as data_paths

        data_paths.dataset_root_dir = self.dart_root / "data"
        data_paths.body_model_dir = (
            data_paths.dataset_root_dir / "smplx_lockedhead_20230207/models_lockedhead/"
        )
        data_paths.amass_dir = data_paths.dataset_root_dir / "amass"
        data_paths.babel_dir = data_paths.amass_dir / "babel-teach"

    def _init_runtime(self) -> None:
        from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
        from model.mld_vae import AutoMldVae
        from mld.train_mvae import Args as MVAEArgs
        from mld.train_mld import (
            DenoiserMLPArgs,
            DenoiserTransformerArgs,
            MLDArgs,
            create_gaussian_diffusion,
        )
        from data_loaders.humanml.data.dataset import SinglePrimitiveDataset

        self.DenoiserMLP = DenoiserMLP
        self.DenoiserTransformer = DenoiserTransformer
        self.DenoiserMLPArgs = DenoiserMLPArgs
        self.DenoiserTransformerArgs = DenoiserTransformerArgs
        self.MLDArgs = MLDArgs
        self.MVAEArgs = MVAEArgs
        self.AutoMldVae = AutoMldVae
        self.create_gaussian_diffusion = create_gaussian_diffusion
        self.SinglePrimitiveDataset = SinglePrimitiveDataset

        self.device_t = torch.device(
            self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        )

        denoiser_args, denoiser_model, vae_args, vae_model = self._load_mld(
            self.denoiser_ckpt, self.device_t
        )
        self.denoiser_args = denoiser_args
        self.denoiser_model = denoiser_model
        self.vae_args = vae_args
        self.vae_model = vae_model

        diffusion_args = denoiser_args.diffusion_args
        diffusion_args.respacing = ""
        self.diffusion = self.create_gaussian_diffusion(diffusion_args)

        sequence_path = self.dart_root / "data" / "stand.pkl"
        if not sequence_path.exists():
            raise FileNotFoundError(
                f"DART seed sequence not found: {sequence_path}"
            )

        cfg_path = Path(vae_args.data_args.cfg_path)
        if not cfg_path.is_absolute():
            cfg_path = self.dart_root / cfg_path
        data_dir = Path(vae_args.data_args.data_dir)
        if not data_dir.is_absolute():
            data_dir = self.dart_root / data_dir

        self.dataset = self.SinglePrimitiveDataset(
            cfg_path=str(cfg_path),
            dataset_path=str(data_dir),
            body_type=vae_args.data_args.body_type,
            sequence_path=str(sequence_path),
            batch_size=self.batch_size,
            device=self.device_t,
            enforce_gender="female",
            enforce_zero_beta=1,
        )

        self.history_length = self.dataset.history_length
        self.future_length = self.dataset.future_length

        from utils.misc_util import encode_text, compose_texts_with_and
        from utils.smpl_utils import PrimitiveUtility

        self.encode_text = encode_text
        self.compose_texts_with_and = compose_texts_with_and
        self.primitive_utility = self.dataset.primitive_utility

    def _load_mld(self, denoiser_checkpoint: str, device: torch.device):
        import yaml
        import tyro
        from dataclasses import asdict
        from pathlib import Path

        denoiser_dir = Path(denoiser_checkpoint).parent
        with open(denoiser_dir / "args.yaml", "r") as f:
            denoiser_args = tyro.extras.from_yaml(
                self.MLDArgs, yaml.safe_load(f)
            ).denoiser_args
        denoiser_class = (
            self.DenoiserMLP
            if isinstance(denoiser_args.model_args, self.DenoiserMLPArgs)
            else self.DenoiserTransformer
        )
        denoiser_model = denoiser_class(**asdict(denoiser_args.model_args)).to(device)
        checkpoint = torch.load(denoiser_checkpoint, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        denoiser_model.load_state_dict(model_state_dict)
        for param in denoiser_model.parameters():
            param.requires_grad = False
        denoiser_model.eval()
        denoiser_model = ClassifierFreeWrapper(denoiser_model)

        vae_checkpoint = denoiser_args.mvae_path
        vae_dir = Path(vae_checkpoint).parent
        with open(vae_dir / "args.yaml", "r") as f:
            vae_args = tyro.extras.from_yaml(self.MVAEArgs, yaml.safe_load(f))
        vae_model = self.AutoMldVae(**asdict(vae_args.model_args)).to(device)
        checkpoint = torch.load(denoiser_args.mvae_path, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        if "latent_mean" not in model_state_dict:
            model_state_dict["latent_mean"] = torch.tensor(0)
        if "latent_std" not in model_state_dict:
            model_state_dict["latent_std"] = torch.tensor(1)
        vae_model.load_state_dict(model_state_dict)
        vae_model.latent_mean = model_state_dict["latent_mean"]
        vae_model.latent_std = model_state_dict["latent_std"]
        for param in vae_model.parameters():
            param.requires_grad = False
        vae_model.eval()

        return denoiser_args, denoiser_model, vae_args, vae_model

    def _parse_texts(self, text_prompt: str) -> List[str]:
        texts: List[str] = []
        if "," in text_prompt:
            for segment in text_prompt.split(","):
                action, num_mp = segment.split("*")
                action = self.compose_texts_with_and(action.split(" and "))
                texts += [action] * int(num_mp)
        else:
            action, num_rollout = text_prompt.split("*")
            action = self.compose_texts_with_and(action.split(" and "))
            for _ in range(int(num_rollout)):
                texts.append(action)
        return texts

    def init_state(self) -> Dict[str, torch.Tensor]:
        device = self.device_t
        batch = self.dataset.get_batch(batch_size=self.batch_size)
        input_motions, model_kwargs = batch[0]["motion_tensor_normalized"], {"y": batch[0]}
        del model_kwargs["y"]["motion_tensor_normalized"]
        gender = "female"
        primitive_length = self.history_length + self.future_length
        betas = model_kwargs["y"]["betas"][:, :primitive_length, :].to(device)
        pelvis_delta = self.primitive_utility.calc_calibrate_offset(
            {"betas": betas[:, 0, :], "gender": gender}
        )
        input_motions = input_motions.to(device)
        motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
        history_motion_gt = motion_tensor[:, : self.history_length, :]
        transf_rotmat = (
            torch.eye(3, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
        )
        transf_transl = (
            torch.zeros(3, device=device, dtype=torch.float32)
            .reshape(1, 1, 3)
            .repeat(self.batch_size, 1, 1)
        )
        return {
            "gender": gender,
            "betas": betas,
            "pelvis_delta": pelvis_delta,
            "history_motion": history_motion_gt,
            "transf_rotmat": transf_rotmat,
            "transf_transl": transf_transl,
        }

    def generate_next_primitive(
        self,
        text_embedding: torch.Tensor,
        state: Dict[str, torch.Tensor],
        segment_id: int,
    ) -> Dict[str, torch.Tensor]:
        device = self.device_t
        batch_size = self.batch_size
        future_length = self.future_length
        history_length = self.history_length
        primitive_length = history_length + future_length
        sample_fn = self.diffusion.p_sample_loop

        history_motion = state["history_motion"]
        transf_rotmat = state["transf_rotmat"]
        transf_transl = state["transf_transl"]

        guidance_param = (
            torch.ones(batch_size, *self.denoiser_args.model_args.noise_shape)
            .to(device=device)
            * self.guidance_param
        )
        y = {
            "text_embedding": text_embedding,
            "history_motion_normalized": history_motion,
            "scale": guidance_param,
        }

        x_start_pred = sample_fn(
            self.denoiser_model,
            (batch_size, *self.denoiser_args.model_args.noise_shape),
            clip_denoised=False,
            model_kwargs={"y": y},
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        latent_pred = x_start_pred.permute(1, 0, 2)
        future_motion_pred = self.vae_model.decode(
            latent_pred,
            history_motion,
            nfuture=future_length,
            scale_latent=self.denoiser_args.rescale_latent,
        )

        future_frames = self.dataset.denormalize(future_motion_pred)
        all_frames = torch.cat(
            [self.dataset.denormalize(history_motion), future_frames], dim=1
        )

        if segment_id == 0:
            future_frames = all_frames
        future_feature_dict = self.primitive_utility.tensor_to_dict(future_frames)
        future_feature_dict.update(
            {
                "transf_rotmat": transf_rotmat,
                "transf_transl": transf_transl,
                "gender": state["gender"],
                "betas": state["betas"][:, :future_length, :]
                if segment_id > 0
                else state["betas"][:, :primitive_length, :],
                "pelvis_delta": state["pelvis_delta"],
            }
        )
        future_primitive_dict = self.primitive_utility.feature_dict_to_smpl_dict(
            future_feature_dict
        )
        future_primitive_dict = self.primitive_utility.transform_primitive_to_world(
            future_primitive_dict
        )

        new_history_frames = all_frames[:, -history_length:, :]
        history_feature_dict = self.primitive_utility.tensor_to_dict(new_history_frames)
        history_feature_dict.update(
            {
                "transf_rotmat": transf_rotmat,
                "transf_transl": transf_transl,
                "gender": state["gender"],
                "betas": state["betas"][:, :history_length, :],
                "pelvis_delta": state["pelvis_delta"],
            }
        )
        canonicalized_history_primitive_dict, blended_feature_dict = (
            self.primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=self.use_predicted_joints
            )
        )
        state["transf_rotmat"] = canonicalized_history_primitive_dict["transf_rotmat"]
        state["transf_transl"] = canonicalized_history_primitive_dict["transf_transl"]
        history_motion = self.primitive_utility.dict_to_tensor(blended_feature_dict)
        state["history_motion"] = self.dataset.normalize(history_motion)

        return {
            "transl": future_primitive_dict["transl"][0].detach().cpu(),
            "global_orient": future_primitive_dict["global_orient"][0].detach().cpu(),
            "body_pose": future_primitive_dict["body_pose"][0].detach().cpu(),
        }


class DartMotionGenerator(MotionGenerator):
    name = "dart-motion-gen"

    async def generate(
        self, text_prompt: str, duration_seconds: Optional[float], fps: float
    ) -> AsyncIterator[MotionFrame]:
        runtime = DartRuntime.get()

        duration = 4.0 if duration_seconds is None else float(duration_seconds)
        num_frames = max(1, int(duration * fps))
        num_primitives = int(math.ceil(num_frames / float(runtime.future_length)))
        if "*" not in text_prompt:
            text_prompt = f"{text_prompt}*{num_primitives}"
        texts = runtime._parse_texts(text_prompt)
        all_text_embedding = runtime.encode_text(
            runtime.dataset.clip_model, texts, force_empty_zero=True
        ).to(dtype=torch.float32, device=runtime.device_t)
        state = runtime.init_state()
        interval = 1.0 / fps
        dart_index = {name: idx for idx, name in enumerate(DART_JOINT_NAMES)}

        sent = 0
        for segment_id in range(len(texts)):
            text_embedding = all_text_embedding[segment_id].expand(runtime.batch_size, -1)
            loop = asyncio.get_running_loop()
            motion = await loop.run_in_executor(
                None, runtime.generate_next_primitive, text_embedding, state, segment_id
            )
            transl = motion["transl"]
            global_orient = motion["global_orient"]
            body_pose = motion["body_pose"]

            for local_idx in range(transl.shape[0]):
                if sent >= num_frames:
                    break
                root_rot = global_orient[local_idx]
                root_quat_wxyz = p3d_transforms.matrix_to_quaternion(root_rot)
                root_quat_xyzw = [
                    float(root_quat_wxyz[1]),
                    float(root_quat_wxyz[2]),
                    float(root_quat_wxyz[3]),
                    float(root_quat_wxyz[0]),
                ]

                joint_quats: Dict[str, List[float]] = {}
                for joint_name in TARGET_JOINT_NAMES:
                    if joint_name == "pelvis":
                        joint_quats[joint_name] = root_quat_xyzw
                        continue
                    dart_idx = dart_index[joint_name] - 1
                    rot = body_pose[local_idx, dart_idx]
                    quat_wxyz = p3d_transforms.matrix_to_quaternion(rot)
                    joint_quats[joint_name] = [
                        float(quat_wxyz[1]),
                        float(quat_wxyz[2]),
                        float(quat_wxyz[3]),
                        float(quat_wxyz[0]),
                    ]

                frame = MotionFrame(
                    index=sent,
                    timestamp=sent * interval,
                    root_translation=[
                        float(transl[local_idx, 0]),
                        float(transl[local_idx, 1]),
                        float(transl[local_idx, 2]),
                    ],
                    root_rotation=root_quat_xyzw,
                    joint_rotations=joint_quats,
                )

                yield frame
                sent += 1
                await asyncio.sleep(interval)
            if sent >= num_frames:
                break
