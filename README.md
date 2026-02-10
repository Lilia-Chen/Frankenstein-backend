# Frankenstein-backend
A motion gen demo backend.

## Run (local venv)
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --host 0.0.0.0 --port 8000`

WebSocket endpoint: `ws://<host>:8000/ws/motion`

## Run (DART conda env)
1. `conda env create -f DART-main/environment.yml`
2. `conda activate DART`
3. Set env vars:
   - `MOTION_MODEL=dart`
   - `DART_ROOT=DART-main`
   - `DART_DENOISER_CKPT=DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt`
   - `DART_DEVICE=cuda`
4. `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Run (docker-compose)
1. `docker compose up --build`
2. WebSocket endpoint: `ws://<host>:8000/ws/motion`
