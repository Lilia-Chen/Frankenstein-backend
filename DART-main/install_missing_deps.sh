#!/bin/bash
# Install missing pip dependencies from original environment.yml

echo "Installing missing dependencies for DART demo..."

pip install \
  absl-py==2.1.0 \
  addict==2.4.0 \
  aiofiles==22.1.0 \
  aiohttp \
  click==8.1.7 \
  chumpy==0.70 \
  einops==0.7.0 \
  ftfy \
  gitpython \
  h5py \
  httpx \
  hydra-core==1.3.2 \
  imageio \
  ipykernel \
  ipywidgets \
  jupyter-server \
  jupyterlab-widgets \
  loralib==0.1.2 \
  lxml \
  markdown \
  matplotlib-inline \
  mesh2sdf \
  nbclassic \
  nbclient \
  nbconvert \
  notebook-shim \
  omegaconf \
  open3d==0.13.0 \
  openai \
  protobuf \
  pydantic \
  pygments \
  pyopengl \
  pyrender \
  pyzmq \
  regex \
  rich \
  scikit-image \
  shapely \
  smplx \
  spacy==2.3.4 \
  tensorboard \
  tensorboardx \
  torch-dct \
  trimesh \
  tyro \
  wandb \
  fvcore \
  iopath \
  portalocker \
  tabulate \
  termcolor \
  yacs

echo "Done!"
