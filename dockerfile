# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/dgl:24.07-py3

# Use bash so we can compute versions
SHELL ["/bin/bash", "-lc"]

# Install system deps for PyVista + Xvfb (notebook viz)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       xvfb \
       && rm -rf /var/lib/apt/lists/*

# Compute TORCH and CUDA variables, install PyG, DGL, and the rest
RUN TORCH_VER=$(python3 -c 'import torch; print(torch.__version__.split("+")[0])') && \
    CUDA_VER=$(python3 -c 'import torch; print("cu" + torch.version.cuda.replace(".",""))') && \
    echo "Installing for torch==$TORCH_VER and cuda==$CUDA_VER" && \
    pip install --no-cache-dir \
      torch-scatter     -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_VER}.html \
      torch-sparse      -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_VER}.html \
      torch-cluster     -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_VER}.html \
      torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_VER}.html \
      torch-geometric && \
    pip install --no-cache-dir \
      loguru==0.7.2 \
      autoflake==2.3.0 \
      pytest==8.0.1 \
      meshio==5.3.5 \
      h5py==3.10.0 && \
    pip install --no-cache-dir \
      pyvista \
      panel \
      "lightning==2.5.0" \
      pytorch-lightning==2.5.0 \
      torchmetrics==1.6.3 \
      wandb[media]
# use --no-install-recommends to keep the image lean
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libosmesa6 libosmesa6-dev \
      libegl1-mesa-dev libegl-mesa0 libglx-mesa0 \
      libxt-dev mesa-utils \
 && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
  && apt-get install -y libxrender1 
# Make a working dir
WORKDIR /app

# Xvfb env for notebooks
# ENV DISPLAY=:99

# Convenient entrypoint: start Xvfb in background
# ENTRYPOINT Xvfb :99 -screen 0 1024x768x24 & exec "$@"
