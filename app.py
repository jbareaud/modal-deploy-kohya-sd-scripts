import modal
import subprocess
import toml
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYTHON_VERSION = "3.10"
KOHYA_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"
KOHYA_COMMIT = "690ea7f96c23182352ec63def76d431c6120bd2f" # https://github.com/kohya-ss/sd-scripts/releases/tag/v0.12.0
#NVIDIA_CUDA_IMAGE = "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
NVIDIA_CUDA_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"


kohya_image = (
    modal.Image.from_registry(
        NVIDIA_CUDA_IMAGE, add_python=PYTHON_VERSION
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "TZ": "Etc/UTC",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128" # Example, you can use: 64 or 32 also
    })
    .apt_install(
        "git",
        "wget",
        "libgl1",
        "libglib2.0-0",
        "python3-tk",
        "libjpeg-dev",
        "libpng-dev",
        "google-perftools",
    )
    #.env({"KOHYA_VERSION_DATE": "2025-05-25"})
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        "set -ex",
        "pip install --upgrade pip",
        f"git clone --recursive {KOHYA_REPO_URL} /kohya_ss",
        f"cd /kohya_ss && git checkout {KOHYA_COMMIT}",
        gpu="any",
    )
    .workdir("/kohya_ss")
    .run_commands(
        "set -ex",

        "echo 'Install torch...'",
        "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",

        "echo 'Install requirements...'",
        "pip install --upgrade -r requirements.txt",

        "echo 'Install xformers...'",
        "pip install xformers --index-url https://download.pytorch.org/whl/cu124",

        "echo 'Install accelerate...'",
        "pip install accelerate",

        "echo 'accelerate config default'",
        "accelerate config default",

        "echo 'Clean up existing run files if exists...'",
        "rm -rf models dataset outputs configs",
        gpu="any",
    )
    .run_commands(
        "echo 'Kohya_SS ready.'",
    )
)

CONFIG_FILE = Path(__file__).parent / "config.toml"

try:
    logger.info("Load config.toml")
    config = toml.load(CONFIG_FILE)
    modal_settings = config.get('modal_settings', {})
    ALLOW_CONCURRENT_INPUTS = modal_settings.get('allow_concurrent_inputs', 10)
    logger.info(f"ALLOW_CONCURRENT_INPUTS: {ALLOW_CONCURRENT_INPUTS}")
    CONTAINER_IDLE_TIMEOUT = modal_settings.get('container_idle_timeout', 600)
    logger.info(f"CONTAINER_IDLE_TIMEOUT: {CONTAINER_IDLE_TIMEOUT}")
    TIMEOUT = modal_settings.get('timeout', 3600)
    logger.info(f"TIMEOUT: {TIMEOUT}")
    CPU_CONFIG = config.get('cpu', 2)
    logger.info(f"CPU_CONFIG: {CPU_CONFIG}")
    GPU_CONFIG = modal_settings.get('gpu', "A10G")
    logger.info(f"GPU_CONFIG: {GPU_CONFIG}")
    MEMORY_CONFIG = config.get('memory', 10240)
    logger.info(f"MEMORY_CONFIG: {MEMORY_CONFIG}")
except Exception as e:
    logger.info("Loading config.toml failed, using default settings")
    ALLOW_CONCURRENT_INPUTS = 5
    CONTAINER_IDLE_TIMEOUT = 300
    TIMEOUT = 1800
    CPU_CONFIG = 2
    GPU_CONFIG = "A10G"
    MEMORY_CONFIG = 10240

app = modal.App(name="kohya-ss-sd-scripts", image=kohya_image)

class Paths:
    CACHE = "/cache"
    KOHYA_BASE = "/kohya_ss"
    MODELS = "/kohya_ss/models"
    DATASET = "/kohya_ss/dataset"
    OUTPUTS = "/kohya_ss/outputs"
    CONFIGS = "/kohya_ss/configs"

cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("kohya-models", create_if_missing=True)
dataset_vol = modal.Volume.from_name("kohya-dataset", create_if_missing=True)
outputs_vol = modal.Volume.from_name("kohya-outputs", create_if_missing=True)
configs_vol = modal.Volume.from_name("kohya-configs", create_if_missing=True)

@app.function(
    memory=MEMORY_CONFIG,
    cpu=CPU_CONFIG,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    volumes={
        Paths.MODELS: models_vol,
        Paths.DATASET: dataset_vol,
        Paths.OUTPUTS: outputs_vol,
        Paths.CONFIGS: configs_vol,
    },
    max_containers=1,
    retries=0,
)

def train_anima():
    import torch
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Pytorch version: {torch.__version__}")

    subprocess.run(
    [
        "accelerate",
        "launch",
        "anima_train_network.py",
        "--config_file",
        #"/kohya_ss/train_toml/train_config.toml",
        f"{Paths.CONFIGS}/train_config.toml",
    ],
    cwd="/kohya_ss",
    check=True,
    )
    logger.info("Training complete. Exiting container.")

@app.local_entrypoint()
def main():
    #call = train_anima.spawn()
    #print(f"Job ID: {call.object_id}")
    train_anima.remote()
