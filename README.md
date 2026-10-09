# Run Kohya_SS SD-SCRIPTS on Modal - Cloud-Based LoRA Training

Modified version of [IjoiK/modal-deploy-kohya-ss](https://github.com/IjoiK/modal-deploy-kohya-ss) to run on a more recent version of [Kohya-ss' sd-scripts](https://github.com/kohya-ss/sd-scripts).

## Prerequisites

Before you begin, ensure you have the following:

1.  **A Modal Account:** Sign up at [modal.com](https://modal.com). New users often receive free credits.
2.  **Modal Client Installed and Configured:**
    ```bash
    pip install modal
    modal setup
    modal token info
    ```
3.  **Python:** Python 3.10 or newer installed locally.
4.  **Git:** For cloning this repository.
5.  **Training Data:** Your base models (e.g., `.safetensors` files) and image datasets prepared for Kohya_SS.

## Setup Instructions

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/jbareaud/modal-deploy-kohya-sd-scripts
    cd modal-deploy-kohya-sd-scripts
    ```

2.  **Configure `config.toml`:**
    Create a file named `config.toml` in the root of the cloned repository with the following content, adjusting parameters as needed:

    ```toml
    [modal_settings]
    container_idle_timeout = 600  # Idle time in seconds before container scales down (used for scaledown_window)
    timeout = 7200                # Max container lifetime in seconds (e.g., 2 hours)
    cpu = 2
    gpu = "A10G"                  # GPU type: "A10G", "T4", "L4", "A100", "H100"
    memory = 10240                # Amount of RAM reserved
    ```
    * `gpu`: Choose based on your needs. `A10G` (24GB VRAM) is a good starting point for many tasks. For SDXL or larger batches, consider `A100` (40GB or 80GB) or `H100` (80GB).
    * `container_idle_timeout`: This is used for `scaledown_window`. 600 seconds = 10 minutes. If `min_containers` is 0 (default or not set in `app.py`), the container will stop after this period of inactivity.
    * `timeout`: Maximum duration a container can run. Adjust if you expect very long training sessions.

3.  **Review `app.py`:**
    The provided `app.py` contains the Modal application definition, including the image build process and runtime function. It is configured based on extensive debugging to ensure a stable environment for Kohya_SS.

## Uploading Data (Models & Datasets)

Your models, datasets, and outputs will be stored in persistent Modal Volumes. You need to upload your base models and datasets to these volumes using the `modal volume put` command from your local terminal.

The `app.py` script maps these volumes to paths inside the container:
* `kohya-models` volume is mounted at `/kohya_ss/models/`
* `kohya-dataset` volume is mounted at `/kohya_ss/dataset/` (Note: singular "dataset" in the path as per your `app.py`)
* `kohya-outputs` volume is mounted at `/kohya_ss/outputs/`
* `kohya-configs` volume is mounted at `/kohya_ss/configs/`

**Uploading Base Models:**
   * Volume Name: `kohya-models`
   * Example: 
        ```bash
        modal volume put kohya-models /mnt/ssd1/StableDiffusion/models/text_encoders/qwen_3_06b_base.safetensors /qwen_3_06b_base.safetensors
        modal volume put kohya-models /mnt/ssd1/StableDiffusion/models/diffusion_models/anima_baseV10.safetensors /anima_baseV10.safetensors
        modal volume put kohya-models /mnt/ssd1/StableDiffusion/models/vae/qwen_image_vae.safetensors /qwen_image_vae.safetensors
        ```

**Uploading Datasets:**
   * Volume Name: `kohya-dataset`
   * Example: 
        ```bash
        modal volume put kohya-dataset /mnt/ssd1/StableDiffusion/lora/MyChara/in/5_MyChara /5_MyChara
        ```

**Uploading train config:**
   * Volume Name: `kohya-configs`
   * Example: 
        ```bash
        modal volume put kohya-configs -f train_config.toml train_config.toml
        modal volume put kohya-configs -f dataset_config.toml dataset_config.toml 
        ```

**Verifying Volume Contents:**
   * You can list the contents of your volumes:
        ```bash
        modal volume ls kohya-models -r
        modal volume ls kohya-dataset -r
        ```

## Running the Application

```bash
modal deploy app.py
```
This deploys the application to Modal, where it will run in the background and be accessible via a persistent URL. You can close your terminal. To update the deployment after code changes, run this command again.

Once models/dataset/configs have been copied in place :

```bash
modal run --detach app.py::train_anima
```

## Using the Kohya_SS with Modal paths

**When specifying paths in the GUI, use the paths *inside the container*:**
    * **Pretrained model name or path:** `/kohya_ss/models/your_model_name.safetensors`
    * **Image folder (Dataset directory):** `/kohya_ss/dataset/` (Kohya will then look for your `Repeats_InstanceToken` subfolders inside this path).
    * **Output folder:** `/kohya_ss/outputs/`
    * **Logging folder:** `/kohya_ss/outputs/logs` (or your preference within `/kohya_ss/outputs/`)
    * **LoRA model output name:** (e.g., `my_awesome_lora`)

## Downloading Results

Your trained models (LoRA files, etc.) will be saved to the `/kohya_ss/outputs/` directory within the `kohya-outputs` volume. Use `modal volume get` to download them:

```bash
modal volume get kohya-outputs /models /mnt/ssd1/StableDiffusion/lora/MyChara/release/v5/models
modal volume get kohya-outputs /log /mnt/ssd1/StableDiffusion/lora/MyChara/release/v5/log
```
