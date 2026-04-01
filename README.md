# Dreambooth Implementation

This repository contains scripts and configurations for fine-tuning Stable Diffusion models using the Dreambooth method. It allows for training custom subjects (e.g., specific dogs, backpacks) utilizing both single GPU and accelerated setups.

## Repository Overview

- `train.py`: Standard Dreambooth fine-tuning script.
- `train_accelerate.py`: Distributed and mixed-precision training script utilizing Hugging Face `accelerate`.
- `inference.py`: Script used to generate images from the fine-tuned checkpoints.
- `generate_class_images.py`: Generates class images needed for prior preservation loss during training.
- `evaluate_dino.py`: Script to compute DINO score assessing the Subject Fidelity of generated images.
- `evaluate_clip.py`: Script to compute CLIP-T score validating the Prompt Fidelity.
- `config/`: Directory containing various YAML configuration files to set hyperparameters, paths, and text encoder settings (e.g., `config_dog_textencoder.yaml`, `config_backpack.yaml`).
- `src/`: Supplementary source codes and utilities.

## Setup Instructions

### Environment Setup

This project was built using a Conda environment. The dependencies have been captured for Conda natively. 

**Option 1: Using `environment.yml` (Recommended)**
You can recreate the exact Conda environment by running:
```bash
conda env create -f environment.yml
conda activate dreambooth
```

**Option 2: Using `requirements.txt`**
Alternatively, if you are setting up an existing Conda environment or standard pip environment, use the provided `requirements.txt`:
```bash
conda create -n dreambooth --file requirements.txt
# OR using pip
pip install -r requirements.txt
```

## Usage

### 1. Training

Choose the appropriate YAML configuration file in the `config/` directory to specify your `instance_data_dir`, `class_data_dir`, learning rate, and whether you want to train the text encoder (`train_text_encoder: true`).

**Standard Run:**
```bash
python train.py --config config/config.yaml
```

**Accelerated Run (Recommended):**
```bash
accelerate launch train_accelerate.py --config config/config.yaml
```

### 2. Inference

Once training is complete, the model checkpoints are saved in `output_model/`. Use the inference script to generate images:
```bash
python inference.py
```

### 3. Evaluation

Assess the quality of the fine-tuned model by running the separate evaluation scripts:
```bash
# Evaluate Subject Fidelity (DINO Score)
python evaluate_dino.py

# Evaluate Prompt Fidelity (CLIP Score)
python evaluate_clip.py
```

## Contributing

Make sure not to commit large generated checkpoint files (`*.ckpt`, `*.safetensors`), virtual environments (`env/`), or the dataset folders as they are listed in `.gitignore`.
