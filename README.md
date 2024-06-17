This is the code for a paper titiled "Diffusion model-based image generation from rat brain activity".

# BrainDiffusion




https://github.com/UT-yakusaku/BrainDiffusion/assets/34576921/c4e90dd7-ec2a-4032-8d8b-0cf1878b1a2a




## Setup
1. Copy and place the `ldm` directory from [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) into the main directory.
2. Download and place the desired diffusion model in the `models` directory. Update the `path.model` in `configs/main_cfg.yaml` accordingly. For example, you could use the stable-diffusion-v1-5 model from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5).

## Prerequisites
To install the required libraries, execute the following command:
```sh
pip install -r requirements.txt
```

## Offline Generation of Paintings from Neural Activity
To generate images from neural activity data, execute the following command:
```sh
python convert_lfp.py
```
This script reads `data/rat_lfp.npy` and generates images from the signal. To use your own data, update the `path.data` in `configs/main_cfg.yaml`.

## Online Generation of Paintings from Neural Activity
To generate images online, use the `generate` function in `utils.py`.

