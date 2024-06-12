# BrainDiffusion


https://github.com/UT-yakusaku/BrainDiffusion/assets/34576921/44785ee4-45b6-4b41-bed0-068f79a82b7a



## Setup
Copy and place `ldm` directory from https://github.com/CompVis/stable-diffusion in the main directory. </br>
Download and place the desired diffusion model to the models directory. Change the `path.model` in `configs/main_cfg.yaml`. </br>
For example, you could use stable-diffusion-v1-5 model from https://huggingface.co/runwayml/stable-diffusion-v1-5. </br>

## prerequisites
Execute following code to install required libraries. </br>
`pip install -r requirements.txt`

## Offline generation of paintings from neural activity
Execute following code. </br>
`python convert_lfp.py`
This will read `data/rat_lfp.npy` and generate images from the singal.
To use your own data, change the `path.data` in `configs/main_cfg.yaml`. </br>

## Online generation of paintings from neural activity
Use function `generate` in `utils.py`.  </br>

