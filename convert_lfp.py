import mne
import numpy as np
import torch

from omegaconf import OmegaConf

from utils import load_model, generate

def main(cfg):
    # load diffusion model
    model = load_model(
        cfg=cfg.path.inference_config,
        ckpt=cfg.path.model,
        is_safetensor=True)

    # load neuroral activity data
    data = np.load(cfg.path.data)

    # preprocess data
    bin_len = 512
    steps = 5

    for i in range(data.shape[0]):
        if data[i:i+512].shape[0] == bin_len:
            data_shifted = np.array(data[i*steps:i*steps+512]).reshape(-1)

            img = generate(data_shifted, model, ddim_steps=5)
            img.save(f"./outputs/image_{i:04d}.png")

if __name__ == "__main__":
    cfg = OmegaConf.load("./configs/main_cfg.yaml")
    main(cfg)