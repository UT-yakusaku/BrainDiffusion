from contextlib import nullcontext

import numpy as np
import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image

from safetensors.torch import load_file

def load_model(
    cfg, 
    ckpt,
    is_safetensor=False,
    verbose=False
):
    # load config 
    config = OmegaConf.load(cfg)

    # load state dict
    if is_safetensor:
        sd = load_file(ckpt)
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]

    # load model
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # move to device
    model.cuda()
    model.eval()

    return model


def get_device() -> torch.device:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


def diffuse(
    model: LatentDiffusion,
    c: torch.Tensor,
    start_code: torch.Tensor,
    ddim_steps: int,
) -> Image:
    batch_size = 1
    device = get_device()
    precision_scope = torch.autocast if device.type == "cuda" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = model.get_learned_conditioning(batch_size * [""])
                shape = [4, 64, 64]
                samples_ddim, _ = DDIMSampler(model).sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=7.5,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=start_code,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                image = (
                    255.0 * x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()[0]
                ).astype(np.uint8)
                return Image.fromarray(image)
            
def generate(
        latent_tensor: np.ndarray,
        model: LatentDiffusion,
        ddim_steps: int = 50,
    ) -> Image:
    torch.manual_seed(0)

    # standarize
    latent_tensor = (latent_tensor - np.mean(latent_tensor))/np.std(latent_tensor)
    c = model.get_learned_conditioning("")

    np.random.seed(34)
    np.random.shuffle(latent_tensor)

    code = torch.from_numpy(latent_tensor.astype(np.float32)).to(device=get_device())
    code = code.view(1, 4, 64, 64)

    img = diffuse(model, c, code, ddim_steps=ddim_steps)

    return img