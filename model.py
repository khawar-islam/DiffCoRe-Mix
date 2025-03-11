import torch
import clip
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler



class CosXL:
    def __init__(self, model_path, vae, scheduler=EDMEulerScheduler, device="cuda"):
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path=model_path,
            vae=vae,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = EDMEulerScheduler()
        self.pipe.to(device)

    def generate_image(self, prompt, negative_prompt, guidance_scale, steps, height, width):
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            height=height,
            width=width
        ).images[0]
