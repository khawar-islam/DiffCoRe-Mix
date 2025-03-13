# models/diffusion_generator.py
import torch
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler

class CosXL:
    def __init__(self, model_path, vae, device="cuda"):
        self.device = device
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            vae=vae,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = EDMEulerScheduler(
            sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
            prediction_type="v_prediction", sigma_schedule="exponential"
        )
        self.pipe.to(self.device)

    def generate_image(self, prompt, negative_prompt="train, car, newspaper, sunlight, people, human, tree, cityscape, road, landscape, bus, building, desk, computer, paper texture, windows, streets, rails, traffic, sky, road lines, traffic lights",
                       guidance_scale=12, steps=50, height=512, width=512):
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            height=height,
            width=width
        )
        return output.images[0]
