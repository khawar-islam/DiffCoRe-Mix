import torch
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler


class CosXL:
    def __init__(self, model_path, vae, scheduler=EDMEulerScheduler, device="cuda"):
        """Initializes Stable Diffusion XL pipeline."""
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load Stable Diffusion XL model from the given path
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            pretrained_model_link_or_path=model_path,  # ✅ Fixed argument
            torch_dtype=torch.float16,
            vae=vae,
        )

        # Assign scheduler
        self.pipe.scheduler = scheduler()

        # Move model to device
        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()  # Reduces memory usage
        self.pipe.enable_model_cpu_offload()  # Optimizes memory on GPU

    def generate_image(self, prompt, negative_prompt, guidance_scale=7.5, steps=50, height=512, width=512):
        """Generates an image based on the given prompt and parameters."""
        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                height=height,
                width=width
            ).images[0]
            return image
        except Exception as e:
            print(f"Error during image generation: {e}")
            return None
