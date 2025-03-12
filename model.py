import torch
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler

class CosXL:
    def __init__(self, model_path, vae, scheduler=EDMEulerScheduler, device="cuda"):

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            vae=vae,  # ✅ Using custom VAE
            torch_dtype=torch.float16,  # ✅ Ensuring consistent FP16 precision
        )

        self.pipe.scheduler = EDMEulerScheduler(
            sigma_min=0.002, sigma_max=120.0, sigma_data=1.0,
            prediction_type="v_prediction", sigma_schedule="exponential"
        )

        self.pipe.to("cuda")  # ✅ Move the model to GPU

    def generate_image(self, prompt, negative_prompt, guidance_scale=12, steps=100, height=512, width=512):
        """Generates an image based on the given prompt and parameters."""
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt is invalid or empty!")

            print(f"Generating image with prompt: {prompt}")  # Debugging info

            # ✅ Convert inputs to match model's dtype
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,  # Convert to correct dtype
                num_inference_steps=steps,
                height=height,
                width=width
            )

            if not output or not output.images or len(output.images) == 0:
                raise RuntimeError("Stable Diffusion pipeline did not return any images!")

            return output.images[0]  # ✅ Return the generated image

        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory! Reducing memory usage...")
            torch.cuda.empty_cache()
            return None  # Return None if OOM occurs

        except Exception as e:
            print(f"Error during image generation: {e}")
            return None  # Return None if any unexpected error occurs
