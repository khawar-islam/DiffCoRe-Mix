# main.py
import argparse
import torch
from diffusers import AutoencoderKL
from model import CosXL
from clip_feature import CLIPFeatureExtractor
from diffcoreMix import DiffCoreMix

def main():
    parser = argparse.ArgumentParser(description="Image generation script for multiple datasets.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., cub200, cars, flower102, aircraft)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the folder where generated images will be saved")
    parser.add_argument("--aug_per", type=float, default=0.3, help="Percentage of images to augment (default: 0.3)")
    args = parser.parse_args()

    # Load the VAE model
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Path to the diffusion model file (update with the actual path)
    generate_file = "/media/cvpr/CM_1/ResMix/model/cosxl.safetensors"

    # Initialize the DiffusionGenerator
    diffusion_generator = CosXL(generate_file, vae, device="cuda")

    # Initialize the CLIPFeatureExtractor
    clip_extractor = CLIPFeatureExtractor(device="cuda")

    # Define the contextual prompts
    contextual_prompts = [
        "A {label_name}, a {dataset_type}, in the garden",
        "A {label_name}, a {dataset_type}, soaring through a stormy sky with dark clouds.",
        "A {label_name}, a {dataset_type}, gliding over calm water under a clear blue sky.",
        "A {label_name}, a {dataset_type}, framed against a vibrant sunset with orange and pink hues.",
        "A {label_name}, a {dataset_type}, flying through a misty morning with soft, diffused light.",
        "A {label_name}, a {dataset_type}, navigating between tall, towering mountain peaks.",
        "A {label_name}, a {dataset_type}, flying low over a dense forest canopy.",
        "A {label_name}, a {dataset_type}, speeding through a cloudless sky at high altitude.",
        "A {label_name}, a {dataset_type}, passing through a vast, open desert landscape.",
        "A {label_name}, a {dataset_type}, flying in the night sky, lit by a full moon.",
        "A {label_name}, a {dataset_type}, surrounded by lightning in a turbulent storm.",
        "A {label_name}, a {dataset_type}, casting a shadow over an expansive ocean.",
        "A {label_name}, a {dataset_type}, flying close to a city skyline during twilight.",
        "A {label_name}, a {dataset_type}, cruising above fields of golden crops under the afternoon sun.",
        "A {label_name}, a {dataset_type}, flying through swirling snowflakes in a winter landscape.",
        "A {label_name}, a {dataset_type}, cutting through thick clouds in an overcast sky."
    ]

    # Initialize and run the dataset augmentor
    augmentor = DiffCoreMix(
        output_folder=args.output_folder,
        diffusion_generator=diffusion_generator,
        clip_extractor=clip_extractor,
        prompts=contextual_prompts,
        dataset_name=args.dataset,
        augment_percentage=args.aug_per
    )
    augmentor.augment_dataset()

if __name__ == "__main__":
    main()
