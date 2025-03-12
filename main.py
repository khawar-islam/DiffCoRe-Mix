import argparse
import torch
import clip
import os
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, AutoencoderKL
from clip_feature import CLIPFeatureExtractor
from similarity_check import SimilarityChecker
from model import CosXL
from prompts import PromptsNegPos
from diffcoreMix import DiffCoreMix


class DataAugmentation:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load VAE
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)

        # Initialize components
        print("Initializing CLIP Feature Extractor...")
        self.extractor = CLIPFeatureExtractor(self.device)
        self.similarity_checker = SimilarityChecker()

        # Validate CosXL model path
        if not os.path.exists(args.cosxl_model_path):
            raise FileNotFoundError(f"Model file not found: {args.cosxl_model_path}")

        print("Loading CosXL model...")
        self.generator = CosXL(args.cosxl_model_path, self.vae, device=self.device)

        print("Initializing Prompt Manager...")
        self.prompt_manager = PromptsNegPos()

        print("Initializing Augmentor...")
        self.augmentor = DiffCoreMix(self.extractor, self.similarity_checker, self.generator, self.prompt_manager,
                                     args.dataset)

    def run(self, output_folder, augment_percentage):
        """Runs the augmentation process."""
        if not os.path.exists(output_folder):
            print(f"Creating output folder: {output_folder}")
            os.makedirs(output_folder, exist_ok=True)

        print(f"Starting augmentation in {output_folder} with {augment_percentage * 100:.1f}% extra images.")

        # ✅ Provide a log file or None if logging is optional
        self.augmentor.augment_folder(output_folder, augment_percentage, log_file="augmentation_log.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Augmentation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name for augmentation")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save augmented images")
    parser.add_argument("--augment_percentage", type=float, default=0.3, help="Percentage of images to augment")
    parser.add_argument("--cosxl_model_path", type=str, required=True, help="Path to the CosXL model file")

    args = parser.parse_args()

    print("Initializing Data Augmentation Pipeline...")
    pipeline = DataAugmentation(args)

    print("Running Data Augmentation...")
    pipeline.run(args.output_folder, args.augment_percentage)  # ✅ Ensure run() is called

    print("Data Augmentation Completed.")
