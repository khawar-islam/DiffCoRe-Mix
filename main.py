import argparse
import torch
import clip
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, AutoencoderKL
from clip_feature import CLIPFeatureExtractor
from similarity_check import SimilarityChecker
from model import CosXL
from prompts import PromptsNegPos
from diffcoreMix import DiffCoreMix

class DataAugmentation:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
        self.extractor = CLIPFeatureExtractor(self.device)
        self.similarity_checker = SimilarityChecker()
        self.generator = CosXL(args.cosxl_model_path, self.vae)
        self.prompt_manager = PromptsNegPos()
        self.augmentor = DiffCoreMix(self.extractor, self.similarity_checker, self.generator, self.prompt_manager, args.dataset)

    def run(self, output_folder, augment_percentage):
        self.augmentor.augment_images(output_folder, augment_percentage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Augmentation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--augment_percentage", type=float, default=0.3)
    parser.add_argument("--cosxl_model_path", type=str, required=True, help="Path to the CosXL model file")
    args = parser.parse_args()

    pipeline = DataAugmentation(args)