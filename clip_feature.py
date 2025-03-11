import torch
import clip
from PIL import Image


class CLIPFeatureExtractor:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def extract_features(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
        return features.cpu().numpy()