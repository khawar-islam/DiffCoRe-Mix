# models/clip_extractor.py
import re
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

class CLIPFeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.cpu().numpy()

    @staticmethod
    def calculate_cosine_similarity(features1, features2):
        return cosine_similarity(features1, features2)[0][0]

    @staticmethod
    def clean_label_name(class_name):
        label_name = re.sub(r'^\d+\.', '', class_name)  # Remove digits and dot at the start
        label_name = label_name.replace('_', ' ')  # Replace underscores with spaces
        return label_name
