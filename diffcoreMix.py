import argparse
import os
import re
import random
import torch
import clip
from tqdm import tqdm



class DiffCoreMix:
    def __init__(self, extractor, similarity_checker, generator, prompt_manager, dataset_name, similarity_threshold=0.6):
        self.extractor = extractor
        self.similarity_checker = similarity_checker
        self.generator = generator
        self.prompt_manager = prompt_manager
        self.dataset_name = dataset_name
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def clean_label_name(class_name):
        return re.sub(r'^\d+\.', '', class_name).strip()

    def augment_folder(self, folder, augment_percentage, log_file):
        processed_images = 0
        discarded_images = 0

        class_names = [cls for cls in os.listdir(folder) if os.path.isdir(os.path.join(folder, cls))]

        for class_name in class_names:
            class_path = os.path.join(folder, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            num_to_generate = int(len(image_files) * augment_percentage)
            clean_class_name = self.clean_label_name(class_name)

            for idx in tqdm(range(num_to_generate), desc=f"Generating {class_name}"):
                prompt = self.prompt_manager.get_random_prompt(clean_class_name, self.dataset_name)
                generated_image = self.generator.generate_image(
                    prompt=prompt,
                    negative_prompt=self.prompt_manager.negative_prompt,
                    guidance_scale=7.5,
                    steps=50,
                    height=512,
                    width=512
                )

                original_image_path = os.path.join(class_path, random.choice(image_files))
                original_features = self.extractor.extract_features(original_image_path)

                generated_image.save("temp.png")
                generated_features = self.extractor.extract_features("temp.png")

                similarity = self.similarity_checker.cosine_similarity(original_features, generated_features)
                if similarity >= self.similarity_threshold:
                    output_image_name = f"{class_name}_gen_{processed_images}.png"
                    generated_image.save(os.path.join(class_path, output_image_name))
                    processed_images += 1
                else:
                    discarded_images += 1
                os.remove("temp.png")

        return processed_images, discarded_images