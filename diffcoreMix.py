import os
import re
import random
import torch
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

        # ✅ Automatically set dataset_type to "bird" if dataset_name is "cub200"
        dataset_type = "bird" if self.dataset_name == "cub200" else self.dataset_name

        for class_name in class_names:
            class_path = os.path.join(folder, class_name)  # ✅ Ensure correct class path
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            num_to_generate = int(len(image_files) * augment_percentage)
            clean_class_name = self.clean_label_name(class_name)

            for idx in tqdm(range(num_to_generate), desc=f"Generating images for {class_name}"):
                # ✅ Use the correct dataset_type
                prompt = self.prompt_manager.get_random_prompt(clean_class_name, dataset_type)

                # Generate the image
                generated_image = self.generator.generate_image(
                    prompt=prompt,
                    negative_prompt=self.prompt_manager.negative_prompt,
                    guidance_scale=12,
                    steps=100,
                    height=512,
                    width=512
                )

                # ✅ Check if image was successfully generated
                if generated_image is None:
                    print(f"Skipping {class_name}_gen_{processed_images}: Image generation failed")
                    discarded_images += 1
                    continue  # Skip this iteration

                # Construct the correct save path
                output_image_name = f"{class_name}_gen_{processed_images}.png"
                output_image_path = os.path.join(class_path, output_image_name)

                # ✅ Save image in the correct class folder
                generated_image.save(output_image_path)
                print(f"Saved {output_image_name} in {class_path}")

                processed_images += 1

        return processed_images, discarded_images
