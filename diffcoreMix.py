# augmentor/dataset_augmentor.py
import os
import time
import random
import torch
from tqdm import tqdm
from clip_feature import CLIPFeatureExtractor
from utils import format_time

class DiffCoreMix:
    def __init__(self, output_folder, diffusion_generator, clip_extractor, prompts, dataset_name, augment_percentage=0.3):
        self.output_folder = output_folder
        self.diffusion_generator = diffusion_generator
        self.clip_extractor = clip_extractor
        self.prompts = prompts
        self.dataset_name = dataset_name
        self.augment_percentage = augment_percentage

    def augment_dataset(self):
        start_time = time.time()
        total_images = 0
        processed_images = 0
        discarded_images = 0

        # Create results folder if it doesn't exist
        results_folder = os.path.join(os.getcwd(), "results")
        os.makedirs(results_folder, exist_ok=True)
        log_file_name = f"{self.dataset_name}_{self.augment_percentage}.txt"
        log_file_path = os.path.join(results_folder, log_file_name)

        with open(log_file_path, 'w') as log_file:
            for class_name in os.listdir(self.output_folder):
                class_output_path = os.path.join(self.output_folder, class_name)
                if os.path.isdir(class_output_path) and class_name != "results":
                    # For non-aircraft datasets, clean the class name
                    if self.dataset_name != "aircraft":
                        clean_name = self.clip_extractor.clean_label_name(class_name)
                    else:
                        clean_name = class_name

                    # Count existing images in the class folder
                    image_filenames = [f for f in os.listdir(class_output_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    num_existing_images = len(image_filenames)
                    generate_count = int(num_existing_images * self.augment_percentage)
                    total_images += generate_count

                    log_file.write(f"Class '{class_name}' ({clean_name}): {num_existing_images} images found. Generating {generate_count} more images.\n")
                    print(f"Class '{class_name}' ({clean_name}): {num_existing_images} images found. Generating {generate_count} more images.")

                    for _ in tqdm(range(generate_count), desc=f"Generating images for class {class_name}"):
                        prompt = random.choice(self.prompts).replace("{label_name}", clean_name).replace("{dataset_type}", self.dataset_name)
                        print(prompt)
                        try:
                            output_image_name = f"{clean_name}_gen_{processed_images}.png"
                            output_image_path = os.path.join(class_output_path, output_image_name)

                            # Generate new image using the diffusion generator
                            generated_image = self.diffusion_generator.generate_image(prompt)

                            # Extract features from an existing original image in the folder
                            original_image_path = os.path.join(class_output_path, image_filenames[0])
                            features_original = self.clip_extractor.extract_features(original_image_path)

                            # Save generated image temporarily to extract its features
                            temp_generated_image_path = os.path.join(class_output_path, f"temp_{output_image_name}")
                            generated_image.save(temp_generated_image_path)
                            features_generated = self.clip_extractor.extract_features(temp_generated_image_path)

                            # Calculate cosine similarity
                            similarity = self.clip_extractor.calculate_cosine_similarity(features_original, features_generated)
                            print(f"Cosine Similarity: {similarity}")

                            # Save image only if similarity is above threshold
                            if similarity > 0.6:
                                os.rename(temp_generated_image_path, output_image_path)
                                print(f"Saved: {output_image_path}")
                            else:
                                os.remove(temp_generated_image_path)
                                discarded_images += 1
                                print(f"Discarded image due to low similarity ({similarity}).")
                        except Exception as e:
                            print(f"Error processing prompt '{prompt}' for class '{class_name}': {e}")

                        torch.cuda.empty_cache()
                        processed_images += 1
                        elapsed_time = time.time() - start_time
                        avg_time_per_image = elapsed_time / processed_images
                        time_left = avg_time_per_image * (total_images - processed_images)
                        print(f"[Time Left] {format_time(time_left)}")

            end_time = time.time()
            total_time = end_time - start_time
            log_file.write(f"\nTotal images generated: {processed_images}\n")
            log_file.write(f"Total discarded images due to low similarity: {discarded_images}\n")
            log_file.write(f"Augmentation percentage: {self.augment_percentage}\n")
            log_file.write(f"Total time taken to generate all images: {format_time(total_time)}\n")

        print(f"Total time taken to generate all images: {format_time(total_time)}")
        print(f"Log saved to: {log_file_path}")
