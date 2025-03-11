import random

class PromptsNegPos:
    def __init__(self):
        self.positive_prompts = [
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

        # It will be aligned according to the dataset classes
        self.negative_prompt = "train, car, newspaper, sunlight, people, human, tree, cityscape, road, landscape, bus, building, desk, computer, paper texture, windows, streets, rails, traffic, sky, road lines, traffic lights"

    def get_random_prompt(self, label_name, dataset_type):
        return random.choice(self.positive_prompts).format(label_name=label_name, dataset_type=dataset_type)