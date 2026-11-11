
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset" / "cifake"
GRAPH_DIR = BASE_DIR / "graphs"

# Number of images to use from each class
TRAIN_PER_CLASS = 5000
TEST_PER_CLASS = 1000

# Download/load CIFAKE dataset
print("Loading CIFAKE dataset...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)

# Original labels: 0 = FAKE, 1 = REAL
# Our labels: 0 = REAL, 1 = AI-generated

label_mapping = {
    1: ("real", 0),
    0: ("ai_generated", 1)
}


def save_images(split_name, max_per_class):

    split_data = dataset[split_name]

    for original_label, (class_name, _) in label_mapping.items():

        print(f"\nProcessing {split_name}: {class_name}")

        # Select images belonging to this class
        class_data = split_data.filter(
            lambda row: row["label"] == original_label
        )

        # Shuffle to obtain a random subset
        class_data = class_data.shuffle(seed=42)

        # Select the required number of images
        selected_data = class_data.select(
            range(max_per_class)
        )

        # Create destination folder
        output_dir = DATA_DIR / split_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save images
        for i, sample in enumerate(
            tqdm(selected_data, desc=class_name)
        ):

            image = sample["image"].convert("RGB")

            image.save(output_dir / f"{class_name}_{i}.png")

        print(f"Saved {len(selected_data)} {class_name} images.")


# Download and save training and testing subsets
save_images("train", TRAIN_PER_CLASS)
save_images("test", TEST_PER_CLASS)


# Display dataset information
print("\n========== DATASET SUMMARY ==========")

for split_name in ["train", "test"]:

    print(f"\n{split_name.upper()}")

    for class_name in ["real", "ai_generated"]:

        folder = DATA_DIR / split_name / class_name
        image_count = len(list(folder.glob("*.png")))

        print(f"{class_name}: {image_count} images")


# Display sample images
fig, axes = plt.subplots(2, 4, figsize=(10, 5))

classes = ["real", "ai_generated"]

for row, class_name in enumerate(classes):

    folder = DATA_DIR / "train" / class_name
    image_paths = sorted(folder.glob("*.png"))[:4]

    for col, image_path in enumerate(image_paths):

        image = Image.open(image_path)

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(GRAPH_DIR / "dataset_samples.png", dpi=200)

plt.show()

print("\nDataset download and inspection completed!")
