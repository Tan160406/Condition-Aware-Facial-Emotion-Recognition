from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw

# 1. PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

IMAGES_DIR = DATASET_DIR / "images"

VAL_LABELS = DATASET_DIR / "val_labels.csv"
TEST_LABELS = DATASET_DIR / "test_labels.csv"

CONDITIONS_DIR = DATASET_DIR / "conditions"


# 2. SETTINGS
# Fixed seed = reproducible results
RANDOM_SEED = 42

# Number of images used from each split.
# None = use the complete split.
MAX_VAL_IMAGES = None
MAX_TEST_IMAGES = None

# Moderate condition parameters
LOW_LIGHT_FACTOR = 0.45

BLUR_RADIUS = 1.5

NOISE_STD = 20

OCCLUSION_RATIO = 0.25

# 3. RANDOM NUMBER GENERATOR
rng = np.random.default_rng(RANDOM_SEED)

# 4. CONDITION FOLDERS
CONDITION_NAMES = [
    "low_light",
    "blur",
    "noise",
    "occlusion",
    "low_light_blur",
    "low_light_noise",
]

# 5. CREATE OUTPUT DIRECTORIES
for split in ["validation", "test"]:

    for condition in CONDITION_NAMES:

        output_dir = CONDITIONS_DIR / split / condition

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

# 6. CONDITION FUNCTIONS
def apply_low_light(image):
    """
    Reduce image brightness while keeping the
    facial structure visible.
    """

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array * LOW_LIGHT_FACTOR

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        image_array,
        mode="L"
    )


def apply_blur(image):
    """
    Apply moderate Gaussian blur.
    """

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=BLUR_RADIUS
        )
    )


def apply_noise(image):
    """
    Add reproducible Gaussian noise.
    """

    image_array = np.array(
        image,
        dtype=np.float32
    )

    noise = rng.normal(
        0,
        NOISE_STD,
        image_array.shape
    )

    noisy = image_array + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        noisy,
        mode="L"
    )


def apply_occlusion(image):
    """
    Apply a partial rectangular occlusion.

    The region is randomly positioned but remains
    reproducible because a fixed random seed is used.
    """

    image = image.copy()

    width, height = image.size

    # Occlusion dimensions
    occ_width = int(
        width * OCCLUSION_RATIO
    )

    occ_height = int(
        height * OCCLUSION_RATIO
    )

    # Random but reproducible location
    x_max = width - occ_width
    y_max = height - occ_height

    x = rng.integers(
        0,
        x_max + 1
    )

    y = rng.integers(
        0,
        y_max + 1
    )

    draw = ImageDraw.Draw(image)

    # Use mean gray level as the occluding region
    gray_value = int(
        np.mean(np.array(image))
    )

    draw.rectangle(
        [
            x,
            y,
            x + occ_width,
            y + occ_height
        ],
        fill=gray_value
    )

    return image


def apply_low_light_blur(image):
    """
    Apply low illumination followed by blur.
    """

    image = apply_low_light(image)

    image = apply_blur(image)

    return image


def apply_low_light_noise(image):
    """
    Apply low illumination followed by noise.
    """

    image = apply_low_light(image)

    image = apply_noise(image)

    return image


# 7. CONDITION FUNCTION MAPPING
CONDITION_FUNCTIONS = {
    "low_light": apply_low_light,
    "blur": apply_blur,
    "noise": apply_noise,
    "occlusion": apply_occlusion,
    "low_light_blur": apply_low_light_blur,
    "low_light_noise": apply_low_light_noise,
}

# 8. GENERATE DATA FOR ONE SPLIT
def generate_split(
    labels_path,
    split_name,
    max_images=None
):

    labels_df = pd.read_csv(
        labels_path
    )

    if max_images is not None:

        labels_df = labels_df.head(
            max_images
        )

    print("\n" + "=" * 65)
    print(
        f"GENERATING CONDITIONS: {split_name.upper()}"
    )
    print("=" * 65)

    print(
        f"Images: {len(labels_df)}"
    )

    metadata = []

    total = len(labels_df)

    for index, row in labels_df.iterrows():

        image_name = row["Image name"]

        emotion = row["emotion"]

        source_path = IMAGES_DIR / image_name

        if not source_path.exists():

            print(
                f"[WARNING] Missing image: {image_name}"
            )

            continue

        try:

            original = Image.open(
                source_path
            ).convert("L")

        except Exception as error:

            print(
                f"[WARNING] Could not open "
                f"{image_name}: {error}"
            )

            continue

        # ----------------------------------------------------
        # Generate each condition
        # ----------------------------------------------------

        for condition_name, function in CONDITION_FUNCTIONS.items():

            output_image = function(
                original
            )

            output_path = (
                CONDITIONS_DIR
                / split_name
                / condition_name
                / image_name
            )

            output_image.save(
                output_path
            )

            metadata.append({
                "Image name": image_name,
                "emotion": emotion,
                "condition": condition_name,
                "severity": "moderate",
                "source_split": split_name,
                "source_path": str(source_path),
                "condition_path": str(output_path)
            })

        # Progress
        processed = index + 1

        if processed % 500 == 0:

            print(
                f"Processed {processed}/{total}"
            )

    return metadata

# 9. GENERATE VALIDATION CONDITIONS
validation_metadata = generate_split(
    VAL_LABELS,
    "validation",
    MAX_VAL_IMAGES
)

# 10. GENERATE TEST CONDITIONS
test_metadata = generate_split(
    TEST_LABELS,
    "test",
    MAX_TEST_IMAGES
)

# 11. COMBINE METADATA
all_metadata = (
    validation_metadata
    + test_metadata
)


metadata_df = pd.DataFrame(
    all_metadata
)

# 12. SAVE METADATA
metadata_path = (
    CONDITIONS_DIR
    / "condition_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# 13. FINAL SUMMARY
print("\n" + "=" * 65)
print("CONDITION GENERATION COMPLETE")
print("=" * 65)

print(
    f"Validation images processed: "
    f"{len(validation_metadata) // len(CONDITION_NAMES)}"
)

print(
    f"Test images processed: "
    f"{len(test_metadata) // len(CONDITION_NAMES)}"
)

print(
    f"Total condition images: "
    f"{len(all_metadata)}"
)

print(
    f"Conditions per image: "
    f"{len(CONDITION_NAMES)}"
)

print("\nConditions generated:")

for condition in CONDITION_NAMES:

    count = (
        metadata_df["condition"]
        == condition
    ).sum()

    print(
        f"  {condition:<20} {count}"
    )

print("\nMetadata saved to:")

print(metadata_path)

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)