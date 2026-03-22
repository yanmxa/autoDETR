"""
Dataset Splitting Script
Splits data/process images and labels into train/test sets with 80:20 ratio
Maintains balanced distribution across categories (one, two, three)
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Define paths
PROCESS_DIR = Path("data/process")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")

IMAGES_DIR = "images"
LABELS_DIR = "labels"

# Train/test split ratio
TRAIN_RATIO = 0.85  # Optimized for small datasets (202 samples)

def create_directories():
    """Create necessary directory structure"""
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        (split_dir / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        (split_dir / LABELS_DIR).mkdir(parents=True, exist_ok=True)
    print("Directory structure created")

def get_files_by_category():
    """Get files grouped by category with deduplication"""
    categories = defaultdict(set)  # Use set for automatic deduplication

    # Get all image files
    images_path = PROCESS_DIR / IMAGES_DIR
    for img_file in images_path.glob("*.jpg"):
        filename = img_file.name

        # Extract category from filename
        if "-one-" in filename:
            category = "one"
        elif "-two-" in filename:
            category = "two"
        elif "-three-" in filename:
            category = "three"
        else:
            print(f"Warning: Cannot identify category for file: {filename}")
            continue

        # Check if corresponding label file exists
        label_file = PROCESS_DIR / LABELS_DIR / filename.replace(".jpg", ".txt")
        if label_file.exists():
            # Add to set (automatically handles duplicates)
            base_name = filename.replace(".jpg", "")
            categories[category].add(base_name)
        else:
            print(f"Warning: Missing label file: {label_file}")

    # Convert sets to lists for shuffling
    categories = {cat: list(files) for cat, files in categories.items()}

    return categories

def split_dataset(categories):
    """Split dataset by category"""
    train_files = []
    test_files = []

    print("\nCategory distribution:")
    for category, files in categories.items():
        # Shuffle randomly
        random.shuffle(files)

        # Calculate training set size
        train_count = int(len(files) * TRAIN_RATIO)

        # Allocate to train and test sets
        category_train = files[:train_count]
        category_test = files[train_count:]

        train_files.extend(category_train)
        test_files.extend(category_test)

        print(f"  {category:6s}: total={len(files):3d}, train={len(category_train):3d}, test={len(category_test):3d}")

    print(f"\nTotal: train={len(train_files)}, test={len(test_files)}")
    return train_files, test_files

def copy_files(file_list, target_dir):
    """Copy files to target directory"""
    for filename_base in file_list:
        # Copy image
        src_img = PROCESS_DIR / IMAGES_DIR / f"{filename_base}.jpg"
        dst_img = target_dir / IMAGES_DIR / f"{filename_base}.jpg"
        shutil.copy2(src_img, dst_img)

        # Copy label
        src_label = PROCESS_DIR / LABELS_DIR / f"{filename_base}.txt"
        dst_label = target_dir / LABELS_DIR / f"{filename_base}.txt"
        shutil.copy2(src_label, dst_label)

def verify_split():
    """Verify splitting results"""
    print("\nVerification results:")

    for split_name, split_dir in [("Train", TRAIN_DIR), ("Test", TEST_DIR)]:
        print(f"\n{split_name}:")

        # Count each category
        images_path = split_dir / IMAGES_DIR
        category_counts = {"one": 0, "two": 0, "three": 0}

        for img_file in images_path.glob("*.jpg"):
            filename = img_file.name
            if "-one-" in filename:
                category_counts["one"] += 1
            elif "-two-" in filename:
                category_counts["two"] += 1
            elif "-three-" in filename:
                category_counts["three"] += 1

        total = sum(category_counts.values())
        print(f"  Total: {total}")
        for category, count in category_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {category:6s}: {count:3d} ({percentage:5.2f}%)")

def main():
    print("Starting dataset split...")
    print("=" * 60)

    # 1. Create directories
    create_directories()

    # 2. Get files by category (with deduplication)
    print("\nScanning files...")
    categories = get_files_by_category()

    # 3. Split dataset
    train_files, test_files = split_dataset(categories)

    # 4. Copy files
    print("\nCopying files to train set...")
    copy_files(train_files, TRAIN_DIR)

    print("Copying files to test set...")
    copy_files(test_files, TEST_DIR)

    # 5. Verify results
    verify_split()

    print("\n" + "=" * 60)
    print("Dataset split completed!")

if __name__ == "__main__":
    main()
