from pathlib import Path

# Root project directory
root = Path("pre_industrial_pipeline")

# Directories to create
directories = [
    "data/raw",
    "data/processed",
    "data/ground_truth",

    "notebooks",

    "src",
    "src/field_segmentation",
    "src/key_field_identification",

    "app",
    "app/pages",

    "tests",
]

# Files to create
files = [
    "notebooks/01_dataset_vectorization.ipynb",
    "notebooks/02_ngram_tokenization_theory.ipynb",
    "notebooks/03_token_frequency_analysis.ipynb",

    "src/__init__.py",
    "src/data_loader.py",

    "src/field_segmentation/__init__.py",
    "src/field_segmentation/tokenization.py",
    "src/field_segmentation/entropy.py",
    "src/field_segmentation/fvi.py",
    "src/field_segmentation/graph_scoring.py",

    "app/main.py",
    "app/pages/1_Dataset_Viewer.py",
    "app/pages/2_Tokenization_Engine.py",

    "requirements.txt",
    "README.md",
]

# Create directories
for directory in directories:
    (root / directory).mkdir(parents=True, exist_ok=True)

# Create files
for file in files:
    file_path = root / file

    # Ensure parent directories exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create empty file if it doesn't exist
    file_path.touch(exist_ok=True)

print(f"Project structure created at: {root.resolve()}")