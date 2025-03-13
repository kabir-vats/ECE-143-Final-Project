import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATA_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")

# make output dirs
for sub_dir in ["Overall", "True", "False"]:
    os.makedirs(os.path.join(VISUALIZATION_DIR, sub_dir), exist_ok=True)