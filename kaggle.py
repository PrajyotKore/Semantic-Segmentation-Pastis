import os

# Path to your training script within the cloned repo
train_script_path = "/kaggle/working/CAT-Seg/train_net.py" # Or plain_train_net.py

# Path to your config file within the cloned repo (or if you copy/edit it)
config_file_path = "/kaggle/working/CAT-Seg/configs/pastis-config.yaml" 

# Ensure the config_file_path has the correct DATASETS.ROOT for Kaggle
# If you haven't edited the YAML, you can override it via command line opts:
kaggle_dataset_root = "/kaggle/input/your-pastis-dataset-name/" # Replace with actual name

command = f"""
python {train_script_path} \
    --config-file {config_file_path} \
    --num-gpus 1 \
    --dist-url auto \
    DATASETS.ROOT "{kaggle_dataset_root}" \
    OUTPUT_DIR "/kaggle/working/output_pastis_experiment"
"""

print(f"Executing command: {command}")
os.system(command)
