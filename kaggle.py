import os

# --- Configuration for Training ---

# 1. Path to your training script within the cloned CAT-Seg repo
train_script_path = "/kaggle/working/Semantic-Segmentation-Pastis/train_net.py"

# 2. Path to your config file within the cloned CAT-Seg repo
config_file_path = "/kaggle/working/Semantic-Segmentation-Pastis/configs/pastis-config.yaml"

# 3. Path to the prepared data root directory (should match `prepare_kaggle_dataset.py`)
# This path will be passed as DATASETS.ROOT to the training script
prepared_data_root = "/kaggle/working/pastis_for_catseg"

# 4. Output directory for training results
output_dir = "/kaggle/working/output_pastis_experiment"

# --- Main script logic for running training ---
def run_training():
    print("--- Starting Training ---")

    # Check if the prepared data root exists (basic check)
    if not os.path.exists(prepared_data_root) or \
       not os.path.exists(os.path.join(prepared_data_root, "PASTIS")) or \
       not os.path.exists(os.path.join(prepared_data_root, "FOLDER_train.txt")):
        print(f"ERROR: Prepared dataset root not found or incomplete at {prepared_data_root}")
        print("Please run the `prepare_kaggle_dataset.py` script first.")
        return

    # Construct the training command
    command = f"""
    python {train_script_path} \\
        --config-file {config_file_path} \\
        --num-gpus 1 \\
        --dist-url auto \\
        DATASETS.ROOT "{prepared_data_root}" \\
        OUTPUT_DIR "{output_dir}"
    """
    # Example: Add more overrides if needed
    # command += " SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHTS /path/to/pretrained.pth"

    print(f"\nExecuting command: {command}")

    # Run the command
    exit_code = os.system(command)

    if exit_code == 0:
        print("\nTraining command executed successfully.")
    else:
        print(f"\nTraining command failed with exit code: {exit_code}")
    
    print("--- Training Finished ---")

if __name__ == "__main__":
    run_training()
