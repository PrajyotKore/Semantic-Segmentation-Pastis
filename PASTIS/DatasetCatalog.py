# Example: my_project/datasets/prepare_pastis.py
import os
import numpy as np
import json # For parsing GeoJSON/JSON
from detectron2.data import DatasetCatalog, MetadataCatalog
from .PastisDatasetMapper import PASTIS_TARGET_NAMES # Import from sibling module
import logging # For logging



# These are the 10 classes the model will learn
PASTIS_SEM_SEG_CLASSES = [name for name in PASTIS_TARGET_NAMES] # From pastis_dataset_mapper.py
# The ignore_label should match len(PASTIS_SEM_SEG_CLASSES) if remapping maps to 0..N-1
# and unknown/discard to N.
IGNORE_LABEL = len(PASTIS_SEM_SEG_CLASSES) # This will be 10

logger = logging.getLogger(__name__)

def get_pastis_dicts(data_root, split="train"):
    """
    Args:
        data_root (str): Path to the PASTIS dataset root.
        split (str): "train", "val", or "test".
                     This will now depend on how splits are defined or inferred from metadata.geojson.
    Returns:
        list[dict]: A list of dicts for Detectron2.
    """
    dataset_dicts = []
    
    # --- Handling split definition ---
    # Option 1: Using FOLDER_{split}.txt files (current implementation)
    split_file = os.path.join(data_root, f"FOLDER_{split}.txt")
    if not os.path.exists(split_file):
        logger.error(f"Split definition file not found: {split_file}")
        logger.error(f"Please ensure FOLDER_{split}.txt exists in {data_root} or modify this script to use metadata.geojson for splits.")
        return []
    
    with open(split_file, 'r') as f:
        patch_ids_for_split = [line.strip() for line in f.readlines()]

    # --- Optional: Loading metadata.geojson (currently not used for splitting in this version) ---
    # If you intend to use metadata.geojson for splitting, the logic above for patch_ids_for_split
    # would need to be replaced or augmented.
    metadata_file = os.path.join(data_root, "metadata.geojson")
    if not os.path.exists(metadata_file):
        logger.warning(f"Metadata file not found: {metadata_file}. This might be okay if not used for splitting.")
        # metadata = None # Not strictly necessary if not used
    else:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f) # This metadata variable is loaded but not used for splitting in this version.
                                    # It could be used for other per-patch information if needed.
    # --- End of metadata loading ---

    if not patch_ids_for_split:
        logger.warning(f"No patch IDs found for split '{split}' using file {split_file}.")
        return []

    for patch_id_str in patch_ids_for_split:
        try:
            # Attempt to convert to int if it's purely numeric, otherwise keep as string
            patch_id = int(patch_id_str) if patch_id_str.isdigit() else patch_id_str
        except ValueError:
            logger.warning(f"Could not parse patch_id '{patch_id_str}' as integer, using as string.")
            patch_id = patch_id_str
            
        s2_file_path = os.path.join(data_root, "DATA_S2", f"{patch_id}_S2.npy")

        if not os.path.exists(s2_file_path):
            logger.warning(f"S2 data file not found for patch {patch_id} at {s2_file_path}, skipping.")
            continue

        try:
            s2_data = np.load(s2_file_path) # Load to get num_timesteps
        except Exception as e:
            logger.error(f"Error loading S2 data file {s2_file_path} for patch {patch_id}: {e}, skipping.")
            continue
            
        if s2_data.ndim < 4 or s2_data.shape[0] == 0: # Expecting (Time, Bands, H, W)
            logger.warning(f"S2 data for patch {patch_id} has unexpected shape {s2_data.shape} or no time steps, skipping.")
            continue
            
        num_timesteps = s2_data.shape[0]
        height, width = s2_data.shape[2], s2_data.shape[3] # from (T,B,H,W)

        for t_idx in range(num_timesteps):
            record = {}
            record["image_id"] = f"{patch_id}_t{t_idx}"
            record["patch_id"] = patch_id
            record["time_index"] = t_idx
            record["data_root"] = data_root # Pass data_root to mapper
            record["height"] = height
            record["width"] = width
            # sem_seg_file_name will be constructed by the mapper
            dataset_dicts.append(record)
            
    logger.info(f"Loaded {len(dataset_dicts)} samples for {split} split from {data_root} using {split_file}.")
    return dataset_dicts

def register_pastis_semantic(name, data_root, split):
    DatasetCatalog.register(name, lambda: get_pastis_dicts(data_root, split))
    MetadataCatalog.get(name).set(
        stuff_classes=PASTIS_SEM_SEG_CLASSES,
        evaluator_type="pastis_sem_seg", # Matches train_net.py modification
        ignore_label=IGNORE_LABEL, # Crucial for training and evaluation
        stuff_colors=None, # Optional: add colors for visualization if needed
        thing_classes=[], # No "thing" classes for semantic segmentation
        # Add any other metadata your evaluator or model might need
    )

# Example registration calls (these should be made from your main training script)
# if __name__ == '__main__':
#     # This is for testing the script directly, not for actual training runs.
#     # In actual training, register_pastis_semantic is called from train_net.py.
#     logging.basicConfig(level=logging.INFO)
#     logger.info("Registering PASTIS datasets for direct script testing...")
#     
#     # Ensure DATA_ROOT is correct for your local setup if running this directly
#     # local_data_root = "H:/CAT-Seg/PASTIS/pastis-benchmark/DATASET" # Or your Kaggle path
#     # if not os.path.exists(os.path.join(local_data_root, "FOLDER_train.txt")):
#     #     logger.error(f"FOLDER_train.txt not found in {local_data_root}. Cannot test.")
#     # else:
#     #     register_pastis_semantic("pastis_sem_seg_train_test", local_data_root, "train")
#     #     register_pastis_semantic("pastis_sem_seg_val_test", local_data_root, "val")
#     #
#     #     # Example: Accessing the registered dataset
#     #     # train_data = DatasetCatalog.get("pastis_sem_seg_train_test")
#     #     # if train_data:
#     #     #     logger.info(f"First item in train_data: {train_data[0]}")
#     #     # else:
#     #     #     logger.info("Train data is empty or not loaded.")
#     #     #
#     #     # train_metadata = MetadataCatalog.get("pastis_sem_seg_train_test")
#     #     # logger.info(f"Train metadata: {train_metadata}")
#     logger.info("Direct script testing registration complete.")
