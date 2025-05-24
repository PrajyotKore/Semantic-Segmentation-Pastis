# pastis_dataset_mapper.py
import copy
import logging
import os

import numpy as np
import torch
import rasterio # For reading .tif files

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import read_image
import json # For loading normalization stat

logger = logging.getLogger(__name__)

# Based on PASTIS benchmark (dataloader_pixelset.py and PASTIS_labels.json)
PASTIS_LABELS_DESC = {
    "Background": {"id": 0, "color": [0,0,0]}, # In PASTIS_labels.json, this is UNKNOWN
    "Urban area": {"id": 1, "color": [200,0,0]},
    "Cultivated land": {"id": 2, "color": [0,200,0]},
    "Vineyard": {"id": 3, "color": [200,200,0]},
    "Forest": {"id": 4, "color": [0,100,0]},
    "Grassland": {"id": 5, "color": [150,255,150]},
    "Shrubland": {"id": 6, "color": [100,50,0]},
    "Water": {"id": 7, "color": [0,0,200]},
    "Wetland": {"id": 8, "color": [0,200,200]},
    "Meadow": {"id": 9, "color": [150,100,50]},
    "Orchard": {"id": 10, "color": [200,150,150]},
    "Bare soil": {"id": 11, "color": [100,100,100]},
    "Natural vegetation": {"id": 12, "color": [50,150,50]}, # Not in target_names of dataloader_pixelset
    "Permanent snow": {"id": 13, "color": [255,255,255]}, # Not in target_names
    "Clouds": {"id": 14, "color": [200,200,200]}, # Not in target_names
    "Mixed forest": {"id": 15, "color": [0,50,0]}, # Not in target_names
    "Coniferous forest": {"id": 16, "color": [0,25,0]}, # Not in target_names
    "Broadleaf forest": {"id": 17, "color": [0,75,0]}, # Not in target_names
    "Moors and heathland": {"id": 18, "color": [100,50,50]}, # Not in target_names
    "Transitional woodland/shrub": {"id": 19, "color": [75,25,0]}, # Not in target_names
    "Unlabelled": {"id": 20, "color": [255,0,255]} # In PASTIS_labels.json, this is DISCARD (ID 255)
}

# Target classes for the semantic segmentation task, as in dataloader_pixelset.py
PASTIS_TARGET_NAMES = (
    'Urban area', 'Cultivated land', 'Vineyard', 'Forest', 'Grassland',
    'Shrubland', 'Water', 'Wetland', 'Meadow', 'Orchard'
)
# Sentinel-2 bands used in PASTIS .npy files (10 bands)
# B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
# For RGB, we typically use B4, B3, B2
S2_BANDS_RGB_IDX = [2, 1, 0] # Indices for B4 (Red), B3 (Green), B2 (Blue)

class PastisDatasetMapper:
    """
    A DatasetMapper for the PASTIS dataset for semantic segmentation.
    It loads a single time-slice from the multi-spectral, multi-temporal data,
    selects RGB bands, normalizes them, loads and remaps the semantic mask,
    and applies augmentations.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: list,
        image_format: str,
        ignore_label: int,
        data_root: str, # Added to make it explicit, though often part of dataset_dict
        # PASTIS specific
        s2_bands_indices_for_rgb = None,
        normalization_mean=None,
        normalization_std=None,
        s2_value_scale=10000.0,
        clip_value_max=1.0, # Clip normalized values to this max
        use_custom_s2_norm=False, # Flag to enable custom S2 normalization
        custom_s2_norm_fold_key=None, # e.g., "Fold_1" if training on fold 1
        metadata_geojson_path=None, # Path to metadata.geojson for fold stats
    ):

        """
        Args:
            is_train: Whether to apply augmentations.
            augmentations: A list of Detectron2 augmentations.
            image_format: Typically "RGB" or "BGR".
            ignore_label: The label value to ignore during training/evaluation.
            data_root: Root directory of the PASTIS dataset.
            s2_bands_indices_for_rgb: List of 3 indices for R, G, B bands from the 10 S2 bands.
            normalization_mean: Per-channel mean for normalization.
            normalization_std: Per-channel std for normalization.
            use_custom_s2_norm: Whether to use custom S2 band normalization from metadata.geojson.
            custom_s2_norm_fold_key: The key for the fold (e.g., "Fold_1") in metadata.geojson to use for norm stats.
            metadata_geojson_path: Path to the metadata.geojson file.
        """
        if s2_bands_indices_for_rgb is None:
            s2_bands_indices_for_rgb = S2_BANDS_RGB_IDX # Default B4, B3, B2

        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.ignore_label = ignore_label
        self.data_root = data_root # Store data_root if passed this way
        self.s2_bands_indices_for_rgb = s2_bands_indices_for_rgb
        self.s2_value_scale = s2_value_scale
        self.clip_value_max = clip_value_max

        
        self.use_custom_s2_norm = use_custom_s2_norm
        self.custom_s2_band_means = None
        self.custom_s2_band_stds = None

        if self.use_custom_s2_norm and custom_s2_norm_fold_key and metadata_geojson_path:
            if os.path.exists(metadata_geojson_path):
                try:
                    with open(metadata_geojson_path, 'r') as f:
                        metadata_gj = json.load(f)
                    fold_stats = metadata_gj.get(custom_s2_norm_fold_key)
                    if fold_stats and "mean" in fold_stats and "std" in fold_stats:
                        self.custom_s2_band_means = np.array(fold_stats["mean"], dtype=np.float32)
                        self.custom_s2_band_stds = np.array(fold_stats["std"], dtype=np.float32)
                        logger.info(f"Loaded custom S2 norm stats for {custom_s2_norm_fold_key} from {metadata_geojson_path}")
                    else:
                        logger.warning(f"Could not find stats for {custom_s2_norm_fold_key} in {metadata_geojson_path}")
                except Exception as e:
                    logger.error(f"Error loading metadata.geojson for S2 norm: {e}")
            else:
                logger.warning(f"metadata.geojson for S2 norm not found at {metadata_geojson_path}")


        # Pixel normalization, if provided (e.g. for ImageNet pre-trained models)
        self.normalize_image = normalization_mean is not None and normalization_std is not None
        if self.normalize_image:
            self.imagenet_pixel_mean = torch.tensor(normalization_mean).view(-1, 1, 1)
            self.imagenet_pixel_std = torch.tensor(normalization_std).view(-1, 1, 1)


        # --- Label remapping logic (inspired by dataloader_pixelset.py) ---
        # Use PASTIS_LABELS_DESC to find original IDs
        # Map target classes to 0...N-1
        # Map "Background" (original ID 0) and "Unlabelled" (original ID 20, or DISCARD 255 in some versions) to ignore_label
        
        # Find max original ID to size the remap vector
        # Note: PASTIS_labels.json has UNKNOWN:0 and DISCARD:255.
        # The provided PASTIS_LABELS_DESC uses Background:0 and Unlabelled:20.
        # We need to be consistent with the actual values in the .tif annotation files.
        # Assuming .tif files use IDs as in PASTIS_LABELS_DESC.
        max_id_in_labels = 0
        for _, desc in PASTIS_LABELS_DESC.items():
            if desc["id"] > max_id_in_labels:
                max_id_in_labels = desc["id"]

        self.remap_vector = np.full(max_id_in_labels + 1, self.ignore_label, dtype=np.int64)
        
        target_class_original_ids = []
        for i, target_name in enumerate(PASTIS_TARGET_NAMES):
            original_id = -1
            # Find the original ID from PASTIS_LABELS_DESC
            for k, v in PASTIS_LABELS_DESC.items():
                # A bit of a hack for matching names if they differ slightly (e.g. "Urban area" vs "URBAN")
                if target_name.lower() in k.lower() or k.lower() in target_name.lower():
                    original_id = v["id"]
                    break
            if original_id == -1:
                logger.warning(f"Target class '{target_name}' not found in PASTIS_LABELS_DESC. Check names.")
                continue
            
            self.remap_vector[original_id] = i
            target_class_original_ids.append(original_id)

        # Handle specific ignore cases based on PASTIS_LABELS_DESC
        # Background (ID 0) and Unlabelled (ID 20) should be ignored   #TODO: Recheck
        if "Background" in PASTIS_LABELS_DESC:
             self.remap_vector[PASTIS_LABELS_DESC["Background"]["id"]] = self.ignore_label
        if "Unlabelled" in PASTIS_LABELS_DESC:
            self.remap_vector[PASTIS_LABELS_DESC["Unlabelled"]["id"]] = self.ignore_label
        
        # If your annotation files use the UNKNOWN:0 and DISCARD:255 convention:
        # self.remap_vector[0] = self.ignore_label # UNKNOWN
        # if 255 < len(self.remap_vector): self.remap_vector[255] = self.ignore_label # DISCARD
        # else: logger.warning("Max ID is less than 255, cannot map DISCARD label if it exists.")

        logger.info(f"PastisDatasetMapper: Remapping {len(target_class_original_ids)} target classes. Ignore label: {self.ignore_label}")
        logger.info(f"Remap vector (first 25 elements): {self.remap_vector[:25]}")


    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        
        # ImageNet normalization for models pre-trained on it
        # Set these in your config if needed:
        # INPUT.PIXEL_MEAN = [123.675, 116.280, 103.530]
        # INPUT.PIXEL_STD = [58.395, 57.120, 57.375]
        # These are typically for images in 0-255 range.
        # For satellite imagery scaled to 0-1, different stats or no further normalization might be better.
        # For now, let's assume we scale to 0-1 and don't apply ImageNet stats unless configured.
        
        norm_mean = cfg.MODEL.PIXEL_MEAN if cfg.MODEL.PIXEL_MEAN else None # Expects list
        norm_std = cfg.MODEL.PIXEL_STD if cfg.MODEL.PIXEL_STD else None # Expects list


        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "data_root": cfg.DATASETS.ROOT, # Assuming you add DATASETS.ROOT to your config
            "s2_bands_indices_for_rgb": cfg.INPUT.get("S2_BANDS_INDICES_RGB", S2_BANDS_RGB_IDX),
            "s2_value_scale": cfg.INPUT.get("S2_VALUE_SCALE", 10000.0),
            "clip_value_max": cfg.INPUT.get("S2_CLIP_VALUE_MAX", 1.0),
            "use_custom_s2_norm": cfg.INPUT.get("USE_CUSTOM_S2_NORM", False),
            "custom_s2_norm_fold_key": cfg.INPUT.get("CUSTOM_S2_NORM_FOLD_KEY", None), # e.g. "Fold_1"
            "metadata_geojson_path": os.path.join(cfg.DATASETS.ROOT, "metadata.geojson") if cfg.INPUT.get("USE_CUSTOM_S2_NORM", False) else None,
            "normalization_mean": norm_mean,
            "normalization_std": norm_std,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, per Detectron2's convention.
                Must contain "patch_id", "time_index".
                "data_root" can be in dataset_dict or pre-configured in __init__.
        Returns:
            dict: a format that Detectron2 C5 models can consume.
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        patch_id = dataset_dict["patch_id"]
        time_index = dataset_dict["time_index"]
        # data_root can come from dataset_dict or be pre-configured
        current_data_root = dataset_dict.get("data_root", self.data_root)

        # 1. Load S2 data (.npy file)
        s2_file_path = os.path.join(current_data_root, "DATA_S2", f"{patch_id}_S2.npy")
        try:
            s2_full_sequence = np.load(s2_file_path) # Shape (Time, Bands, H, W)
        except FileNotFoundError:
            logger.error(f"S2 data file not found: {s2_file_path}")
            # Return None or raise error to skip this sample
            # For robustness in large datasets, you might want to return None
            # and have the dataloader handle it (e.g. with a collate_fn that filters Nones)
            # However, Detectron2's default loader might not like None.
            # Crashing might be better to identify issues early.
            raise
        
        # 2. Select time slice
        # Assuming raw S2 values are scaled by s2_value_scale (e.g. 10000 for reflectance)
        s2_data_at_t = s2_full_sequence[time_index].astype(np.float32)

        # 2a. Apply S2 specific normalization
        if self.use_custom_s2_norm and self.custom_s2_band_means is not None and self.custom_s2_band_stds is not None:
            # Assuming custom_s2_band_means/stds are for the raw 0-10000 range S2 values
            # And s2_data_at_t is (Bands, H, W)
            means = self.custom_s2_band_means[:, np.newaxis, np.newaxis]
            stds = self.custom_s2_band_stds[:, np.newaxis, np.newaxis]
            stds[stds == 0] = 1e-6 # Avoid division by zero
            s2_data_at_t = (s2_data_at_t - means) / stds
            # After this custom normalization, data might not be in [0,1]. Clipping might still be useful.
            # s2_data_at_t = np.clip(s2_data_at_t, -3.0, 3.0) # Example clip after Z-score like norm
        else:
            # Original simple scaling
            s2_data_at_t /= self.s2_value_scale
            # Clip values to a reasonable range, e.g. [0, self.clip_value_max]
            # This clipping is for the 0-1 scaled data
            s2_data_at_t = np.clip(s2_data_at_t, 0, self.clip_value_max)


        # 3. Select R, G, B bands
        # Image shape will be (3, H, W)
        image = s2_data_at_t[self.s2_bands_indices_for_rgb, :, :]

        # 4. Load semantic segmentation mask (.tif file)
        sem_seg_gt_file_path = os.path.join(current_data_root, "ANNOTATIONS", f"{patch_id}_ANO.tif")
        try:
            with rasterio.open(sem_seg_gt_file_path) as src:
                sem_seg_gt = src.read(1).astype(np.int64) # Shape (H, W)
        except FileNotFoundError:
            logger.error(f"Annotation file not found: {sem_seg_gt_file_path}")
            raise # Or handle more gracefully

        # 5. Apply label remapping
        # Ensure sem_seg_gt values are within the bounds of remap_vector
        sem_seg_gt[sem_seg_gt >= len(self.remap_vector)] = self.ignore_label # Handle out-of-bounds IDs
        sem_seg_gt = self.remap_vector[sem_seg_gt]

        # Augmentations expect HWC image
        image = image.transpose(1, 2, 0) # C,H,W to H,W,C

        # 6. Apply augmentations
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # 7. Convert to torch tensors
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))) # HWC to CHW
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # Apply pixel normalization if configured (e.g. for ImageNet pre-trained models)
        # This is usually done after augmentations that might change pixel values (like brightness/contrast)
        # but before conversion to tensor if it expects a tensor.
        # Here, we do it on the tensor.
        if self.normalize_image:
             # This assumes dataset_dict["image"] (selected RGB bands) is in a range ImageNet stats expect
             dataset_dict["image"] = (dataset_dict["image"] - self.imagenet_pixel_mean) / self.imagenet_pixel_std

        # Populate other standard fields
        if "height" not in dataset_dict:
            dataset_dict["height"] = sem_seg_gt.shape[0]
        if "width" not in dataset_dict:
            dataset_dict["width"] = sem_seg_gt.shape[1]
        
        # Ensure image_id is present
        if "image_id" not in dataset_dict:
            dataset_dict["image_id"] = f"{patch_id}_t{time_index}"
            
        return dataset_dict

