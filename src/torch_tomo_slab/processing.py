# src/torch_tomo_slab/processing.py
import logging
import gc
from pathlib import Path
from typing import List, Tuple
import mrcfile
import numpy as np
import torch
from tqdm import tqdm

from . import config, constants
from .utils.twoD import robust_normalization, local_variance_2d
from .utils.threeD import resize_and_pad_3d

log = logging.getLogger(__name__)

class TrainingDataGenerator:
    """
    Generates standardized 2D training samples from 3D tomograms and their corresponding masks.
    This class encapsulates the logic from the original `p02_data_preparation.py` script,
    providing a reusable API for data preparation.
    """

    def __init__(self,
                 volume_dir: Path = constants.REFERENCE_TOMOGRAM_DIR,
                 mask_dir: Path = constants.MASK_OUTPUT_DIR,
                 output_train_dir: Path = constants.TRAIN_DATA_DIR,
                 output_val_dir: Path = constants.VAL_DATA_DIR,
                 validation_fraction: float = config.VALIDATION_FRACTION,
                 target_volume_shape: Tuple[int, int, int] = constants.TARGET_VOLUME_SHAPE):
        """
        Initializes the data generator with configuration parameters.
        Args:
            volume_dir: Directory containing the input tomogram volumes (.mrc files).
            mask_dir: Directory containing the input mask volumes (.mrc files).
            output_train_dir: Directory where training samples (.pt files) will be saved.
            output_val_dir: Directory where validation samples (.pt files) will be saved.
            validation_fraction: The fraction of data to be set aside for validation.
            target_volume_shape: The universal 3D shape to which all volumes will be resized.
        """
        self.volume_dir = volume_dir
        self.mask_dir = mask_dir
        self.output_train_dir = output_train_dir
        self.output_val_dir = output_val_dir
        self.validation_fraction = validation_fraction
        self.target_shape = target_volume_shape
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            log.info("CUDA is available. Using GPU.")
            return torch.device("cuda")
        log.info("CUDA not available. Using CPU.")
        return torch.device("cpu")

    def _find_data_pairs(self) -> List[Tuple[Path, Path]]:
        pairs = []
        label_files_map = {f.stem: f for f in self.mask_dir.glob("*.mrc")}
        for vol_path in sorted(list(self.volume_dir.glob("*.mrc"))):
            if vol_path.stem in label_files_map:
                pairs.append((vol_path, label_files_map[vol_path.stem]))
            else:
                log.warning(f"No matching label for {vol_path.name}")
        return pairs

    def run(self):
        """
        Executes the data preparation pipeline: finds data pairs, splits them, processes each volume,
        and saves the resulting 2D slices to disk.
        """
        self.output_train_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

        all_data_pairs = self._find_data_pairs()
        if not all_data_pairs:
            raise FileNotFoundError(f"No tomogram/mask pairs found in {self.volume_dir} and {self.mask_dir}.")

        log.info(f"All volumes will be resized and padded to the universal 3D shape: {self.target_shape}")

        np.random.seed(42)
        np.random.shuffle(all_data_pairs)
        val_split_idx = int(len(all_data_pairs) * self.validation_fraction)
        train_pairs, val_pairs = all_data_pairs[val_split_idx:], all_data_pairs[:val_split_idx]
        log.info(f"Splitting into {len(train_pairs)} train and {len(val_pairs)} val volumes.")

        for split_name, data_pairs, output_dir in [("TRAIN", train_pairs, self.output_train_dir),
                                                   ("VAL", val_pairs, self.output_val_dir)]:
            log.info(f"--- Processing {split_name} set ---")
            for vol_file, label_file in tqdm(data_pairs, desc=f"Processing {split_name} Tomograms"):
                volume_std, label_std = None, None
                try:
                    volume = torch.from_numpy(mrcfile.open(vol_file, permissive=True).data.astype(np.float32))
                    label = torch.from_numpy(mrcfile.open(label_file, permissive=True).data.astype(np.int8))

                    volume_std = resize_and_pad_3d(volume, self.target_shape, mode='image').to(self.device)
                    label_std = resize_and_pad_3d(label, self.target_shape, mode='label').to(self.device)

                    for i in range(constants.NUM_SECTIONS_PER_VOLUME):
                        margin = 7
                        axis_to_slice = torch.randint(1, 3, (1,)).item()
                        D, H, W = volume_std.shape
                        if axis_to_slice == 1:  # Slice along Y-axis
                            slice_idx = torch.randint(margin, H - margin, (1,)).item()
                            vol_slab = volume_std[:, slice_idx - margin: slice_idx + margin + 1, :]
                            ortho_img = torch.mean(vol_slab, dim=1)
                            ortho_label = label_std[:, slice_idx, :]
                        else:  # Slice along X-axis
                            slice_idx = torch.randint(margin, W - margin, (1,)).item()
                            vol_slab = volume_std[:, :, slice_idx - margin: slice_idx + margin + 1]
                            ortho_img = torch.mean(vol_slab, dim=2)
                            ortho_label = label_std[:, :, slice_idx]

                        ortho_img_norm = robust_normalization(ortho_img)
                        ortho_img_var = local_variance_2d(ortho_img_norm)
                        two_channel_input = torch.stack([ortho_img_norm, ortho_img_var], dim=0)

                        save_path = output_dir / f"{output_dir.name}_{vol_file.stem}_view_{i}.pt"
                        torch.save({'image': two_channel_input.cpu(), 'label': ortho_label.cpu().long()}, save_path)

                except Exception as e:
                    log.error(f"Failed to process {vol_file.name}. Error: {e}", exc_info=True)
                finally:
                    del volume_std, label_std, volume, label
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()