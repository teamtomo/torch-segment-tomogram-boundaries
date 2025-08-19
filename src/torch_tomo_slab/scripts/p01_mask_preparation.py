================
File: src/torch_tomo_slab/scripts/p01_mask_preparation.py
================
import imodmodel
import numpy as np
import mrcfile
from sklearn.linear_model import RANSACRegressor, LinearRegression
import pandas as pd
from pathlib import Path
import sys
import argparse



sys.path.append(str(Path(__file__).resolve().parents[2]))
from torch_tomo_slab import constants


def ransac_plane_detection(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("DataFrame must contain 'x', 'y', and 'z' columns.")

    if len(df) < 3:
        raise ValueError(f"Not enough points ({len(df)}) in the model to fit any plane.")

    z_median = df['z'].median()

    bottom_region_candidates = df[df['z'] <= z_median]
    top_region_candidates = df[df['z'] >= z_median]

    planes = {}

    def fit_single_plane(region_df, region_name):
        if len(region_df) >= 3:
            X = region_df[['x', 'y']].values
            y = region_df['z'].values

            try:
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(X, y)
                if not hasattr(ransac, 'estimator_') or ransac.estimator_ is None:
                    raise ValueError("RANSAC could not find a valid model (e.g., no inliers).")


                coeffs = [float(c) for c in ransac.estimator_.coef_]
                intercept = float(ransac.estimator_.intercept_)

                return {
                    'coefficients': [*coeffs, intercept],
                    'inliers_df': region_df[ransac.inlier_mask_].copy(),
                    'outliers_df': region_df[~ransac.inlier_mask_].copy()
                }
            except ValueError as e:
                print(f"RANSAC failed for {region_name} region: {e}")
                print(f"Attempting a simple least squares fit for {region_name} region.")
                lr = LinearRegression()
                lr.fit(X, y)

                coeffs = [float(c) for c in lr.coef_]
                intercept = float(lr.intercept_)

                return {
                    'coefficients': [*coeffs, intercept],
                    'inliers_df': region_df.copy(),
                    'outliers_df': pd.DataFrame(columns=region_df.columns)
                }
        else:
            print(f"Warning: fit_single_plane called with insufficient points ({len(region_df)}) for {region_name}.")
            return None


    if len(bottom_region_candidates) >= 3:
        planes['bottom'] = fit_single_plane(bottom_region_candidates, 'bottom')
    if planes.get('bottom') is None and len(df) >= 3:
        print("Warning: Fitting 'bottom' plane using all points due to issues with bottom region.")
        planes['bottom'] = fit_single_plane(df, 'bottom (all points)')


    if len(top_region_candidates) >= 3:
        planes['top'] = fit_single_plane(top_region_candidates, 'top')
    if planes.get('top') is None and len(df) >= 3:
        print("Warning: Fitting 'top' plane using all points due to issues with top region.")
        planes['top'] = fit_single_plane(df, 'top (all points)')

    if planes.get('bottom') is None:
        raise RuntimeError("Failed to fit the bottom plane.")
    if planes.get('top') is None:
        raise RuntimeError("Failed to fit the top plane.")

    return planes


def generate_volume_mask(planes, reference_mrc_path, output_mrc_path,
                         diagnostic_mrc_path=None):

    with mrcfile.open(reference_mrc_path, permissive=True) as mrc:
        if diagnostic_mrc_path is not None:
            original_data = mrc.data.copy()
        volume_shape = mrc.data.shape
        angpix = float(mrc.voxel_size.x)

    Nz, Ny, Nx = volume_shape
    print(f"Reference volume dimensions: Nz={Nz}, Ny={Ny}, Nx={Nx}")

    if not ('bottom' in planes and 'top' in planes and \
            planes['bottom'] is not None and planes['top'] is not None):
        raise ValueError("Planes dictionary must contain valid 'bottom' and 'top' plane definitions.")

    coef_b = planes['bottom']['coefficients']
    coef_t = planes['top']['coefficients']
    print(f"Bottom plane coefficients (z = ax + by + d): {coef_b}")
    print(f"Top plane coefficients (z = ax + by + d): {coef_t}")

    yy_grid, xx_grid = np.mgrid[0:Ny, 0:Nx]

    z_values_bottom_plane = coef_b[0] * xx_grid + coef_b[1] * yy_grid + coef_b[2]
    z_values_top_plane = coef_t[0] * xx_grid + coef_t[1] * yy_grid + coef_t[2]

    min_plane_z = np.minimum(z_values_bottom_plane, z_values_top_plane)
    max_plane_z = np.maximum(z_values_bottom_plane, z_values_top_plane)

    zz_indices = np.arange(Nz, dtype=np.float32)

    condition_lower = zz_indices[:, np.newaxis, np.newaxis] >= min_plane_z
    condition_upper = zz_indices[:, np.newaxis, np.newaxis] <= max_plane_z

    mask_volume = (condition_lower & condition_upper).astype(np.int8)


    output_mrc_path.parent.mkdir(parents=True, exist_ok=True)
    mrcfile.write(output_mrc_path, mask_volume, overwrite=True, voxel_size=angpix)


    if diagnostic_mrc_path is not None:
        diagnostic_mrc_path.parent.mkdir(parents=True, exist_ok=True)
        masked_volume = original_data * mask_volume
        mrcfile.write(diagnostic_mrc_path, masked_volume.astype(np.float32),
                     overwrite=True, voxel_size=angpix)
        print(f"Diagnostic masked volume saved to {diagnostic_mrc_path}")

    print(f"Mask volume saved to {output_mrc_path}")


def main(args):

    imod_model_dir = Path(args.imod_dir)
    ref_tomo_dir = Path(args.ref_dir)
    output_mask_dir = Path(args.out_dir)
    print(f"Searching for IMOD models in: {imod_model_dir}")
    imod_files = sorted(list(imod_model_dir.glob("*.mod")))
    if not imod_files:
        print(f"Error: No '.mod' files found in '{imod_model_dir}'. Please check the path in constants.py.")
        return

    if constants.MULTIPLY_TOMO_MASK:
        diagnostics_dir = output_mask_dir / "diagnostic_vols"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for imod_path in imod_files:
        print(f"\n--- Processing model: {imod_path.name} ---")

        file_stem = imod_path.stem
        ref_tomo_path = ref_tomo_dir / f"{file_stem}.mrc"
        output_mask_path = output_mask_dir / f"{file_stem}.mrc"


        diagnostic_path = None
        if constants.MULTIPLY_TOMO_MASK:
            diagnostic_path = diagnostics_dir / f"{file_stem}_masked.mrc"

        if not ref_tomo_path.exists():
            print(
                f"Warning: Skipping model '{imod_path.name}'. Corresponding reference tomogram not found at '{ref_tomo_path}'.")
            continue
        try:
            print(f"Reading IMOD model from: {imod_path}")
            df = imodmodel.read(imod_path)
            if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['x', 'y', 'z']):
                print(f"Error: Invalid data format from '{imod_path}'. Expected DataFrame with 'x, y, z' columns.")
                continue
            if len(df) < 3:
                print(f"Error: Not enough points ({len(df)}) in '{imod_path.name}' to proceed.")
                continue
            print(f"Loaded {len(df)} points.")
            print("Detecting planes using RANSAC...")
            planes = ransac_plane_detection(df)
            print("Generating volume mask...")
            generate_volume_mask(planes, ref_tomo_path, output_mask_path, diagnostic_path)

        except Exception as e:
            print(f"An error occurred while processing {imod_path.name}: {e}")
            print("Skipping to the next file.")
            continue
    print("\n--- Script finished successfully. ---")

def main_cli():

    parser = argparse.ArgumentParser(...)
    parser.add_argument("--imod_dir", type=str, default=constants.IMOD_MODEL_DIR,
                        help="Directory containing IMOD (.mod) files.")
    parser.add_argument("--ref_dir", type=str, default=constants.REFERENCE_TOMOGRAM_DIR,
                        help="Directory containing reference tomogram (.mrc) files.")
    parser.add_argument("--out_dir", type=str, default=constants.MASK_OUTPUT_DIR,
                        help="Directory to save the generated mask (.mrc) files.")
    parser.add_argument("--diagnostics", type=bool, default=constants.MULTIPLY_TOMO_MASK,
                       help="Save mask multipled tomos for diagnostic reasons.")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    main_cli()
