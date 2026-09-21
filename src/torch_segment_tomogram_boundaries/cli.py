"""Command-line interface for torch-tomo-slab.

Heavy dependencies (torch, lightning, ...) are imported lazily inside each command
so that ``--help`` and ``--version`` stay fast.
"""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from torch_segment_tomogram_boundaries import __version__

app = typer.Typer(
    name="tomo-slab",
    help="Segment slab boundaries in tomographic volumes with a 2D U-Net.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)


class Accelerator(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"
    mps = "mps"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tomo-slab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Segment slab boundaries in tomographic volumes."""
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)


def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise typer.BadParameter(f"{what} not found: {path}")


def _report_thickness(rows: list[dict], output: Optional[Path]) -> None:
    """Print thickness rows to screen and optionally write them to a CSV file."""
    from torch_segment_tomogram_boundaries.measure import write_thickness_csv

    for r in rows:
        if r["mean_vox"] != r["mean_vox"]:  # NaN -> empty mask
            typer.secho(f"{r['name']}: could not measure thickness (mask empty or too small to fit planes)", fg=typer.colors.YELLOW)
            continue
        msg = f"{r['name']}: thickness {r['median_vox']:.1f} vox (mean {r['mean_vox']:.1f} +/- {r['std_vox']:.1f})"
        if r["voxel_size_A"] == r["voxel_size_A"]:
            msg += f" = {r['median_nm']:.1f} nm (mean {r['mean_nm']:.1f} +/- {r['std_nm']:.1f} nm)"
        else:
            msg += " [no voxel size in header; nm not reported]"
        typer.echo(msg)
    if output is not None and rows:
        write_thickness_csv(rows, output)
        typer.echo(f"Thickness table written to {output}")


@app.command()
def prepare(
    volume_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, metavar="VOLUME_DIR",
        help="Directory containing the input tomograms (*.mrc).",
    ),
    mask_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, metavar="MASK_DIR",
        help="Directory containing the ground-truth boundary masks (*.mrc). "
        "File names must match those in VOLUME_DIR.",
    ),
    output_dir: Path = typer.Option(
        Path("prepared_data"), "--output-dir", "-o",
        help="Output root; slices go to <output-dir>/train and <output-dir>/val.",
    ),
    validation_fraction: float = typer.Option(
        0.2, "--val-fraction", min=0.0, max=1.0,
        help="Fraction of volumes reserved for validation.",
    ),
) -> None:
    """Convert 3D tomograms + masks into 2D training slices."""
    from torch_segment_tomogram_boundaries.processing import TrainingDataGenerator

    generator = TrainingDataGenerator(
        volume_dir=volume_dir,
        mask_dir=mask_dir,
        output_train_dir=output_dir / "train",
        output_val_dir=output_dir / "val",
        validation_fraction=validation_fraction,
    )
    generator.run()
    typer.secho(f"Prepared data written to {output_dir}", fg=typer.colors.GREEN)


@app.command()
def train(
    data_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, metavar="DATA_DIR",
        help="Prepared data root containing train/ and val/ subdirectories "
        "(the --output-dir of `tomo-slab prepare`).",
    ),
    ckpt_dir: Optional[Path] = typer.Option(
        None, "--ckpt-dir", "-o", help="Where to save checkpoints and logs."
    ),
    learning_rate: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    max_epochs: Optional[int] = typer.Option(None, "--epochs", "-e", min=1),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", min=1),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", min=0),
    accelerator: Accelerator = typer.Option(Accelerator.auto, "--accelerator"),
    devices: Optional[int] = typer.Option(None, "--devices", min=1, help="Number of devices."),
) -> None:
    """Train a segmentation model on prepared 2D slices."""
    from torch_segment_tomogram_boundaries import config
    from torch_segment_tomogram_boundaries.trainer import train as run_train

    train_dir, val_dir = data_dir / "train", data_dir / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise typer.BadParameter(f"Expected directory {d} (run `tomo-slab prepare` first).")

    # Settings not exposed by `train()` are read from `config` at setup time.
    if batch_size is not None:
        config.BATCH_SIZE = batch_size
    if num_workers is not None:
        config.NUM_WORKERS = num_workers

    kwargs: dict = {"accelerator": accelerator.value}
    if devices is not None:
        kwargs["devices"] = devices
    run_train(
        train_data_dir=train_dir,
        val_data_dir=val_dir,
        ckpt_save_dir=ckpt_dir or config.CKPT_SAVE_PATH,
        learning_rate=learning_rate if learning_rate is not None else config.LEARNING_RATE,
        max_epochs=max_epochs if max_epochs is not None else config.MAX_EPOCHS,
        **kwargs,
    )


@app.command()
def predict(
    tomograms: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, metavar="TOMOGRAMS...",
        help="One or more tomograms (.mrc) to segment.",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", "-c", exists=True, dir_okay=False,
        help="Model checkpoint. Defaults to the pretrained model (downloaded if needed).",
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", file_okay=False, help="Directory for output masks."
    ),
    threshold: float = typer.Option(0.5, "--threshold", "-t", min=0.0, max=1.0),
    slab_size: int = typer.Option(15, "--slab-size", min=1, help="Odd slab size for blending; 1 disables."),
    batch_size: int = typer.Option(16, "--batch-size", "-b", min=1),
    smoothing_sigma: Optional[float] = typer.Option(
        None, "--smoothing-sigma", min=0.0, help="3D Gaussian smoothing of probabilities."
    ),
    save_probabilities: bool = typer.Option(
        False, "--save-probabilities", help="Also write the probability map."
    ),
    compile_model: bool = typer.Option(
        False, "--compile/--no-compile", help="Use torch.compile (slower start, faster inference)."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs."),
    fit_planes_mask: bool = typer.Option(
        False, "--fit-planes", help="Also write the plane-fitted mask (<stem>_fitted_mask.mrc)."
    ),
    downsample_grid_size: int = typer.Option(
        8, "--downsample-grid-size", min=1, help="Surface-point downsampling grid for plane fitting."
    ),
    thickness_file: Optional[Path] = typer.Option(
        None, "--thickness-file", dir_okay=False,
        help="Measure slab thickness and write it for all tomograms to this CSV file (also printed).",
    ),
) -> None:
    """Predict slab masks for one or more tomograms, optionally fitting planes and measuring thickness."""
    import mrcfile
    import numpy as np

    from torch_segment_tomogram_boundaries.measure import measure_thickness
    from torch_segment_tomogram_boundaries.postprocess import fit_slab_planes, generate_mask_from_planes
    from torch_segment_tomogram_boundaries.predict import TomoSlabPredictor

    if slab_size % 2 == 0:
        raise typer.BadParameter("must be odd", param_hint="--slab-size")

    if checkpoint is None:
        from torch_segment_tomogram_boundaries.fetch import get_latest_checkpoint

        checkpoint = get_latest_checkpoint()

    do_thickness = thickness_file is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = TomoSlabPredictor(checkpoint, compile_model=compile_model)
    thickness_rows: list[dict] = []

    for tomo in tomograms:
        mask_path = output_dir / f"{tomo.stem}_mask.mrc"
        prob_path = output_dir / f"{tomo.stem}_probabilities.mrc"
        fitted_path = output_dir / f"{tomo.stem}_fitted_mask.mrc"
        targets = (
            [mask_path]
            + ([prob_path] if save_probabilities else [])
            + ([fitted_path] if fit_planes_mask else [])
        )
        existing = [p for p in targets if p.exists()]
        if existing and not overwrite:
            typer.secho(f"Skipping {tomo.name}: {existing[0]} exists (use --overwrite).", fg=typer.colors.YELLOW)
            continue

        probs = predictor.predict_probabilities(
            tomo, slab_size=slab_size, batch_size=batch_size, smoothing_sigma=smoothing_sigma
        )
        with mrcfile.open(tomo, permissive=True, header_only=True) as src:
            voxel_size = src.voxel_size.copy()
        binary = probs > threshold
        mrcfile.write(mask_path, binary.astype(np.float32), voxel_size=voxel_size, overwrite=True)

        # Fit the top/bottom planes once; reuse them for the fitted mask and thickness.
        planes = None
        if fit_planes_mask or do_thickness:
            try:
                planes = fit_slab_planes(binary.astype(np.uint8), downsample_grid_size)
            except ValueError as e:
                typer.secho(f"{tomo.name}: plane fitting failed ({e})", fg=typer.colors.YELLOW, err=True)
        if do_thickness:
            thickness_rows.append(
                {"name": tomo.name, **measure_thickness(binary, float(voxel_size.x), planes=planes)}
            )
        if fit_planes_mask and planes is not None:
            fitted = generate_mask_from_planes(planes, binary.shape)
            mrcfile.write(fitted_path, fitted.astype(np.float32), voxel_size=voxel_size, overwrite=True)
        if save_probabilities:
            mrcfile.write(prob_path, probs.astype(np.float32), voxel_size=voxel_size, overwrite=True)
        typer.secho(f"{tomo.name} -> {mask_path}", fg=typer.colors.GREEN)

    if do_thickness:
        _report_thickness(thickness_rows, thickness_file)


@app.command("fit-planes")
def fit_planes(
    input_mask: Path = typer.Argument(
        ..., exists=True, dir_okay=False, metavar="INPUT_MASK",
        help="Existing binary mask (.mrc) to refine.",
    ),
    output_mask: Path = typer.Argument(
        ..., dir_okay=False, metavar="OUTPUT_MASK",
        help="Path of the fitted mask (.mrc) to write.",
    ),
    downsample_grid_size: int = typer.Option(
        8, "--downsample-grid-size", "-g", min=1,
        help="Voxel grid size for downsampling surface points before fitting.",
    ),
) -> None:
    """Refine a binary mask by fitting planes to its top and bottom surfaces."""
    import mrcfile
    import numpy as np

    from torch_segment_tomogram_boundaries.postprocess import fit_and_generate_mask

    with mrcfile.open(input_mask, permissive=True) as mrc:
        mask = mrc.data.astype(np.uint8)
        voxel_size = mrc.voxel_size.copy()

    try:
        fitted = fit_and_generate_mask(mask, downsample_grid_size)
    except (ValueError, RuntimeError) as e:
        typer.secho(f"Plane fitting failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mrcfile.write(output_mask, fitted, voxel_size=voxel_size, overwrite=True)
    typer.secho(f"Fitted mask saved to {output_mask}", fg=typer.colors.GREEN)


@app.command()
def thickness(
    masks: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, metavar="MASKS...",
        help="One or more existing binary masks (.mrc) to measure.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", dir_okay=False, help="Also write results to this CSV file."
    ),
) -> None:
    """Measure slab thickness (perpendicular top-bottom distance) of binary masks."""
    import mrcfile

    from torch_segment_tomogram_boundaries.measure import measure_thickness

    rows = []
    for path in masks:
        with mrcfile.open(path, permissive=True) as mrc:
            rows.append({"name": path.name, **measure_thickness(mrc.data, float(mrc.voxel_size.x))})
    _report_thickness(rows, output)


@app.command()
def fetch(
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", file_okay=False, help="Directory to store the checkpoint."
    ),
    filename: Optional[str] = typer.Option(None, "--filename", help="Local filename for the checkpoint."),
) -> None:
    """Download the pretrained checkpoint and print its path."""
    from torch_segment_tomogram_boundaries.fetch import get_latest_checkpoint

    typer.echo(get_latest_checkpoint(cache_dir=cache_dir, filename=filename))


if __name__ == "__main__":
    app()
