import csv

import mrcfile
import numpy as np
from typer.testing import CliRunner

from torch_segment_tomogram_boundaries.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("prepare", "train", "predict", "fit-planes", "fetch"):
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "tomo-slab" in result.output


def test_predict_rejects_even_slab_size(tmp_path):
    tomo = tmp_path / "t.mrc"
    mrcfile.write(tomo, np.zeros((4, 4, 4), np.float32))
    result = runner.invoke(app, ["predict", str(tomo), "--slab-size", "4"])
    assert result.exit_code != 0


def test_fit_planes_roundtrip(tmp_path):
    mask = np.zeros((64, 128, 128), np.float32)
    mask[20:44] = 1
    src, dst = tmp_path / "m.mrc", tmp_path / "out" / "fit.mrc"
    mrcfile.write(src, mask)
    result = runner.invoke(app, ["fit-planes", str(src), str(dst)])
    assert result.exit_code == 0, result.output
    with mrcfile.open(dst) as mrc:
        assert mrc.data.shape == mask.shape


def _tilted_slab(shape=(96, 128, 128), thickness=20.0, slope=0.5):
    """Slab whose surfaces tilt along X; true perpendicular thickness is given."""
    zz, yy, xx = np.mgrid[0: shape[0], 0: shape[1], 0: shape[2]]
    z0 = 20 + slope * xx
    dz = thickness * np.sqrt(1 + slope**2)  # vertical extent for perpendicular thickness
    return ((zz >= z0) & (zz <= z0 + dz)).astype(np.float32)


def test_thickness_is_perpendicular_not_along_z(tmp_path):
    src, out = tmp_path / "m.mrc", tmp_path / "t.csv"
    with mrcfile.new(src) as mrc:
        mrc.set_data(_tilted_slab())
        mrc.voxel_size = 10.0
    result = runner.invoke(app, ["thickness", str(src), "-o", str(out)])
    assert result.exit_code == 0, result.output
    with open(out) as f:
        row = next(csv.DictReader(f))
    # Along-Z would give ~22.4 vox; perpendicular should be ~20 vox = ~20 nm.
    assert abs(float(row["median_vox"]) - 20) < 1.5, row
    assert abs(float(row["median_nm"]) - 20) < 1.5, row


def test_predict_fits_planes_once_and_reuses_them(tmp_path, monkeypatch):
    from torch_segment_tomogram_boundaries import measure, postprocess, predict as predict_mod

    slab = _tilted_slab(shape=(96, 128, 128), thickness=20.0, slope=0.3)

    class FakePredictor:
        def __init__(self, *args, **kwargs):
            pass

        def predict_probabilities(self, *args, **kwargs):
            return slab

    calls = []
    real_fit = postprocess.fit_slab_planes

    def counting_fit(*args, **kwargs):
        calls.append(1)
        return real_fit(*args, **kwargs)

    def forbidden_refit(*args, **kwargs):
        raise AssertionError("measure_thickness must reuse the fitted planes")

    monkeypatch.setattr(predict_mod, "TomoSlabPredictor", FakePredictor)
    monkeypatch.setattr(postprocess, "fit_slab_planes", counting_fit)
    monkeypatch.setattr(measure, "fit_slab_planes", forbidden_refit)

    tomo, ckpt = tmp_path / "t.mrc", tmp_path / "m.ckpt"
    with mrcfile.new(tomo) as mrc:
        mrc.set_data(np.zeros(slab.shape, np.float32))
        mrc.voxel_size = 10.0
    ckpt.write_bytes(b"")
    out, csv_path = tmp_path / "out", tmp_path / "t.csv"

    result = runner.invoke(
        app,
        ["predict", str(tomo), "-c", str(ckpt), "-o", str(out), "--fit-planes",
         "--thickness-file", str(csv_path)],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert (out / "t_fitted_mask.mrc").exists()
    with open(csv_path) as f:
        row = next(csv.DictReader(f))
    assert abs(float(row["median_nm"]) - 20) < 2, row
