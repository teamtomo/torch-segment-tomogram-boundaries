import torch_tomo_slab


def test_imports_with_version():
    assert isinstance(torch_tomo_slab.__version__, str)
