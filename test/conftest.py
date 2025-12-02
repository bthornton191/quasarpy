import pytest
import pandas as pd
from pathlib import Path
import shutil


@pytest.fixture
def mock_quasar_exe(tmp_path):
    exe = tmp_path / "QuasarEngine_2.exe"
    exe.touch()
    return str(exe)


@pytest.fixture
def mock_scripts_dir(tmp_path):
    scripts_dir = tmp_path / "qsr_scripts"
    scripts_dir.mkdir()
    (scripts_dir / "Simulation.qsr").touch()
    return scripts_dir


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0],
        'x2': [0.1, 0.2, 0.3]
    })
    Y = pd.DataFrame({
        '0': [10.0, 20.0, 30.0],
        '1': [11.0, 21.0, 31.0]
    })
    return X, Y
