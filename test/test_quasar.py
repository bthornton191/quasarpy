import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quasarpy.quasar import DatasetConfig, KrigingConfig, Quasar


def test_initialization(mock_quasar_exe, tmp_path):
    """
    Test the initialization of the Quasar class.

    Verifies that the working directory is created and that the necessary
    scripts (Simulation.qsr) are copied to it.

    Parameters
    ----------
    mock_quasar_exe : Path
        Fixture providing a path to a mock Quasar executable.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    # Test with bundled scripts (default behavior)
    # This verifies that the package structure is correct and scripts are found
    q = Quasar(mock_quasar_exe, work_dir=tmp_path / "work")
    assert q.work_dir.exists()
    assert (q.work_dir / "Simulation.qsr").exists()


def test_write_x_file(mock_quasar_exe, sample_data, tmp_path):
    """
    Test the _write_x_file method.

    Verifies that the input file is written with the correct format:
    - Semicolon delimiter
    - Scientific notation for numbers
    - Correct header

    Parameters
    ----------
    mock_quasar_exe : Path
        Fixture providing a path to a mock Quasar executable.
    sample_data : tuple
        Fixture providing sample X and Y DataFrames.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    X, _ = sample_data
    q = Quasar(mock_quasar_exe, work_dir=tmp_path)

    q._write_x_file(X, "test_X.csv")

    file_path = tmp_path / "test_X.csv"
    assert file_path.exists()

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check header
    assert lines[0].strip() == "x1;x2"

    # Check data format (scientific notation)
    # 1.0 -> 1.00000000E+00
    assert "1.00000000E+00" in lines[1]
    assert ";" in lines[1]


def test_write_y_file(mock_quasar_exe, sample_data, tmp_path):
    """
    Test the _write_y_file method.

    Verifies that the output file is written with the correct format:
    - No header
    - Semicolon delimiter
    - Scientific notation for numbers

    Parameters
    ----------
    mock_quasar_exe : Path
        Fixture providing a path to a mock Quasar executable.
    sample_data : tuple
        Fixture providing sample X and Y DataFrames.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    _, Y = sample_data
    q = Quasar(mock_quasar_exe, work_dir=tmp_path)

    q._write_y_file(Y, tmp_path / "test_Y.csv")

    file_path = tmp_path / "test_Y.csv"
    assert file_path.exists()

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check data format (No header, just data)
    # 10.0 -> 1.00000000E+01
    assert "1.00000000E+01" in lines[0]


def test_command_generation(mock_quasar_exe, sample_data, tmp_path):
    """
    Test the generation of the Quasar command line arguments.

    Verifies that the command list contains the correct paths, dataset arguments,
    and solver configuration parameters.

    Parameters
    ----------
    mock_quasar_exe : Path
        Fixture providing a path to a mock Quasar executable.
    sample_data : tuple
        Fixture providing sample X and Y DataFrames.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    X, Y = sample_data
    ds = DatasetConfig(
        name='test_ds',
        data=Y,
        kriging_config=KrigingConfig(basis_function=3, stationarity=1)
    )

    q = Quasar(mock_quasar_exe, work_dir=tmp_path)
    q.train(X, [ds])

    cmd = q.command

    # Check basic args
    assert str(mock_quasar_exe) in cmd[0]
    assert 'Simulation.qsr' in cmd[1]
    assert str(tmp_path / 'X_FILE.csv').replace("\\", "/") in cmd[3].replace("\\", "/")  # Normalize paths for check

    # Check dataset args
    # We need to find where the dataset args start.
    # Arg 7 is number of datasets (1)
    assert cmd[7] == '1'

    # Dataset args start after that
    # Local arg count should be 5 + 4 (kriging args) = 9
    assert cmd[8] == '9'
    assert cmd[9] == 'test_ds'
    assert str(tmp_path / 'test_ds' / 'Y_FILE.csv').replace("\\", "/") in cmd[10].replace("\\", "/")

    # Check Kriging args
    # Solver ID = 2
    assert cmd[12] == '2'
    # Basis function = 3
    assert cmd[13] == '3'
    # Stationarity = 1
    assert cmd[14] == '1'


# Skip if Quasar is not installed
QUASAR_INSTALLED = 'ODYSSEE_CAE_INSTALLDIR' in os.environ or 'ODYSSEE_SOLVER_INSTALLDIR' in os.environ


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_quasar_integration():
    """
    Integration test running the actual Quasar executable.

    This test is skipped if the Quasar environment variables are not set.
    It performs a full training and prediction cycle on a simple linear problem
    and verifies that the predictions are reasonably close to the expected values.
    """
    # 1. Setup Data
    # Simple function y(t) = (2*x1 + x2) * (1 + 0.5*t)
    # This creates a time-varying response curve for each sample.
    x = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'x2': [1.0, 2.0, 1.0, 2.0, 1.0]
    })

    # Generate curves with more time steps to look like a "full curve"
    # t goes from 0 to 2 in 21 steps
    t_steps = np.linspace(0, 2, 21)

    y_data = {}
    for label, row in x.iterrows():
        base = 2 * row['x1'] + row['x2']
        # Curve: base * (1 + 0.5*t)
        curve = base * (1 + 0.5 * t_steps)
        y_data[label] = curve

    y = pd.DataFrame(y_data).T

    # 2. Configure Datasets
    ds1 = DatasetConfig(
        name='test_dataset',
        data=y,
        kriging_config=KrigingConfig(basis_function=2)  # Linear basis function
    )

    # 3. Initialize Quasar (auto-detect exe and scripts)
    q = Quasar(keep_work_dir=True)  # Keep work dir for debugging if needed

    # 4. Train
    q.train(x, [ds1])

    # 5. Predict
    X_new = pd.DataFrame({
        'x1': [1.5, 3.5],
        'x2': [1.5, 1.5]
    })

    results = q.predict(X_new)

    # Check results
    y_pred = results['test_dataset']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot training data
    # X-axis: Time (t_steps)
    # Y-axis: Input parameter x1
    # Z-axis: Response value
    for i in range(len(y)):
        x1_val = x.iloc[i]['x1']
        # Create arrays for 3D plotting
        xs = t_steps
        ys = np.full(xs.shape, x1_val)
        zs = y.iloc[i].values
        ax.plot(xs, ys, zs, color='gray', alpha=0.3, linestyle='-', label='Train' if i == 0 else "")

    for i in range(len(X_new)):
        x1_val = X_new.iloc[i]['x1']

        # Plot prediction
        xs = t_steps
        ys = np.full(xs.shape, x1_val)
        zs_pred = y_pred.iloc[i].values
        ax.plot(xs, ys, zs_pred, marker='o', markersize=4, label=f'Pred Sample {i}')

        # Plot expected
        x2 = X_new.iloc[i]['x2']
        base = 2 * x1_val + x2
        expected = base * (1 + 0.5 * t_steps)
        ax.plot(xs, ys, expected, linestyle='--', label=f'Exp Sample {i}')

    ax.set_title('Quasar Integration Test: Predictions vs Expected (3D)')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Input Parameter x1')
    ax.set_zlabel('Response Value')
    ax.legend()
    fig.savefig(Path(__file__).parent / 'quasar_integration_test.png')  # Save figure instead of showing

    for _, row in X_new.iterrows():
        expected = (2 * row['x1'] + row['x2']) * (1 + 0.5 * t_steps)
        actual = y_pred.loc[row.name]
        assert np.allclose(actual, expected, atol=0.01)


def test_quasar_init_no_exe():
    """
    Test initialization failure when Quasar executable is missing.

    Verifies that a FileNotFoundError is raised if the Quasar executable path
    is not provided and cannot be found in the environment variables.
    """
    # Test that it raises error if env vars are missing
    old_cae = os.environ.pop('ODYSSEE_CAE_INSTALLDIR', None)
    old_solver = os.environ.pop('ODYSSEE_SOLVER_INSTALLDIR', None)

    try:
        with pytest.raises(FileNotFoundError):
            Quasar()
    finally:
        if old_cae:
            os.environ['ODYSSEE_CAE_INSTALLDIR'] = old_cae
        if old_solver:
            os.environ['ODYSSEE_SOLVER_INSTALLDIR'] = old_solver


@patch("subprocess.Popen")
def test_predict_mocked(mock_popen, mock_quasar_exe, mock_scripts_dir, sample_data, tmp_path):
    """
    Test the predict method with a mocked subprocess.

    Verifies that the predict method correctly writes the input file,
    calls the subprocess, and reads the output file.

    Parameters
    ----------
    mock_popen : MagicMock
        Mock object for subprocess.Popen.
    mock_quasar_exe : Path
        Fixture providing a path to a mock Quasar executable.
    mock_scripts_dir : Path
        Fixture providing a path to a mock scripts directory.
    sample_data : tuple
        Fixture providing sample X and Y DataFrames.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    # Setup mock process
    process_mock = MagicMock()
    process_mock.communicate.return_value = ("Output", "Error")
    process_mock.returncode = 0
    mock_popen.return_value.__enter__.return_value = process_mock

    X, Y = sample_data
    ds = DatasetConfig(name="test_ds", data=Y)

    q = Quasar(mock_quasar_exe, qsr_scripts_dir=mock_scripts_dir, work_dir=tmp_path)
    q.train(X, [ds])

    # Create a dummy prediction result file
    pred_dir = tmp_path / "test_ds"
    pred_dir.mkdir(exist_ok=True)

    # Create YN.csv with some dummy data
    # Format: val1;val2
    # No header, no index in the output file based on _read_y_file implementation
    # Wait, _read_y_file expects semicolon separated, no header.
    # Let's write a file that matches what we expect to read.

    # If we predict for 1 sample, we expect 1 row.
    with open(pred_dir / "YN.csv", "w") as f:
        f.write("1.1;1.2\n")

    X_new = pd.DataFrame({'x1': [1.5], 'x2': [0.15]})

    results = q.predict(X_new)

    assert "test_ds" in results
    res_df = results["test_ds"]
    assert res_df.shape == (1, 2)
    assert res_df.iloc[0, 0] == 1.1
    assert res_df.iloc[0, 1] == 1.2

    # Verify subprocess was called
    assert mock_popen.called
