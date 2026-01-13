import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quasarpy.quasar import DatasetConfig, KrigingConfig, Quasar

PLOT_DIR = Path(__file__).parent / 'plots'


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


def y1_func(t, x1, x2):
    return (2 * x1 + x2) * (1 + 0.5 * t)


def y2_func(t, x1, x2):
    return (3 * x1 - x2) * np.exp(-0.3 * t)


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

    datasets: List[DatasetConfig] = []
    for i, func in enumerate([y1_func, y2_func]):
        y_data = {}
        for label, row in x.iterrows():
            curve = func(t_steps, row['x1'], row['x2'])
            y_data[label] = curve

        # 2. Configure Dataset
        ds = DatasetConfig(
            name=f'test_dataset_{i}',
            data=pd.DataFrame(y_data).T,
            kriging_config=KrigingConfig(basis_function=2)  # Linear basis function
        )

        datasets.append(ds)

    # 3. Initialize Quasar (auto-detect exe and scripts)
    q = Quasar(keep_work_dir=True)  # Keep work dir for debugging if needed

    # 4. Train
    q.train(x, datasets)

    # 5. Predict
    X_new = pd.DataFrame({
        'x1': [1.5, 3.5],
        'x2': [1.5, 1.5]
    })

    results = q.predict(X_new)

    # Check results
    failures = []
    for ds, func in zip(datasets, [y1_func, y2_func]):
        y_pred = results[ds.name]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot training data
        # X-axis: Time (t_steps)
        # Y-axis: Input parameter x1
        # Z-axis: Response value
        for i in range(len(x)):
            x1_val = x.iloc[i]['x1']
            # Create arrays for 3D plotting
            ys = np.full(t_steps.shape, x1_val)
            zs = ds.data.iloc[i].values
            ax.plot(t_steps, ys, zs, color='gray', alpha=0.3, linestyle='-', label='Train' if i == 0 else "")

        for i in range(len(X_new)):
            x1_val = X_new.iloc[i]['x1']

            # Plot prediction
            ys = np.full(t_steps.shape, x1_val)
            zs_pred = y_pred.iloc[i].values
            ax.plot(t_steps, ys, zs_pred, marker='o', markersize=4, label=f'Pred Sample {i}')

            # Plot expected
            x2 = X_new.iloc[i]['x2']
            expected = func(t_steps, x1_val, x2)
            ax.plot(t_steps, ys, expected, linestyle='--', label=f'Exp Sample {i}')

        ax.set_title('Quasar Integration Test: Predictions vs Expected (3D)')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Input Parameter x1')
        ax.set_zlabel('Response Value')
        ax.legend()

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_DIR / f'integration_{ds.name}.png')

        for _, row in X_new.iterrows():
            expected = func(t_steps, row['x1'], row['x2'])
            actual = y_pred.loc[row.name].values
            if not np.allclose(actual, expected, atol=0.01):
                failures.append({'dataset': ds.name, 'row': row.name, 'actual': actual, 'expected': expected})

    if failures != []:
        pytest.fail(f'Failures in predictions: {failures}')


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_quasar_predict_with_relative_work_dir():
    """
    Integration test verifying that Quasar works with a relative path for work_dir.

    This test is skipped if the Quasar environment variables are not set.
    It performs a full training and prediction cycle using a relative path
    for the working directory to ensure path handling is correct.
    """
    import shutil

    # Use a relative path for work_dir
    relative_work_dir = Path('test_relative_work_dir')

    # Clean up if it exists from a previous run
    if relative_work_dir.exists():
        shutil.rmtree(relative_work_dir)

    try:
        # 1. Setup Data
        x = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x2': [1.0, 2.0, 1.0, 2.0, 1.0]
        })

        t_steps = np.linspace(0, 2, 21)

        y_data = {}
        for label, row in x.iterrows():
            curve = y1_func(t_steps, row['x1'], row['x2'])
            y_data[label] = curve

        ds = DatasetConfig(
            name='relative_path_test',
            data=pd.DataFrame(y_data).T,
            kriging_config=KrigingConfig(basis_function=2)
        )

        # 2. Initialize Quasar with relative path
        q = Quasar(work_dir=relative_work_dir, keep_work_dir=True)

        # Verify the work_dir was created
        assert relative_work_dir.exists(), "Relative work_dir should be created"

        # 3. Train
        q.train(x, [ds])

        # 4. Predict
        X_new = pd.DataFrame({
            'x1': [1.5, 3.5],
            'x2': [1.5, 1.5]
        })

        results = q.predict(X_new)

        # 5. Verify results
        assert 'relative_path_test' in results
        y_pred = results['relative_path_test']

        for _, row in X_new.iterrows():
            expected = y1_func(t_steps, row['x1'], row['x2'])
            actual = y_pred.loc[row.name].values
            assert np.allclose(actual, expected, atol=0.01), \
                f"Prediction mismatch for row {row.name}: {actual} vs {expected}"

    finally:
        # Clean up
        if relative_work_dir.exists():
            shutil.rmtree(relative_work_dir)


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


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_quasar_validation():
    """
    Integration test for the validation feature.
    Verifies SRMSE behavior across three scenarios:
    1. Training Data (Should be ~0)
    2. Unseen Clean Data (Should be small)
    3. Noisy Data (Should be significant)
    """
    # 1. Setup Training Data (Same as test_quasar_integration)
    X_train = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'x2': [1.0, 2.0, 1.0, 2.0, 1.0]
    })
    t_steps = np.linspace(0, 2, 21)

    y_data_train = {}
    for i, row in X_train.iterrows():
        curve = y1_func(t_steps, row['x1'], row['x2'])
        y_data_train[i] = curve
    Y_train = pd.DataFrame(y_data_train).T

    ds_name = 'ds_0'
    ds_train = DatasetConfig(
        name=ds_name,
        data=Y_train,
        kriging_config=KrigingConfig(basis_function=2)  # Linear basis
    )

    # 2. Train
    q = Quasar(keep_work_dir=True)
    q.train(X_train, [ds_train])

    # --- Scenario 1: Validate on Training Data ---
    # Should be perfect reconstruction
    val_res_train = q.validate(X_train, [ds_train])
    summary_train = val_res_train.summary()

    assert summary_train.loc[ds_name, 'SRMSE'] < 1e-6
    assert summary_train.loc[ds_name, 'RMSE'] < 1e-6

    # --- Scenario 2: Validate on Unseen Clean Data ---
    X_val = pd.DataFrame({
        'x1': [1.5, 3.5, 2.5, 4.5],
        'x2': [1.5, 1.5, 2.5, 0.5]
    })

    y_data_val_clean = {}
    for i, row in X_val.iterrows():
        curve = y1_func(t_steps, row['x1'], row['x2'])
        y_data_val_clean[i] = curve
    Y_val_clean = pd.DataFrame(y_data_val_clean).T

    ds_val_clean = DatasetConfig(name=ds_name, data=Y_val_clean)
    val_res_clean = q.validate(X_val, [ds_val_clean])
    summary_clean = val_res_clean.summary()

    # Error should be small (interpolation error)
    assert summary_clean.loc[ds_name, 'SRMSE'] < 0.05

    # --- Scenario 3: Validate on Noisy Data ---
    y_data_val_noisy = {}
    noise_std = 0.5
    np.random.seed(42)
    for i, row in X_val.iterrows():
        curve = y1_func(t_steps, row['x1'], row['x2'])
        curve += np.random.normal(0, noise_std, size=curve.shape)
        y_data_val_noisy[i] = curve
    Y_val_noisy = pd.DataFrame(y_data_val_noisy).T

    ds_val_noisy = DatasetConfig(name=ds_name, data=Y_val_noisy)
    val_res_noisy = q.validate(X_val, [ds_val_noisy])
    summary_noisy = val_res_noisy.summary()

    # Error should reflect noise
    assert summary_noisy.loc[ds_name, 'SRMSE'] > 0.05

    # 6. Check HTML export (smoke test)
    html_path = Path(__file__).parent / 'reports' / 'validation_report.html'
    html_path.parent.mkdir(parents=True, exist_ok=True)
    val_res_noisy.save_html(str(html_path))
    assert html_path.exists()
