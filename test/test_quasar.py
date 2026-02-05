import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quasarpy.quasar import DatasetConfig, KrigingConfig, Quasar
from quasarpy.validation import ConfigSearchResult, LearningCurveResult

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


# ============================================================================
# Learning Curve Tests
# ============================================================================

def test_learning_curve_train_sizes_calculation():
    """
    Unit test for verifying train_sizes are computed correctly from n_slices.
    """
    # With 100 samples and n_slices=10, no min_samples -> default behavior
    # Using np.linspace logic: linspace(10, 100, 10) = [10, 20, ..., 100]
    n_samples = 100
    n_slices = 10
    min_samples = n_samples // n_slices  # default
    expected = np.linspace(min_samples, n_samples, n_slices, dtype=int).tolist()

    assert expected == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # With 50 samples and n_slices=5, no min_samples
    n_samples = 50
    n_slices = 5
    min_samples = n_samples // n_slices
    expected = np.linspace(min_samples, n_samples, n_slices, dtype=int).tolist()

    assert expected == [10, 20, 30, 40, 50]

    # With min_samples=15, n_samples=30, n_slices=5
    # linspace(15, 30, 5) = [15, 18, 22, 26, 30] (approximately)
    n_samples = 30
    n_slices = 5
    min_samples = 15
    expected = np.linspace(min_samples, n_samples, n_slices, dtype=int).tolist()

    assert expected == [15, 18, 22, 26, 30]

    # With min_samples=20, n_samples=30, n_slices=3
    # linspace(20, 30, 3) = [20, 25, 30]
    n_samples = 30
    n_slices = 3
    min_samples = 20
    expected = np.linspace(min_samples, n_samples, n_slices, dtype=int).tolist()

    assert expected == [20, 25, 30]


def test_learning_curve_result_summary():
    """
    Unit test for LearningCurveResult.summary() method.
    """
    # Create mock results structure
    results = {
        'dataset_1': {
            10: {'metrics': {'RMSE': 1.0, 'SRMSE': 0.5}, 'x_val': pd.DataFrame()},
            20: {'metrics': {'RMSE': 0.8, 'SRMSE': 0.4}, 'x_val': pd.DataFrame()},
        },
        'dataset_2': {
            10: {'metrics': {'RMSE': 2.0, 'SRMSE': 0.6}, 'x_val': pd.DataFrame()},
            20: {'metrics': {'RMSE': 1.5, 'SRMSE': 0.45}, 'x_val': pd.DataFrame()},
        }
    }

    lc_result = LearningCurveResult(results)

    # Test summary
    summary = lc_result.summary()
    assert summary.index.names == ['Dataset', 'TrainSize']
    assert 'RMSE' in summary.columns
    assert 'SRMSE' in summary.columns
    assert summary.loc[('dataset_1', 10), 'RMSE'] == 1.0
    assert summary.loc[('dataset_2', 20), 'SRMSE'] == 0.45

    # Test properties
    assert lc_result.dataset_names == ['dataset_1', 'dataset_2']
    assert lc_result.train_sizes == [10, 20]
    assert 'RMSE' in lc_result.metric_names
    assert 'SRMSE' in lc_result.metric_names


def test_learning_curve_result_plot():
    """
    Unit test for LearningCurveResult.plot() method.
    """
    results = {
        'dataset_1': {
            10: {'metrics': {'RMSE': 1.0, 'SRMSE': 0.5}, 'x_val': pd.DataFrame()},
            20: {'metrics': {'RMSE': 0.8, 'SRMSE': 0.4}, 'x_val': pd.DataFrame()},
            30: {'metrics': {'RMSE': 0.6, 'SRMSE': 0.3}, 'x_val': pd.DataFrame()},
        }
    }

    lc_result = LearningCurveResult(results)
    fig = lc_result.plot('SRMSE')

    # Check that figure was created with correct data
    assert len(fig.data) == 1  # One trace for dataset_1
    assert fig.data[0].x == (10, 20, 30)
    assert fig.data[0].y == (0.5, 0.4, 0.3)
    assert fig.layout.title.text == "Learning Curve: SRMSE"


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_learning_curve_integration():
    """
    Integration test for learning_curve feature.
    Verifies that SRMSE improves (decreases) as training size increases.
    Tests with two datasets to verify multi-dataset support in dashboard.
    """
    # Generate larger training dataset for meaningful learning curve
    np.random.seed(42)
    n_train = 30  # 30 training samples

    X_train = pd.DataFrame({
        'x1': np.random.uniform(1, 5, n_train),
        'x2': np.random.uniform(0.5, 2.5, n_train)
    })

    t_steps = np.linspace(0, 2, 21)

    # Generate training curves for both datasets
    y_data_train_1 = {}
    y_data_train_2 = {}
    for i, row in X_train.iterrows():
        y_data_train_1[i] = y1_func(t_steps, row['x1'], row['x2'])
        y_data_train_2[i] = y2_func(t_steps, row['x1'], row['x2'])
    Y_train_1 = pd.DataFrame(y_data_train_1).T
    Y_train_2 = pd.DataFrame(y_data_train_2).T

    ds_name_1 = 'lc_dataset_1'
    ds_name_2 = 'lc_dataset_2'
    ds_train_1 = DatasetConfig(
        name=ds_name_1,
        data=Y_train_1,
        kriging_config=KrigingConfig(basis_function=2)
    )
    ds_train_2 = DatasetConfig(
        name=ds_name_2,
        data=Y_train_2,
        kriging_config=KrigingConfig(basis_function=2)
    )

    # Validation data (unseen points)
    X_val = pd.DataFrame({
        'x1': [1.5, 2.5, 3.5, 4.5],
        'x2': [1.0, 1.5, 2.0, 1.25]
    })

    y_data_val_1 = {}
    y_data_val_2 = {}
    for i, row in X_val.iterrows():
        y_data_val_1[i] = y1_func(t_steps, row['x1'], row['x2'])
        y_data_val_2[i] = y2_func(t_steps, row['x1'], row['x2'])
    Y_val_1 = pd.DataFrame(y_data_val_1).T
    Y_val_2 = pd.DataFrame(y_data_val_2).T

    ds_val_1 = DatasetConfig(name=ds_name_1, data=Y_val_1)
    ds_val_2 = DatasetConfig(name=ds_name_2, data=Y_val_2)

    # Run learning curve
    q = Quasar(keep_work_dir=True)
    lc_result = q.learning_curve(
        x=X_train,
        datasets=[ds_train_1, ds_train_2],
        x_val=X_val,
        val_datasets=[ds_val_1, ds_val_2],
        n_slices=5  # Will test with 6, 12, 18, 24, 30 samples
    )

    # Check structure
    assert ds_name_1 in lc_result.dataset_names
    assert ds_name_2 in lc_result.dataset_names
    assert len(lc_result.train_sizes) == 5

    # Check that SRMSE generally improves (decreases) with more data for both datasets
    summary = lc_result.summary()
    for ds_name in [ds_name_1, ds_name_2]:
        srmse_values = summary.loc[ds_name, 'SRMSE'].values

        # The last (largest training set) should have lower SRMSE than the first
        assert srmse_values[-1] < srmse_values[0], \
            f'Expected SRMSE to decrease for {ds_name}: first={srmse_values[0]}, last={srmse_values[-1]}'

    # Check HTML export
    html_path = Path(__file__).parent / 'reports' / 'learning_curve_report.html'
    html_path.parent.mkdir(parents=True, exist_ok=True)
    lc_result.save_html(str(html_path))
    assert html_path.exists()


def test_learning_curve_store_predictions_false():
    """
    Unit test to verify store_predictions=False omits y_pred and y_true.
    """
    results_with = {
        'ds': {
            10: {
                'metrics': {'RMSE': 1.0},
                'y_pred': pd.DataFrame({'a': [1, 2]}),
                'y_true': pd.DataFrame({'a': [1, 2]}),
                'x_val': pd.DataFrame()
            }
        }
    }

    results_without = {
        'ds': {
            10: {
                'metrics': {'RMSE': 1.0},
                'x_val': pd.DataFrame()
            }
        }
    }

    # With predictions
    lc_with = LearningCurveResult(results_with)
    assert 'y_pred' in lc_with.results['ds'][10]
    assert 'y_true' in lc_with.results['ds'][10]

    # Without predictions
    lc_without = LearningCurveResult(results_without)
    assert 'y_pred' not in lc_without.results['ds'][10]
    assert 'y_true' not in lc_without.results['ds'][10]

    # Both should still have working summary()
    assert 'RMSE' in lc_with.summary().columns
    assert 'RMSE' in lc_without.summary().columns


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_learning_curve_final_slice_matches_direct_validate():
    """
    Regression test: the last slice of learning_curve (using full dataset)
    should produce identical results to a direct train() + validate() call.
    """
    np.random.seed(42)
    n_train = 20

    X_train = pd.DataFrame({
        'x1': np.random.uniform(1, 5, n_train),
        'x2': np.random.uniform(0.5, 2.5, n_train)
    })

    t_steps = np.linspace(0, 2, 21)

    y_data_train = {}
    for i, row in X_train.iterrows():
        y_data_train[i] = y1_func(t_steps, row['x1'], row['x2'])
    Y_train = pd.DataFrame(y_data_train).T

    ds_name = 'consistency_test'
    ds_train = DatasetConfig(
        name=ds_name,
        data=Y_train,
        kriging_config=KrigingConfig(basis_function=2)
    )

    # Validation data
    X_val = pd.DataFrame({
        'x1': [1.5, 2.5, 3.5],
        'x2': [1.0, 1.5, 2.0]
    })

    y_data_val = {}
    for i, row in X_val.iterrows():
        y_data_val[i] = y1_func(t_steps, row['x1'], row['x2'])
    Y_val = pd.DataFrame(y_data_val).T

    ds_val = DatasetConfig(name=ds_name, data=Y_val)

    # --- Method 1: Direct train + validate ---
    q1 = Quasar(keep_work_dir=True)
    q1.train(X_train, [ds_train])
    direct_result = q1.validate(X_val, [ds_val])
    direct_srmse = direct_result.summary().loc[ds_name, 'SRMSE']

    # --- Method 2: Learning curve (last slice uses full dataset) ---
    q2 = Quasar(keep_work_dir=True)
    lc_result = q2.learning_curve(
        x=X_train,
        datasets=[ds_train],
        x_val=X_val,
        val_datasets=[ds_val],
        n_slices=4  # Last slice will use all 20 samples
    )

    # Get the SRMSE from the last slice (should use all training data)
    last_size = lc_result.train_sizes[-1]
    lc_srmse = lc_result.results[ds_name][last_size]['metrics']['SRMSE']

    # They should be identical (or very close due to floating point)
    assert last_size == n_train, f'Last slice size {last_size} != n_train {n_train}'
    assert np.isclose(direct_srmse, lc_srmse, rtol=1e-6), \
        f'Direct SRMSE ({direct_srmse}) != Learning curve SRMSE ({lc_srmse})'


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_learning_curve_with_nonsequential_indices():
    """
    Test that learning_curve works correctly when DataFrames have
    non-sequential indices (e.g., from shuffling or filtering).
    """
    np.random.seed(42)
    n_train = 20

    # Create data with non-sequential indices
    random_indices = np.random.permutation(range(100, 100 + n_train))

    X_train = pd.DataFrame({
        'x1': np.random.uniform(1, 5, n_train),
        'x2': np.random.uniform(0.5, 2.5, n_train)
    }, index=random_indices)

    t_steps = np.linspace(0, 2, 21)

    y_data_train = {}
    for i, row in X_train.iterrows():
        y_data_train[i] = y1_func(t_steps, row['x1'], row['x2'])
    Y_train = pd.DataFrame(y_data_train).T

    # Verify indices match
    assert list(X_train.index) == list(Y_train.index), 'X and Y indices should match'

    ds_name = 'nonseq_index_test'
    ds_train = DatasetConfig(
        name=ds_name,
        data=Y_train,
        kriging_config=KrigingConfig(basis_function=2)
    )

    # Validation data (also with non-sequential indices)
    val_indices = [500, 501, 502]
    X_val = pd.DataFrame({
        'x1': [1.5, 2.5, 3.5],
        'x2': [1.0, 1.5, 2.0]
    }, index=val_indices)

    y_data_val = {}
    for i, row in X_val.iterrows():
        y_data_val[i] = y1_func(t_steps, row['x1'], row['x2'])
    Y_val = pd.DataFrame(y_data_val).T

    ds_val = DatasetConfig(name=ds_name, data=Y_val)

    # Run learning curve
    q = Quasar(keep_work_dir=True)
    lc_result = q.learning_curve(
        x=X_train,
        datasets=[ds_train],
        x_val=X_val,
        val_datasets=[ds_val],
        n_slices=4
    )

    # Check that SRMSE values are reasonable (not NaN or huge)
    summary = lc_result.summary()
    for train_size in lc_result.train_sizes:
        srmse = summary.loc[(ds_name, train_size), 'SRMSE']
        assert not np.isnan(srmse), f'SRMSE is NaN for train_size={train_size}'
        assert srmse < 1.0, f'SRMSE ({srmse}) unexpectedly large for train_size={train_size}'

    # Verify SRMSE decreases (or stays similar) as training size increases
    srmse_values = summary.loc[ds_name, 'SRMSE'].values
    assert srmse_values[-1] <= srmse_values[0] * 1.1, \
        f'SRMSE did not decrease: first={srmse_values[0]}, last={srmse_values[-1]}'


# =============================================================================
# Config Search Tests
# =============================================================================


def test_config_search_grid_size():
    """
    Unit test verifying that config_search generates the correct number of configs.
    """
    import itertools

    # Test default values
    basis_functions = [1, 2, 3, 4, 5]
    stationarities = [0, 1, 2, 3, 4]
    pulsations = [1.5708]
    nugget_effects = [0.4, 0.8, 1.2, 1.6, 2.0]

    expected_count = len(basis_functions) * len(stationarities) * len(pulsations) * len(nugget_effects)
    assert expected_count == 125

    configs = list(itertools.product(basis_functions, stationarities, pulsations, nugget_effects))
    assert len(configs) == 125

    # Test custom values
    bf = [2, 3]
    st = [3, 4]
    puls = [1.5]
    nug = [0.5, 1.0]

    expected_count = len(bf) * len(st) * len(puls) * len(nug)
    assert expected_count == 8


def test_config_search_result_summary():
    """
    Unit test for ConfigSearchResult.summary() method.
    """
    from quasarpy.quasar import KrigingConfig

    # Create mock results structure
    config1 = KrigingConfig(basis_function=2, stationarity=4, pulsation=1.5708, nugget_effect=0.4)
    config2 = KrigingConfig(basis_function=3, stationarity=4, pulsation=1.5708, nugget_effect=0.8)

    results = {
        'dataset_1': {
            0: {
                'metrics': {'RMSE': 1.0, 'MAE': 0.8, 'Peak Error': 0.5, 'SRMSE': 0.1},
                'config': config1,
                'x_val': pd.DataFrame()
            },
            1: {
                'metrics': {'RMSE': 0.9, 'MAE': 0.7, 'Peak Error': 0.4, 'SRMSE': 0.08},
                'config': config2,
                'x_val': pd.DataFrame()
            },
        },
        'dataset_2': {
            0: {
                'metrics': {'RMSE': 2.0, 'MAE': 1.5, 'Peak Error': 1.0, 'SRMSE': 0.2},
                'config': config1,
                'x_val': pd.DataFrame()
            },
            1: {
                'metrics': {'RMSE': 1.8, 'MAE': 1.3, 'Peak Error': 0.9, 'SRMSE': 0.15},
                'config': config2,
                'x_val': pd.DataFrame()
            },
        }
    }

    cs_result = ConfigSearchResult(results)

    # Test summary
    summary = cs_result.summary()
    assert summary.index.name == 'Dataset'
    assert 'basis_function' in summary.columns
    assert 'stationarity' in summary.columns
    assert 'pulsation' in summary.columns
    assert 'nugget_effect' in summary.columns
    assert 'RMSE' in summary.columns
    assert 'SRMSE' in summary.columns

    # Check specific values
    ds1_rows = summary.loc['dataset_1']
    assert len(ds1_rows) == 2
    assert ds1_rows.iloc[0]['basis_function'] == 2
    assert ds1_rows.iloc[1]['basis_function'] == 3

    # Test properties
    assert cs_result.dataset_names == ['dataset_1', 'dataset_2']
    assert 'RMSE' in cs_result.metric_names
    assert 'SRMSE' in cs_result.metric_names
    assert len(cs_result.configs) == 2


def test_config_search_result_best():
    """
    Unit test for ConfigSearchResult.best() method with different weight combinations.
    """
    from quasarpy.quasar import KrigingConfig

    config1 = KrigingConfig(basis_function=2, stationarity=4, pulsation=1.5708, nugget_effect=0.4)
    config2 = KrigingConfig(basis_function=3, stationarity=4, pulsation=1.5708, nugget_effect=0.8)
    config3 = KrigingConfig(basis_function=4, stationarity=3, pulsation=1.5708, nugget_effect=1.2)

    # config1: SRMSE=0.1, MAE=0.9
    # config2: SRMSE=0.2, MAE=0.3  (best MAE)
    # config3: SRMSE=0.05, MAE=0.8 (best SRMSE)
    results = {
        'dataset_1': {
            0: {
                'metrics': {'RMSE': 1.0, 'MAE': 0.9, 'Peak Error': 0.5, 'SRMSE': 0.1},
                'config': config1,
                'x_val': pd.DataFrame()
            },
            1: {
                'metrics': {'RMSE': 0.9, 'MAE': 0.3, 'Peak Error': 0.4, 'SRMSE': 0.2},
                'config': config2,
                'x_val': pd.DataFrame()
            },
            2: {
                'metrics': {'RMSE': 1.1, 'MAE': 0.8, 'Peak Error': 0.6, 'SRMSE': 0.05},
                'config': config3,
                'x_val': pd.DataFrame()
            },
        }
    }

    cs_result = ConfigSearchResult(results)

    # Test default (SRMSE only) - config3 has lowest SRMSE (0.05)
    best = cs_result.best()
    assert best['dataset_1'].basis_function == 4
    assert best['dataset_1'].nugget_effect == 1.2

    # Test SRMSE only explicitly
    best = cs_result.best(weights={'SRMSE': 1.0})
    assert best['dataset_1'].basis_function == 4

    # Test MAE only - config2 has lowest MAE (0.3)
    best = cs_result.best(weights={'MAE': 1.0})
    assert best['dataset_1'].basis_function == 3
    assert best['dataset_1'].nugget_effect == 0.8

    # Test weighted combination (SRMSE=1.0, MAE=1.0)
    # config1: (0.1 + 0.9) / 2 = 0.5
    # config2: (0.2 + 0.3) / 2 = 0.25  <- best
    # config3: (0.05 + 0.8) / 2 = 0.425
    best = cs_result.best(weights={'SRMSE': 1.0, 'MAE': 1.0})
    assert best['dataset_1'].basis_function == 3


def test_config_search_result_plot():
    """
    Unit test for ConfigSearchResult.plot() method.
    """
    from quasarpy.quasar import KrigingConfig

    config1 = KrigingConfig(basis_function=2, stationarity=4, pulsation=1.5708, nugget_effect=0.4)
    config2 = KrigingConfig(basis_function=3, stationarity=4, pulsation=1.5708, nugget_effect=0.8)

    results = {
        'dataset_1': {
            0: {
                'metrics': {'RMSE': 1.0, 'MAE': 0.8, 'Peak Error': 0.5, 'SRMSE': 0.1},
                'config': config1,
                'x_val': pd.DataFrame()
            },
            1: {
                'metrics': {'RMSE': 0.9, 'MAE': 0.7, 'Peak Error': 0.4, 'SRMSE': 0.08},
                'config': config2,
                'x_val': pd.DataFrame()
            },
        }
    }

    cs_result = ConfigSearchResult(results)
    fig = cs_result.plot('SRMSE')

    # Check that figure was created with correct data
    assert len(fig.data) == 1  # One trace for dataset_1
    # Configs should be sorted by SRMSE (0.08, then 0.1)
    assert fig.data[0].y[0] == 0.08
    assert fig.data[0].y[1] == 0.1
    assert fig.layout.title.text == 'Configuration Comparison: SRMSE'


def test_config_search_result_save_html(tmp_path):
    """
    Unit test for ConfigSearchResult.save_html() method.
    """
    from quasarpy.quasar import KrigingConfig

    config1 = KrigingConfig(basis_function=2, stationarity=4, pulsation=1.5708, nugget_effect=0.4)
    config2 = KrigingConfig(basis_function=3, stationarity=4, pulsation=1.5708, nugget_effect=0.8)

    results = {
        'dataset_1': {
            0: {
                'metrics': {'RMSE': 1.0, 'MAE': 0.8, 'Peak Error': 0.5, 'SRMSE': 0.1},
                'config': config1,
                'x_val': pd.DataFrame()
            },
            1: {
                'metrics': {'RMSE': 0.9, 'MAE': 0.7, 'Peak Error': 0.4, 'SRMSE': 0.08},
                'config': config2,
                'x_val': pd.DataFrame()
            },
        }
    }

    cs_result = ConfigSearchResult(results)

    # Save to HTML
    html_path = tmp_path / 'config_search_report.html'
    cs_result.save_html(str(html_path))

    assert html_path.exists()

    # Check content
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    assert 'Configuration Search Report' in content
    assert 'dataset_1' in content
    assert 'SRMSE' in content
    assert 'input type="range"' in content  # Weight sliders
    assert 'updateWeights' in content  # JavaScript function


@pytest.mark.skipif(not QUASAR_INSTALLED, reason='Quasar not installed')
def test_config_search_integration():
    """
    Integration test for config_search feature.
    Verifies that different configurations produce different SRMSE values
    and that the best() method correctly identifies the optimal config.
    """
    # Generate training dataset
    np.random.seed(42)

    X_train = pd.DataFrame({
        'x1': np.linspace(1, 5, 10),
        'x2': np.random.uniform(0.5, 2.5, 10)
    })

    t_steps = np.linspace(0, 2, 21)

    y_data = {}
    for label, row in X_train.iterrows():
        curve = y1_func(t_steps, row['x1'], row['x2'])
        y_data[label] = curve

    Y_train = pd.DataFrame(y_data).T

    ds_train = DatasetConfig(
        name='config_search_test',
        data=Y_train,
        kriging_config=KrigingConfig()  # Will be overwritten by config_search
    )

    # Generate validation data
    X_val = pd.DataFrame({
        'x1': [1.5, 2.5, 3.5, 4.5],
        'x2': [1.0, 1.5, 2.0, 1.0]
    })

    y_data_val = {}
    for label, row in X_val.iterrows():
        curve = y1_func(t_steps, row['x1'], row['x2'])
        # Add small noise
        curve += np.random.normal(0, 0.1, size=curve.shape)
        y_data_val[label] = curve

    Y_val = pd.DataFrame(y_data_val).T

    ds_val = DatasetConfig(name='config_search_test', data=Y_val)

    # Run config search with limited parameter space for speed
    q = Quasar(keep_work_dir=True)
    cs_result = q.config_search(
        x=X_train,
        datasets=[ds_train],
        x_val=X_val,
        val_datasets=[ds_val],
        basis_functions=[2, 3],  # linear, quadratic
        stationarities=[3, 4],   # exp, matern32
        nugget_effects=[0.4, 0.8]
    )

    # Should have 2 * 2 * 1 * 2 = 8 configurations
    assert len(cs_result.configs) == 8

    # Check summary structure
    summary = cs_result.summary()
    assert len(summary) == 8
    assert 'SRMSE' in summary.columns
    assert 'basis_function' in summary.columns

    # Check that SRMSE values are reasonable (not NaN, not huge)
    srmse_values = summary['SRMSE'].values
    assert all(not np.isnan(v) for v in srmse_values), 'SRMSE values contain NaN'
    assert all(v < 1.0 for v in srmse_values), 'SRMSE values are unexpectedly large'

    # Check best() returns a valid config
    best = cs_result.best()
    assert 'config_search_test' in best
    best_config = best['config_search_test']
    assert best_config.basis_function in [2, 3]
    assert best_config.stationarity in [3, 4]
    assert best_config.nugget_effect in [0.4, 0.8]

    # Verify best config has the lowest (or tied for lowest) SRMSE
    best_srmse = min(srmse_values)
    # Allow for floating point tolerance
    assert any(
        np.isclose(res['metrics']['SRMSE'], best_srmse, rtol=1e-6) and
        res['config'].basis_function == best_config.basis_function and
        res['config'].nugget_effect == best_config.nugget_effect
        for res in cs_result.results['config_search_test'].values()
    ), 'Best config does not match the one with lowest SRMSE'

    # Save report
    report_path = Path(__file__).parent / 'reports' / 'config_search_report.html'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cs_result.save_html(str(report_path))
