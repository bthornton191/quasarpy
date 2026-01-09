# Copilot Instructions for quasarpy

## Project Overview
`quasarpy` is a Python wrapper for the ODYSSEE CAE Quasar Engine, enabling programmatic surrogate modeling (Kriging) for simulation data. It wraps `QuasarEngine_2.exe` and handles input/output file formatting, execution, and result parsing.

## Architecture
- **[quasarpy/quasar.py](../quasarpy/quasar.py)**: Core `Quasar` class with `train()` and `predict()` workflow. Manages temp directories, CSV file formatting, and subprocess execution
- **[quasarpy/validation.py](../quasarpy/validation.py)**: `ValidationResult` class with metrics (RMSE, MAE, SRMSE) and interactive Plotly dashboards
- **[quasarpy/qsr_scripts/](../quasarpy/qsr_scripts/)**: QSR script files copied to work directories at runtime

## Key Patterns

### DataFrames as Primary Data Structures
All data flows through pandas DataFrames:
- **X (inputs)**: Rows = samples, columns = parameters (e.g., `x1`, `x2`)
- **Y (outputs)**: Rows = samples, columns = curve points (time series values)

### Configuration via Dataclasses
```python
# Always use DatasetConfig + KrigingConfig for training
ds = DatasetConfig(
    name='my_dataset',
    data=Y_train,
    kriging_config=KrigingConfig(basis_function=2, stationarity=4)
)
```

### File Format Requirements (Critical)
Quasar expects specific CSV formats - see `_write_x_file()` and `_write_y_file()`:
- Semicolon delimiters (`;`)
- Scientific notation: `f'{x:1.8E}'`
- X files include headers, Y files do not

### Working Directory Lifecycle
`Quasar` creates temp directories by default and cleans up on deletion. Use `keep_work_dir=True` for debugging.

## Development Workflow

### Running Tests
```bash
# Unit tests only (no Quasar installation needed)
pytest test/test_quasar.py -v

# Integration tests (requires ODYSSEE CAE installed)
# Skipped automatically if ODYSSEE_CAE_INSTALLDIR or ODYSSEE_SOLVER_INSTALLDIR not set
pytest test/test_quasar.py -v -k "integration or validation"
```

### Test Fixtures ([test/conftest.py](../test/conftest.py))
- `mock_quasar_exe`: Creates dummy executable for unit tests
- `sample_data`: Standard X, Y DataFrames for testing

### Demo Notebook
Run [demo.ipynb](../demo.ipynb) for interactive exploration with validation dashboards.

## External Dependencies
- **ODYSSEE CAE**: Required for actual predictions. Set `ODYSSEE_CAE_INSTALLDIR` or `ODYSSEE_SOLVER_INSTALLDIR` environment variable
- Core: `pandas`, `numpy`, `plotly`, `ipywidgets`

## Common Tasks

### Adding a New Solver Type
1. Add configuration dataclass (like `KrigingConfig`)
2. Update `DatasetConfig.get_args()` to serialize solver params
3. Reference [Simulation.qsr](../quasarpy/qsr_scripts/Simulation.qsr) for argument format (lines 8-45 document all solver args)

### Adding Validation Metrics
Add to `Quasar.validate()` method and update `ValidationResult.summary()`.
