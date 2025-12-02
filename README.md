# quasarpy

A Python wrapper for the Quasar Engine (ODYSSEE CAE), enabling programmatic control over Quasar simulations, data formatting, and execution.

## Description

`quasarpy` allows you to interface with the ODYSSEE CAE Quasar engine directly from Python. It handles:
- Formatting input files (`X_FILE.csv`, `Y_FILE.csv`) with the specific delimiters and scientific notation required by Quasar.
- Managing temporary working directories and script files.
- Executing the `QuasarEngine_2.exe` process.
- Parsing output files (`YN.csv`) back into Pandas DataFrames.

## Prerequisites

- **ODYSSEE CAE**: You must have ODYSSEE CAE installed. The package looks for the `ODYSSEE_CAE_INSTALLDIR` or `ODYSSEE_SOLVER_INSTALLDIR` environment variables to locate `QuasarEngine_2.exe`.

## Installation

You can install `quasarpy` directly from GitHub:

```bash
pip install git+https://github.com/bthornton191/quasarpy.git
```

## Usage

```python
import pandas as pd
from quasarpy.quasar import Quasar, DatasetConfig, KrigingConfig

# 1. Prepare your data
# X: Design of Experiments (Parameters)
X_train = pd.DataFrame({
    'x1': [1.0, 2.0, 3.0],
    'x2': [0.5, 1.5, 2.5]
})

# Y: Simulation Results (Curves/Time-series)
Y_train = pd.DataFrame({
    't0': [10.0, 20.0, 30.0],
    't1': [15.0, 25.0, 35.0]
})

# 2. Configure the dataset
ds = DatasetConfig(
    name='my_dataset',
    data=Y_train,
    kriging_config=KrigingConfig(basis_function=2)
)

# 3. Initialize Quasar
# Automatically detects executable from environment variables
q = Quasar()

# 4. Train
q.train(X_train, [ds])

# 5. Predict
X_new = pd.DataFrame({'x1': [1.5], 'x2': [1.0]})
results = q.predict(X_new)

print(results['my_dataset'])
```
