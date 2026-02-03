from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .validation import LearningCurveResult, ValidationResult
from .optimization import (
    ObjectiveConfig, ConstraintConfig, OptimizationResult, run_optimization
)


@dataclass
class KrigingConfig:
    """
    Configuration for the Kriging solver.

    Parameters
    ----------
    basis_function : int, optional
        The basis function type.
        0=none, 1=constant, 2=linear, 3=quadratic, 4=cubic, 5=trigonometric.
        Default is 5.
    stationarity : int, optional
        The stationarity type.
        0=h1, 1=h2, 2=h3, 3=exp, 4=matern32.
        Default is 4.
    pulsation : float, optional
        Pulsation parameter for trigonometric basis function.
        Default is 1.5708.
    nugget_effect : float, optional
        Nugget effect parameter for noise handling.
        Default is 0.5.
    """
    basis_function: int = 5  # 0=none, 1=constant, 2=linear, 3=quadratic, 4=cubic, 5=trigonometric
    stationarity: int = 4    # 0=h1, 1=h2, 2=h3, 3=exp, 4=matern32
    pulsation: float = 1.5708
    nugget_effect: float = 0.5


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset to be modeled.

    Parameters
    ----------
    name : str
        Unique name for the dataset.
    data : pd.DataFrame
        The target data (Y values).
        Rows correspond to samples, columns to curve points.
    solver_id : int, optional
        ID of the solver to use.
        2=Kriging. Default is 2.
    kriging_config : KrigingConfig, optional
        Configuration for the Kriging solver.
        Used only if solver_id is 2.
    """
    name: str
    data: pd.DataFrame
    solver_id: int = 2  # 2=Kriging
    kriging_config: KrigingConfig = field(default_factory=KrigingConfig)

    def get_args(self) -> List[str]:
        """
        Generates the command-line arguments for this dataset's solver configuration.

        Returns
        -------
        List[str]
            A list of string arguments to be passed to the Quasar engine.
        """
        args = []
        if self.solver_id == 2:  # Kriging
            args.extend([
                str(self.kriging_config.basis_function),
                str(self.kriging_config.stationarity),
                str(self.kriging_config.pulsation),
                str(self.kriging_config.nugget_effect)
            ])
        # Add other solvers here if needed
        return args


class Quasar:
    """
    Main interface for interacting with the Quasar Engine (ODYSSEE CAE).

    This class handles the setup, execution, and result retrieval for Quasar simulations.
    It manages temporary directories, formats input/output files, and constructs the
    command-line arguments required by the engine.

    Parameters
    ----------
    quasar_exe : Union[str, Path], optional
        Path to the Quasar executable.
        If None, attempts to find it using ODYSSEE_CAE_INSTALLDIR or
        ODYSSEE_SOLVER_INSTALLDIR environment variables.
    qsr_scripts_dir : Union[str, Path], optional
        Path to the directory containing .qsr scripts.
        If None, uses the 'qsr_scripts' directory within the package.
    work_dir : Union[str, Path], optional
        Directory to use for temporary files.
        If None, a temporary directory is created.
    keep_work_dir : bool, optional
        If True, the working directory is not deleted when the object is
        garbage collected. Default is False.

    Attributes
    ----------
    quasar_exe : Path
        Path to the QuasarEngine_2.exe executable.
    qsr_scripts_dir : Path
        Directory containing the .qsr scripts (Simulation.qsr, etc.).
    work_dir : Path
        The working directory for the current session.
    keep_work_dir : bool
        Whether to keep the working directory after the object is destroyed.
    """

    def __init__(
        self,
        quasar_exe: Optional[Union[str, Path]] = None,
        qsr_scripts_dir: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        keep_work_dir: bool = False
    ):
        """
        Initializes the Quasar interface.

        Raises
        ------
        FileNotFoundError
            If the Quasar executable cannot be found.
        """
        if quasar_exe is not None:
            self.quasar_exe = Path(quasar_exe)
        elif 'ODYSSEE_CAE_INSTALLDIR' in os.environ:
            self.quasar_exe = Path(os.environ['ODYSSEE_CAE_INSTALLDIR']) / 'QuasarEngine_2.exe'
        elif 'ODYSSEE_SOLVER_INSTALLDIR' in os.environ:
            self.quasar_exe = Path(os.environ['ODYSSEE_SOLVER_INSTALLDIR']) / 'QuasarEngine_2.exe'
        else:
            raise FileNotFoundError('Could not find QuasarEngine_2.exe. Please provide path or set '
                                    'ODYSSEE_CAE_INSTALLDIR or ODYSSEE_SOLVER_INSTALLDIR '
                                    'environment variables.')

        if qsr_scripts_dir is None:
            self.qsr_scripts_dir = Path(__file__).parent / 'qsr_scripts'
        else:
            self.qsr_scripts_dir = Path(qsr_scripts_dir)

        self.keep_work_dir = keep_work_dir

        if work_dir is None:
            self.work_dir = Path(tempfile.mkdtemp())
            self._temp_dir = True
        else:
            self.work_dir = Path(work_dir).absolute()
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = False

        self._setup_scripts()

    def _setup_scripts(self):
        """Copies necessary QSR scripts to the working directory."""
        for script in self.qsr_scripts_dir.glob('*.qsr'):
            shutil.copy(script, self.work_dir / script.name)

    def __del__(self):
        """Clean up the temporary directory if keep_work_dir is False."""
        if self._temp_dir and not self.keep_work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def train(self, x: pd.DataFrame, datasets: List[DatasetConfig]):
        """
        Sets up the training data and generates the Quasar script command.

        This method prepares the working directory by writing the input (X) and
        target (Y) files, and then generates the command arguments required to
        run the training process.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame containing the input parameters (DOE).
            Rows are samples, columns are parameters.
        datasets : List[DatasetConfig]
            List of DatasetConfig objects, each containing the target data
            and solver settings for a specific output.
        """
        self.x = x
        self.datasets = datasets

        # Write X file
        self._write_x_file(x, 'X_FILE.csv')

        # Write Validation file (empty for now)
        (self.work_dir / 'VALIDATION_FILE.csv').touch()

        # Write Y files for each dataset
        for ds in datasets:
            ds_dir = self.work_dir / ds.name
            ds_dir.mkdir(exist_ok=True)
            self._write_y_file(ds.data, ds_dir / 'Y_FILE.csv')

        # Generate the command arguments
        self._generate_command()

    def predict(self, x_new: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Runs predictions for new input parameters.

        This method writes the new input parameters to a file, executes the Quasar
        engine, and reads the generated output files.

        Parameters
        ----------
        x_new : pd.DataFrame
            DataFrame containing the new input parameters for which predictions
            are desired.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary where keys are dataset names and values are DataFrames
            containing the predicted curves. Rows correspond to the samples in x_new.

        Raises
        ------
        RuntimeError
            If the Quasar engine execution fails.
        """
        # Write XN file
        self._write_x_file(x_new, 'XN_FILE.csv')

        # Run Quasar
        self._run_quasar()

        # Read results
        results = {}
        for ds in self.datasets:
            results[ds.name] = self._read_y_file(self.work_dir / ds.name / 'YN.csv')

        return results

    def validate(self, x_val: pd.DataFrame, datasets: List[DatasetConfig]) -> ValidationResult:
        """
        Validates the model against a new set of data.

        Calculates the following metrics:
        - RMSE: Root Mean Square Error
        - MAE: Mean Absolute Error
        - Peak Error: Mean absolute difference of peaks
        - SRMSE: Standardized RMSE ($SRMSE = \\frac{RMSE}{\\sigma_{obs}}$)

        Parameters
        ----------
        x_val : pd.DataFrame
            Validation input parameters.
        datasets : List[DatasetConfig]
            List of DatasetConfig objects containing the actual validation data (Y).
            Note: The solver configuration in these objects is ignored.

        Returns
        -------
        ValidationResult
            Object containing metrics and visualization methods.
        """
        # 1. Predict
        predictions = self.predict(x_val)

        results = {}
        for ds in datasets:
            if ds.name not in predictions:
                continue

            y_pred = predictions[ds.name]
            y_true = ds.data

            # Calculate metrics
            # RMSE
            mse = ((y_pred.values - y_true.values) ** 2).mean().mean()
            rmse = np.sqrt(mse)

            # MAE
            mae = np.abs(y_pred.values - y_true.values).mean().mean()
            # Peak Error (Mean absolute difference of peaks)
            peak_error = np.abs(y_pred.max(axis=1).values - y_true.max(axis=1).values).mean()

            # SRMSE
            std_obs = y_true.stack().std()
            if std_obs == 0:
                srmse = 0.0 if rmse == 0 else np.inf
            else:
                srmse = rmse / std_obs

            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'Peak Error': peak_error,
                'SRMSE': srmse
            }

            results[ds.name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_true': y_true,
                'x_val': x_val
            }

        return ValidationResult(results)

    def learning_curve(
        self,
        x: pd.DataFrame,
        datasets: List[DatasetConfig],
        x_val: pd.DataFrame,
        val_datasets: List[DatasetConfig],
        n_slices: int = 10,
        min_samples: Optional[int] = None,
        store_predictions: bool = True
    ) -> LearningCurveResult:
        """
        Performs iterative validation over increasing training dataset sizes.

        This method trains the model on progressively larger subsets of the
        training data and evaluates performance on a fixed validation set,
        enabling analysis of how model accuracy improves with more training data.

        Parameters
        ----------
        x : pd.DataFrame
            Full training input parameters (DOE).
        datasets : List[DatasetConfig]
            List of DatasetConfig objects containing full training data.
        x_val : pd.DataFrame
            Validation input parameters (held constant across all slices).
        val_datasets : List[DatasetConfig]
            List of DatasetConfig objects containing validation data (Y values).
        n_slices : int, optional
            Number of slices to divide the training data into. Default is 10.
            For example, with 100 samples and n_slices=10, training sizes will
            be [10, 20, 30, ..., 100].
        min_samples : int, optional
            Minimum number of samples for the first training slice. Use this
            when your model requires a minimum amount of data to train successfully.
            If None, defaults to n_samples // n_slices. Training sizes will be
            evenly spaced from min_samples to n_samples.
        store_predictions : bool, optional
            If True, stores y_pred and y_true for each training size.
            If False, only stores metrics (lower memory). Default is True.

        Returns
        -------
        LearningCurveResult
            Object containing metrics for each training size with visualization
            methods including summary(), plot(), dashboard(), and save_html().

        Examples
        --------
        >>> q = Quasar()
        >>> lc_result = q.learning_curve(
        ...     x=X_train,
        ...     datasets=[ds_train],
        ...     x_val=X_val,
        ...     val_datasets=[ds_val],
        ...     n_slices=5,
        ...     min_samples=15  # Start with at least 15 training samples
        ... )
        >>> print(lc_result.summary())
        >>> lc_result.plot('SRMSE').show()
        >>> lc_result.dashboard()
        """
        n_samples = len(x)
        for ds in datasets:
            if len(ds.data) != n_samples:
                raise ValueError(
                    f'Dataset "{ds.name}" has {len(ds.data)} rows but X has {n_samples} rows. '
                    f'X and Y must have the same number of samples.'
                )

        # Determine minimum samples for first slice
        if min_samples is None:
            min_samples = n_samples // n_slices

        if min_samples < 1:
            min_samples = 1

        if min_samples > n_samples:
            raise ValueError(
                f'min_samples ({min_samples}) cannot exceed total samples ({n_samples}).'
            )

        # Compute evenly spaced training sizes from min_samples to n_samples
        train_sizes = np.linspace(min_samples, n_samples, n_slices, dtype=int).tolist()
        # Remove duplicates while preserving order (can happen with small ranges)
        train_sizes = list(dict.fromkeys(train_sizes))

        # Initialize results structure: {dataset_name: {train_size: {...}}}
        all_results: Dict[str, Dict[int, Dict]] = {ds.name: {} for ds in val_datasets}

        for size in tqdm(train_sizes, desc='Learning Curve', unit='slice'):
            # Subset training data
            x_subset = x.iloc[:size]
            datasets_subset = [
                DatasetConfig(
                    name=ds.name,
                    data=ds.data.iloc[:size],
                    solver_id=ds.solver_id,
                    kriging_config=ds.kriging_config
                )
                for ds in datasets
            ]

            # Train with subset
            self.train(x_subset, datasets_subset)

            # Validate
            val_result = self.validate(x_val, val_datasets)

            # Store results for each dataset
            for ds_name, res in val_result.results.items():
                if store_predictions:
                    all_results[ds_name][size] = res
                else:
                    # Only store metrics and x_val
                    all_results[ds_name][size] = {
                        'metrics': res['metrics'],
                        'x_val': res['x_val']
                    }

        return LearningCurveResult(all_results)

    def _write_x_file(self, df: pd.DataFrame, filename: str):
        """
        Writes the X (input) file in the format Quasar expects.

        Format:
            Header: param1;param2;...
            Data: val1;val2;... (semicolon separated, scientific notation)

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to write.
        filename : str
            The name of the file to create in the working directory.
        """
        # Quasar expects semicolon separated values
        # Header: param1;param2;...
        # Data: val1;val2;...

        # Format numbers to scientific notation
        df_formatted = df.map(lambda x: f'{x:1.8E}')

        path = self.work_dir / filename
        df_formatted.to_csv(path, sep=';', index=False)
        # with open(path, 'w') as f:
        #     # Write header
        #     f.write(';'.join(df.columns) + '\n')

        #     # Write data
        #     for _, row in df_formatted.iterrows():
        #         f.write(' ' + ';  '.join(row.values) + '\n')

    def _write_y_file(self, df: pd.DataFrame, path: Path):
        """
        Writes the Y (output) file in the format Quasar expects.

        Format:
            No Header
            Data: val1;val2;... (semicolon separated, scientific notation)

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to write.
        path : Path
            The full path to the file to create.
        """
        # Quasar expects semicolon separated values
        # No Header
        # Data: val1;val2;...

        df_formatted = df.map(lambda x: f'{x:1.8E}')

        with open(path, 'w') as f:
            for _, row in df_formatted.iterrows():
                f.write(' ' + ';  '.join(row.values) + '\n')

    def _read_y_file(self, path: Path) -> pd.DataFrame:
        """
        Reads a Y file produced by Quasar.

        Parameters
        ----------
        path : Path
            The path to the file to read.

        Returns
        -------
        pd.DataFrame
            The data read from the file.
        """
        # Quasar output format is semicolon separated values
        # No header, no index
        return pd.read_csv(path, sep=';', header=None)

    def _generate_command(self):
        """
        Generates the command arguments for Simulation.qsr.

        Constructs the list of arguments required to invoke the Quasar engine
        with the 'Simulation.qsr' script, including paths to input/output files
        and solver configurations for each dataset.
        """

        # $args[0]->"WorkingDirPath"
        # $args[1]->"X_FILE.csv"
        # $args[2]->"VALIDATION_FILE.csv"
        # $args[3]->"XN_FILE.csv"
        # $args[4]->-1
        # $args[5]->nbDatasets

        args = [
            str(self.quasar_exe),
            'Simulation.qsr',
            self.work_dir.as_posix(),
            (self.work_dir / 'X_FILE.csv').as_posix(),
            (self.work_dir / 'VALIDATION_FILE.csv').as_posix(),
            (self.work_dir / 'XN_FILE.csv').as_posix(),
            '-1',
            str(len(self.datasets))
        ]

        # Dataset args
        # $args[datasetOffset]->Local arguments number
        # $args[datasetOffset+1]->"datasetName"
        # $args[datasetOffset+2]->"PATH/Y_FILE.csv"
        # $args[datasetOffset+3]->"datasetName/prefix"
        # $args[datasetOffset+4]->Solver ID
        # ... Solver args

        for ds in self.datasets:
            solver_args = ds.get_args()
            local_arg_count = 5 + len(solver_args)

            args.extend([
                str(local_arg_count),
                ds.name,
                (self.work_dir / ds.name / 'Y_FILE.csv').as_posix(),
                f'{ds.name}/data',
                str(ds.solver_id)
            ])
            args.extend(solver_args)

        self.command = args

    def _run_quasar(self):
        """
        Runs the Quasar executable.

        Executes the command generated by _generate_command() in a subprocess.
        Captures stdout and stderr.

        Raises
        ------
        RuntimeError
            If the subprocess returns a non-zero exit code.
        """
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        with subprocess.Popen(self.command,
                              cwd=self.work_dir,
                              startupinfo=startupinfo,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True) as proc:

            stdout, _ = proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f'Quasar execution failed:\n{stdout}')
            else:
                (self.work_dir / '_quasar_output.txt').write_text(stdout)

    def optimize(
        self,
        objectives: List['ObjectiveConfig'],
        bounds: Dict[str, Tuple[float, float]],
        constraints: Optional[List['ConstraintConfig']] = None,
        algorithm: str = 'auto',
        pop_size: int = 100,
        n_gen: int = 100,
        seed: Optional[int] = None,
        verbose: bool = True,
        **algorithm_kwargs
    ) -> 'OptimizationResult':
        """
        Perform multi-objective optimization using the trained surrogate model.

        This method uses pymoo's evolutionary algorithms to find Pareto-optimal
        solutions that trade off between multiple objectives. The surrogate model
        must be trained (via ``train()``) before calling this method.

        Parameters
        ----------
        objectives : List[ObjectiveConfig]
            List of objectives to optimize. Each objective specifies:

            - ``name``: Human-readable name for plots/reports
            - ``dataset_name``: Which trained dataset to use
            - ``aggregation``: How to convert curves to scalars (see below)
            - ``direction``: ``'minimize'`` or ``'maximize'``

            **Aggregation options for curve datasets:**

            - ``'scalar'`` or ``None``: Use value directly (for scalar datasets)
            - ``'max'``: Maximum value of curve (peak stress, max displacement)
            - ``'min'``: Minimum value (minimum safety factor)
            - ``'mean'``: Mean value (average power)
            - ``'integral'``: Area under curve (total energy, impulse)
            - ``'final'``: Last value (steady-state response)
            - ``'initial'``: First value
            - ``'peak'``: Maximum absolute value
            - ``'range'``: Max - Min (oscillation amplitude)
            - ``'std'``: Standard deviation (variability)
            - ``'rms'``: Root mean square (signal power)
            - Custom callable: ``f(y: np.ndarray) -> np.ndarray`` where
              input shape is ``(n_samples, n_curve_points)`` and output
              shape is ``(n_samples,)``

        bounds : Dict[str, Tuple[float, float]]
            Parameter bounds as ``{param_name: (lower, upper)}``.
            Keys must match column names from the training data ``X``.

        constraints : List[ConstraintConfig], optional
            List of constraints to enforce. Each constraint specifies:

            - ``name``: Human-readable name
            - ``dataset_name``: Which trained dataset to use
            - ``aggregation``: How to convert curves to scalars
            - ``constraint_type``: One of:

              - ``'<='`` or ``'le'``: value ≤ threshold (upper bound)
              - ``'>='`` or ``'ge'``: value ≥ threshold (lower bound)
              - ``'=='`` or ``'eq'``: |value - threshold| ≤ tolerance
              - ``'range'``: lower ≤ value ≤ upper

            - ``threshold``: Value for ``<=``, ``>=``, ``==`` types
            - ``lower``, ``upper``: Bounds for ``'range'`` type
            - ``tolerance``: For equality constraints (default 1e-6)

        algorithm : str, optional
            Optimization algorithm. Options:

            - ``'auto'``: Automatically select (NSGA2 for ≤3 objectives,
              NSGA3 for >3 objectives). **Recommended for most cases.**
            - ``'NSGA2'``: Non-dominated Sorting Genetic Algorithm II.
              Best for 2-3 objectives. Uses crowding distance for diversity.
            - ``'NSGA3'``: Reference-direction based NSGA-III.
              Best for 3-15 objectives. Uses structured reference points.
            - ``'MOEAD'``: Multi-Objective EA based on Decomposition.
              Decomposes into weighted scalar subproblems. Good for
              many objectives with uniform Pareto fronts.
            - ``'CTAEA'``: Constrained Two-Archive EA.
              Excellent for heavily constrained problems.
            - ``'AGEMOEA'``: Adaptive Geometry Estimation MOEA.
              Adapts to Pareto front geometry during search.
            - ``'AGEMOEA2'``: Improved AGE-MOEA with better convergence.
            - ``'SMSEMOA'``: S-Metric Selection EMOA.
              Uses hypervolume indicator. Slow but high-quality fronts.

            Default is ``'auto'``.

        pop_size : int, optional
            Population size (number of solutions per generation).
            Larger values improve diversity but increase computation.
            Typical range: 50-200. Default is 100.

        n_gen : int, optional
            Number of generations to evolve.
            More generations allow better convergence but take longer.
            Typical range: 50-500. Default is 100.

        seed : int, optional
            Random seed for reproducibility.

        verbose : bool, optional
            Whether to print progress during optimization.
            Default is True.

        **algorithm_kwargs
            Additional keyword arguments passed to the pymoo algorithm
            constructor. See pymoo documentation for algorithm-specific
            options.

        Returns
        -------
        OptimizationResult
            Object containing:

            - ``pareto_front``: DataFrame of objective values for
              Pareto-optimal solutions
            - ``pareto_set``: DataFrame of parameter values for
              Pareto-optimal solutions
            - ``summary()``: Summary statistics
            - ``plot_pareto_2d()``, ``plot_pareto_3d()``: Pareto visualizations
            - ``plot_parallel_coordinates()``: High-dimensional visualization
            - ``plot_convergence()``: Convergence history
            - ``dashboard()``: Interactive Jupyter exploration
            - ``save_html(filename)``: Export to HTML report

        Raises
        ------
        RuntimeError
            If the model has not been trained (``train()`` not called).
        ValueError
            If bounds contain unknown parameter names or algorithm is invalid.

        See Also
        --------
        ObjectiveConfig : Objective configuration class.
        ConstraintConfig : Constraint configuration class.
        OptimizationResult : Result container with visualization methods.

        Notes
        -----
        **Algorithm Selection Guide:**

        +------------+------------+------------------------------------------+
        | Algorithm  | Objectives | Best Use Case                            |
        +============+============+==========================================+
        | NSGA2      | 2-3        | General purpose, fast, robust            |
        +------------+------------+------------------------------------------+
        | NSGA3      | 3-15       | Many objectives, structured fronts       |
        +------------+------------+------------------------------------------+
        | MOEAD      | 3-10+      | Uniform Pareto fronts, decomposable      |
        +------------+------------+------------------------------------------+
        | CTAEA      | Any        | Heavy constraints, infeasible regions    |
        +------------+------------+------------------------------------------+
        | AGEMOEA    | 2-10       | Unknown front geometry                   |
        +------------+------------+------------------------------------------+
        | SMSEMOA    | 2-4        | Quality over speed, hypervolume          |
        +------------+------------+------------------------------------------+

        **Convergence Tips:**

        - Start with fewer generations (50) to verify setup
        - Increase ``pop_size`` if Pareto front has gaps
        - Increase ``n_gen`` if convergence plot shows improvement
        - Use ``seed`` for reproducible results

        Examples
        --------
        Basic two-objective optimization:

        >>> from quasarpy import Quasar, DatasetConfig, KrigingConfig
        >>> from quasarpy.optimization import ObjectiveConfig
        >>>
        >>> # Train surrogate
        >>> q = Quasar()
        >>> q.train(X_train, [ds_stress, ds_weight])
        >>>
        >>> # Define objectives
        >>> objectives = [
        ...     ObjectiveConfig(
        ...         name='Peak Stress',
        ...         dataset_name='stress',
        ...         aggregation='max',
        ...         direction='minimize'
        ...     ),
        ...     ObjectiveConfig(
        ...         name='Weight',
        ...         dataset_name='weight',
        ...         aggregation='scalar',
        ...         direction='minimize'
        ...     )
        ... ]
        >>>
        >>> # Define bounds
        >>> bounds = {
        ...     'thickness': (0.5, 10.0),
        ...     'density': (1.0, 5.0)
        ... }
        >>>
        >>> # Run optimization
        >>> result = q.optimize(objectives, bounds, pop_size=50, n_gen=100)
        >>> print(result.summary())
        >>> result.plot_pareto_2d().show()

        With constraints:

        >>> from quasarpy.optimization import ConstraintConfig
        >>>
        >>> constraints = [
        ...     ConstraintConfig(
        ...         name='Max Stress Limit',
        ...         dataset_name='stress',
        ...         aggregation='max',
        ...         constraint_type='<=',
        ...         threshold=500.0
        ...     ),
        ...     ConstraintConfig(
        ...         name='Min Thickness',
        ...         dataset_name='thickness',
        ...         aggregation='scalar',
        ...         constraint_type='>=',
        ...         threshold=2.0
        ...     )
        ... ]
        >>>
        >>> result = q.optimize(
        ...     objectives, bounds,
        ...     constraints=constraints,
        ...     algorithm='CTAEA'  # Good for constrained problems
        ... )

        Many-objective optimization (>3 objectives):

        >>> objectives = [
        ...     ObjectiveConfig('Stress', 'stress', 'max', 'minimize'),
        ...     ObjectiveConfig('Weight', 'weight', 'scalar', 'minimize'),
        ...     ObjectiveConfig('Cost', 'cost', 'scalar', 'minimize'),
        ...     ObjectiveConfig('Stiffness', 'stiffness', 'mean', 'maximize'),
        ...     ObjectiveConfig('Fatigue Life', 'fatigue', 'min', 'maximize'),
        ... ]
        >>>
        >>> # Auto-selects NSGA3 for >3 objectives
        >>> result = q.optimize(objectives, bounds, n_gen=200)
        >>> result.plot_parallel_coordinates().show()
        >>> result.save_html('optimization_report.html')
        """
        # Validate model is trained
        if not hasattr(self, 'datasets') or self.datasets is None:
            raise RuntimeError(
                'Model not trained. Call train() before optimize().'
            )

        # Validate bounds match training parameters
        train_params = set(self.x.columns)
        bound_params = set(bounds.keys())
        if not bound_params.issubset(train_params):
            unknown = bound_params - train_params
            raise ValueError(
                f'Unknown parameters in bounds: {unknown}. '
                f'Training parameters: {train_params}'
            )

        return run_optimization(
            quasar=self,
            objectives=objectives,
            bounds=bounds,
            constraints=constraints,
            algorithm=algorithm,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            verbose=verbose,
            **algorithm_kwargs
        )
