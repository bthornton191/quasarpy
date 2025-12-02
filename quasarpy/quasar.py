from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


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
            self.work_dir = Path(work_dir)
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
