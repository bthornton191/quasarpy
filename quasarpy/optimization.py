"""
Multi-objective optimization module for quasarpy.

This module provides classes and utilities for performing multi-objective
optimization using trained Quasar surrogate models with pymoo algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from ipywidgets import widgets
from IPython.display import display

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

if TYPE_CHECKING:
    from .quasar import Quasar


# Built-in aggregation functions for curves
AGGREGATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'max': lambda y: np.max(y, axis=1),
    'min': lambda y: np.min(y, axis=1),
    'mean': lambda y: np.mean(y, axis=1),
    'integral': lambda y: np.trapz(y, axis=1),
    'final': lambda y: y[:, -1],
    'initial': lambda y: y[:, 0],
    'peak': lambda y: np.max(np.abs(y), axis=1),
    'range': lambda y: np.max(y, axis=1) - np.min(y, axis=1),
    'std': lambda y: np.std(y, axis=1),
    'rms': lambda y: np.sqrt(np.mean(y ** 2, axis=1)),
}


@dataclass
class ObjectiveConfig:
    """
    Configuration for an optimization objective.

    An objective defines what quantity to optimize (minimize or maximize)
    by specifying a dataset from the Quasar model and how to aggregate
    its curve output into a scalar value.

    Parameters
    ----------
    name : str
        Human-readable name for the objective (used in plots and reports).
    dataset_name : str
        Name of the dataset from the trained Quasar model to use for this
        objective. Must match a dataset name passed to ``Quasar.train()``.
    aggregation : str or Callable, optional
        Method to convert curve predictions to scalar values. For scalar
        datasets (single output per sample), use ``'scalar'`` or ``None``.

        Built-in options:

        - ``'scalar'`` or ``None``: Use the value directly (for scalar datasets)
        - ``'max'``: Maximum value of the curve. Use for peak stress, max displacement.
        - ``'min'``: Minimum value of the curve. Use for minimum safety factor.
        - ``'mean'``: Mean value across the curve. Use for average power.
        - ``'integral'``: Trapezoidal integral (area under curve). Use for total energy.
        - ``'final'``: Last value in the curve. Use for steady-state response.
        - ``'initial'``: First value in the curve.
        - ``'peak'``: Maximum absolute value. Use when sign doesn't matter.
        - ``'range'``: Difference between max and min. Use for oscillation amplitude.
        - ``'std'``: Standard deviation. Use for variability measures.
        - ``'rms'``: Root mean square. Use for signal power.

        Custom callable: A function with signature ``f(y: np.ndarray) -> np.ndarray``
        where input shape is ``(n_samples, n_curve_points)`` and output shape
        is ``(n_samples,)``.

        Default is ``'scalar'``.

    direction : str, optional
        Optimization direction: ``'minimize'`` or ``'maximize'``.
        Default is ``'minimize'``.

    Examples
    --------
    Minimize peak stress from a curve dataset:

    >>> obj_stress = ObjectiveConfig(
    ...     name='Peak Stress',
    ...     dataset_name='stress_curve',
    ...     aggregation='max',
    ...     direction='minimize'
    ... )

    Maximize final displacement (steady-state):

    >>> obj_disp = ObjectiveConfig(
    ...     name='Final Displacement',
    ...     dataset_name='displacement',
    ...     aggregation='final',
    ...     direction='maximize'
    ... )

    Minimize a scalar output directly:

    >>> obj_weight = ObjectiveConfig(
    ...     name='Weight',
    ...     dataset_name='total_weight',
    ...     aggregation='scalar',
    ...     direction='minimize'
    ... )

    Use a custom aggregation (95th percentile):

    >>> obj_custom = ObjectiveConfig(
    ...     name='95th Percentile Stress',
    ...     dataset_name='stress_curve',
    ...     aggregation=lambda y: np.percentile(y, 95, axis=1),
    ...     direction='minimize'
    ... )
    """
    name: str
    dataset_name: str
    aggregation: Union[str, Callable[[np.ndarray], np.ndarray], None] = 'scalar'
    direction: str = 'minimize'

    def __post_init__(self):
        if self.direction not in ('minimize', 'maximize'):
            raise ValueError(f'direction must be "minimize" or "maximize", got "{self.direction}"')

    def aggregate(self, y: np.ndarray) -> np.ndarray:
        """
        Apply the aggregation function to curve predictions.

        Parameters
        ----------
        y : np.ndarray
            Predicted values with shape ``(n_samples, n_curve_points)``.
            For scalar datasets, shape is ``(n_samples, 1)``.

        Returns
        -------
        np.ndarray
            Aggregated scalar values with shape ``(n_samples,)``.
        """
        # Handle scalar datasets
        if y.shape[1] == 1:
            return y[:, 0]

        # Handle aggregation specification
        if self.aggregation is None or self.aggregation == 'scalar':
            # Assume scalar, take first column
            return y[:, 0]
        elif callable(self.aggregation):
            return self.aggregation(y)
        elif isinstance(self.aggregation, str):
            if self.aggregation not in AGGREGATIONS:
                valid = list(AGGREGATIONS.keys()) + ['scalar']
                raise ValueError(
                    f'Unknown aggregation "{self.aggregation}". '
                    f'Valid options: {valid}'
                )
            return AGGREGATIONS[self.aggregation](y)
        else:
            raise TypeError(
                f'aggregation must be str, callable, or None, got {type(self.aggregation)}'
            )


@dataclass
class ConstraintConfig:
    """
    Configuration for an optimization constraint.

    Constraints restrict the feasible design space. pymoo uses the convention
    that a constraint is satisfied when ``g(x) <= 0``. This class provides
    two modes of operation:

    1. **Threshold-based** (default): Specify ``dataset_name``, ``aggregation``,
       ``constraint_type``, and ``threshold`` to constrain model outputs.

    2. **Function-based**: Provide a custom ``func`` callable for full flexibility,
       including constraints on inputs, outputs, or combinations thereof.

    Parameters
    ----------
    name : str
        Human-readable name for the constraint (used in reports).
    dataset_name : str, optional
        Name of the dataset from the trained Quasar model to use.
        Required for threshold-based constraints, ignored if ``func`` is provided.
    aggregation : str or Callable, optional
        Method to convert curve predictions to scalar values.
        Same options as ``ObjectiveConfig.aggregation``.
        Ignored if ``func`` is provided.
        Default is ``'scalar'``.
    constraint_type : str, optional
        Type of constraint:

        - ``'<='`` or ``'le'``: value <= threshold (upper bound)
        - ``'>='`` or ``'ge'``: value >= threshold (lower bound)
        - ``'=='`` or ``'eq'``: |value - threshold| <= tolerance (equality)
        - ``'range'``: lower <= value <= upper

        Ignored if ``func`` is provided.
        Default is ``'<='``.

    threshold : float, optional
        The constraint threshold for ``'<='``, ``'>='``, and ``'=='`` types.
        Required for these constraint types (unless ``func`` is provided).
    lower : float, optional
        Lower bound for ``'range'`` constraint type.
    upper : float, optional
        Upper bound for ``'range'`` constraint type.
    tolerance : float, optional
        Tolerance for equality constraints (``'=='``).
        The constraint is satisfied when ``|value - threshold| <= tolerance``.
        Default is ``1e-6``.
    func : Callable, optional
        Custom constraint function with signature:

        ``func(predictions: Dict[str, pd.DataFrame], X: pd.DataFrame) -> np.ndarray``

        Where:

        - ``predictions``: Dictionary of model predictions (from ``Quasar.predict()``).
          Keys are dataset names, values are DataFrames with shape
          ``(n_samples, n_curve_points)``.
        - ``X``: Input parameter DataFrame with shape ``(n_samples, n_params)``.
          Column names match the parameter names from ``bounds``.

        The function must return an array with shape ``(n_samples,)`` or
        ``(n_samples, n_constraints)`` where values ``<= 0`` indicate feasibility.

        When ``func`` is provided, all other parameters except ``name`` are ignored.

    Notes
    -----
    pymoo constraint convention:
        - ``g(x) <= 0`` is feasible
        - ``g(x) > 0`` is infeasible

    The constraint is automatically converted to this form internally.

    Examples
    --------
    Upper bound constraint (stress <= 500 MPa):

    >>> c_stress = ConstraintConfig(
    ...     name='Max Stress Limit',
    ...     dataset_name='stress_curve',
    ...     aggregation='max',
    ...     constraint_type='<=',
    ...     threshold=500.0
    ... )

    Lower bound constraint (safety factor >= 1.5):

    >>> c_sf = ConstraintConfig(
    ...     name='Min Safety Factor',
    ...     dataset_name='safety_factor',
    ...     aggregation='min',
    ...     constraint_type='>=',
    ...     threshold=1.5
    ... )

    Range constraint (10 <= thickness <= 50):

    >>> c_thick = ConstraintConfig(
    ...     name='Thickness Range',
    ...     dataset_name='thickness',
    ...     aggregation='scalar',
    ...     constraint_type='range',
    ...     lower=10.0,
    ...     upper=50.0
    ... )

    Equality constraint (volume ≈ 100 ± 0.1):

    >>> c_vol = ConstraintConfig(
    ...     name='Target Volume',
    ...     dataset_name='volume',
    ...     aggregation='scalar',
    ...     constraint_type='==',
    ...     threshold=100.0,
    ...     tolerance=0.1
    ... )

    Custom function constraint on inputs (x1 + x2 <= 10):

    >>> c_input = ConstraintConfig(
    ...     name='Parameter Sum Limit',
    ...     func=lambda preds, X: X['x1'].values + X['x2'].values - 10.0
    ... )

    Custom function constraint combining inputs and outputs:

    >>> def efficiency_constraint(predictions, X):
    ...     # power / weight >= 50, where power is predicted and weight is input
    ...     power = predictions['power'].values[:, 0]  # scalar dataset
    ...     weight = X['weight'].values
    ...     # Rearrange: power / weight >= 50  =>  50 - power / weight <= 0
    ...     return 50.0 - power / weight
    >>>
    >>> c_efficiency = ConstraintConfig(
    ...     name='Min Power-to-Weight Ratio',
    ...     func=efficiency_constraint
    ... )
    """
    name: str
    dataset_name: Optional[str] = None
    aggregation: Union[str, Callable[[np.ndarray], np.ndarray], None] = 'scalar'
    constraint_type: str = '<='
    threshold: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    tolerance: float = 1e-6
    func: Optional[Callable[[Dict[str, pd.DataFrame], pd.DataFrame], np.ndarray]] = None

    def __post_init__(self):
        # If func is provided, skip validation of threshold-based parameters
        if self.func is not None:
            return

        # For threshold-based constraints, require dataset_name
        if self.dataset_name is None:
            raise ValueError('dataset_name required when func is not provided')

        valid_types = ('<=', 'le', '>=', 'ge', '==', 'eq', 'range')
        if self.constraint_type not in valid_types:
            raise ValueError(f'constraint_type must be one of {valid_types}')

        # Validate required parameters
        if self.constraint_type in ('<=', 'le', '>=', 'ge', '==', 'eq'):
            if self.threshold is None:
                raise ValueError(f'threshold required for constraint_type="{self.constraint_type}"')
        elif self.constraint_type == 'range':
            if self.lower is None or self.upper is None:
                raise ValueError('lower and upper required for constraint_type="range"')

    def aggregate(self, y: np.ndarray) -> np.ndarray:
        """Apply aggregation function (same logic as ObjectiveConfig)."""
        if y.shape[1] == 1:
            return y[:, 0]

        if self.aggregation is None or self.aggregation == 'scalar':
            return y[:, 0]
        elif callable(self.aggregation):
            return self.aggregation(y)
        elif isinstance(self.aggregation, str):
            if self.aggregation not in AGGREGATIONS:
                raise ValueError(f'Unknown aggregation "{self.aggregation}"')
            return AGGREGATIONS[self.aggregation](y)
        else:
            raise TypeError('aggregation must be str, callable, or None')

    def evaluate(
        self,
        values: Optional[np.ndarray] = None,
        predictions: Optional[Dict[str, pd.DataFrame]] = None,
        X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Evaluate constraint violation (g <= 0 is feasible).

        Parameters
        ----------
        values : np.ndarray, optional
            Aggregated values with shape ``(n_samples,)``.
            Used for threshold-based constraints (when ``func`` is None).
        predictions : Dict[str, pd.DataFrame], optional
            Model predictions from ``Quasar.predict()``.
            Used for function-based constraints (when ``func`` is provided).
        X : pd.DataFrame, optional
            Input parameters with shape ``(n_samples, n_params)``.
            Used for function-based constraints (when ``func`` is provided).

        Returns
        -------
        np.ndarray
            Constraint values where <= 0 means feasible.
            Shape is ``(n_samples,)`` or ``(n_samples, n_constraints)``.
            For ``'range'`` type, returns two columns (lower and upper bounds).
        """
        # Function-based constraint
        if self.func is not None:
            if predictions is None or X is None:
                raise ValueError('predictions and X required for func-based constraint')
            return self.func(predictions, X)

        # Threshold-based constraint
        if values is None:
            raise ValueError('values required for threshold-based constraint')

        if self.constraint_type in ('<=', 'le'):
            # value <= threshold  =>  value - threshold <= 0
            return values - self.threshold
        elif self.constraint_type in ('>=', 'ge'):
            # value >= threshold  =>  threshold - value <= 0
            return self.threshold - values
        elif self.constraint_type in ('==', 'eq'):
            # |value - threshold| <= tolerance  =>  |value - threshold| - tolerance <= 0
            return np.abs(values - self.threshold) - self.tolerance
        elif self.constraint_type == 'range':
            # lower <= value <= upper
            # Two constraints: lower - value <= 0 AND value - upper <= 0
            g_lower = self.lower - values
            g_upper = values - self.upper
            return np.column_stack([g_lower, g_upper])

    @property
    def n_constraints(self) -> int:
        """
        Number of constraint functions.

        For threshold-based constraints, range type produces 2 constraints.
        For function-based constraints, this returns 1 (the function itself
        may return multiple columns, which is handled during evaluation).
        """
        if self.func is not None:
            return 1  # Will be updated during first evaluation if needed
        return 2 if self.constraint_type == 'range' else 1


class QuasarProblem(Problem):
    """
    pymoo Problem wrapper for Quasar surrogate models.

    This class wraps a trained Quasar model as a pymoo optimization problem,
    enabling multi-objective optimization with NSGA-II, NSGA-III, MOEA/D,
    and other algorithms.

    Parameters
    ----------
    quasar : Quasar
        A trained Quasar instance (must have called ``train()`` first).
    objectives : List[ObjectiveConfig]
        List of objective configurations defining what to optimize.
    bounds : Dict[str, Tuple[float, float]]
        Parameter bounds as ``{param_name: (lower, upper)}``.
        Keys must match column names from the training data.
    constraints : List[ConstraintConfig], optional
        List of constraint configurations. Default is no constraints.

    Attributes
    ----------
    quasar : Quasar
        The trained Quasar model.
    objectives : List[ObjectiveConfig]
        Objective configurations.
    constraints : List[ConstraintConfig]
        Constraint configurations.
    param_names : List[str]
        Ordered list of parameter names.
    n_evals : int
        Counter for number of function evaluations.
    """

    def __init__(
        self,
        quasar: Quasar,
        objectives: List[ObjectiveConfig],
        bounds: Dict[str, Tuple[float, float]],
        constraints: Optional[List[ConstraintConfig]] = None
    ):
        self.quasar = quasar
        self.objectives = objectives
        self.constraints = constraints or []
        self.param_names = list(bounds.keys())
        self.n_evals = 0

        # Extract bounds
        xl = np.array([bounds[p][0] for p in self.param_names])
        xu = np.array([bounds[p][1] for p in self.param_names])

        # Count total constraints (range type counts as 2)
        n_ieq_constr = sum(c.n_constraints for c in self.constraints)

        super().__init__(
            n_var=len(self.param_names),
            n_obj=len(objectives),
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluate objectives and constraints for a population.

        Parameters
        ----------
        X : np.ndarray
            Decision variables with shape ``(pop_size, n_var)``.
        out : dict
            Output dictionary to store ``'F'`` (objectives) and ``'G'`` (constraints).
        """
        # Convert to DataFrame for Quasar
        x_df = pd.DataFrame(X, columns=self.param_names)

        # Get predictions
        predictions = self.quasar.predict(x_df)
        self.n_evals += len(X)

        # Evaluate objectives
        F = np.zeros((len(X), self.n_obj))
        for i, obj in enumerate(self.objectives):
            y = predictions[obj.dataset_name].values
            f_val = obj.aggregate(y)

            # Handle maximize by negating
            if obj.direction == 'maximize':
                f_val = -f_val

            F[:, i] = f_val

        out['F'] = F

        # Evaluate constraints
        if self.constraints:
            G_list = []
            for constr in self.constraints:
                # Function-based constraint
                if constr.func is not None:
                    g = constr.evaluate(predictions=predictions, X=x_df)
                else:
                    # Threshold-based constraint
                    y = predictions[constr.dataset_name].values
                    values = constr.aggregate(y)
                    g = constr.evaluate(values=values)

                # Ensure 2D
                if g.ndim == 1:
                    g = g.reshape(-1, 1)
                G_list.append(g)

            out['G'] = np.hstack(G_list)


class ConvergenceCallback(Callback):
    """Callback to track convergence history during optimization."""

    def __init__(self):
        super().__init__()
        self.history = []

    def notify(self, algorithm):
        """Record population statistics at each generation."""
        pop = algorithm.pop
        F = pop.get('F')

        self.history.append({
            'n_gen': algorithm.n_gen,
            'n_evals': algorithm.evaluator.n_eval,
            'F_min': F.min(axis=0).tolist(),
            'F_max': F.max(axis=0).tolist(),
            'F_mean': F.mean(axis=0).tolist(),
            'n_nds': len(algorithm.opt) if hasattr(algorithm, 'opt') else len(pop)
        })


class OptimizationResult:
    """
    Container for multi-objective optimization results with visualization.

    This class stores the Pareto front, Pareto set, and optimization history,
    providing methods for analysis and visualization.

    Parameters
    ----------
    pymoo_result : pymoo.core.result.Result
        The result object from pymoo's ``minimize()`` function.
    objectives : List[ObjectiveConfig]
        List of objective configurations used in optimization.
    constraints : List[ConstraintConfig]
        List of constraint configurations used in optimization.
    param_names : List[str]
        Ordered list of parameter names.
    history : List[dict]
        Convergence history from callback.

    Attributes
    ----------
    pareto_front : pd.DataFrame
        Objective values for Pareto-optimal solutions.
        Shape: ``(n_solutions, n_objectives)``.
    pareto_set : pd.DataFrame
        Parameter values for Pareto-optimal solutions.
        Shape: ``(n_solutions, n_params)``.
    n_solutions : int
        Number of Pareto-optimal solutions found.
    n_evals : int
        Total number of function evaluations.
    """

    def __init__(
        self,
        pymoo_result,
        objectives: List[ObjectiveConfig],
        constraints: List[ConstraintConfig],
        param_names: List[str],
        history: List[dict]
    ):
        self._result = pymoo_result
        self.objectives = objectives
        self.constraints = constraints
        self.param_names = param_names
        self.history = history

        # Extract Pareto front and set, handling edge cases
        F = pymoo_result.F.copy()
        X = pymoo_result.X.copy()

        # Ensure 2D arrays (pymoo returns 1D for single solutions)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Undo maximize negation
        obj_names = []
        for i, obj in enumerate(objectives):
            if obj.direction == 'maximize':
                F[:, i] = -F[:, i]
            obj_names.append(obj.name)

        self.pareto_front = pd.DataFrame(F, columns=obj_names)
        self.pareto_set = pd.DataFrame(X, columns=param_names)

    @property
    def n_solutions(self) -> int:
        """Number of Pareto-optimal solutions."""
        return len(self.pareto_front)

    @property
    def n_evals(self) -> int:
        """Total number of function evaluations."""
        return self._result.algorithm.evaluator.n_eval

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame of the optimization results.

        Returns
        -------
        pd.DataFrame
            Summary statistics including min, max, mean for each objective.
        """
        stats = []
        for col in self.pareto_front.columns:
            stats.append({
                'Objective': col,
                'Min': self.pareto_front[col].min(),
                'Max': self.pareto_front[col].max(),
                'Mean': self.pareto_front[col].mean(),
                'Std': self.pareto_front[col].std()
            })
        return pd.DataFrame(stats).set_index('Objective')

    def get_solution(self, index: int) -> Dict:
        """
        Get a specific Pareto-optimal solution.

        Parameters
        ----------
        index : int
            Index of the solution (0 to n_solutions-1).

        Returns
        -------
        dict
            Dictionary with 'parameters' and 'objectives' DataFrames.
        """
        return {
            'parameters': self.pareto_set.iloc[[index]],
            'objectives': self.pareto_front.iloc[[index]]
        }

    def plot_pareto_2d(
        self,
        obj_x: Optional[str] = None,
        obj_y: Optional[str] = None
    ) -> go.Figure:
        """
        Create a 2D Pareto front scatter plot.

        Parameters
        ----------
        obj_x : str, optional
            Name of objective for x-axis. Defaults to first objective.
        obj_y : str, optional
            Name of objective for y-axis. Defaults to second objective.

        Returns
        -------
        go.Figure
            Plotly figure with the Pareto front.
        """
        obj_names = list(self.pareto_front.columns)
        obj_x = obj_x or obj_names[0]
        obj_y = obj_y or obj_names[1] if len(obj_names) > 1 else obj_names[0]

        fig = go.Figure()

        # Create hover text with all parameters
        hover_text = []
        for idx in range(len(self.pareto_set)):
            params = '<br>'.join(
                f'{p}: {self.pareto_set.iloc[idx][p]:.4g}'
                for p in self.param_names
            )
            hover_text.append(f'Solution {idx}<br>{params}')

        fig.add_trace(go.Scatter(
            x=self.pareto_front[obj_x],
            y=self.pareto_front[obj_y],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.7),
            text=hover_text,
            hovertemplate='%{text}<br>' + f'{obj_x}: ' + '%{x:.4g}<br>' + f'{obj_y}: ' + '%{y:.4g}<extra></extra>'
        ))

        fig.update_layout(
            title='Pareto Front',
            xaxis_title=obj_x,
            yaxis_title=obj_y,
            width=700,
            height=500
        )

        return fig

    def plot_pareto_3d(
        self,
        obj_x: Optional[str] = None,
        obj_y: Optional[str] = None,
        obj_z: Optional[str] = None
    ) -> go.Figure:
        """
        Create a 3D Pareto front scatter plot.

        Parameters
        ----------
        obj_x, obj_y, obj_z : str, optional
            Names of objectives for each axis. Defaults to first three objectives.

        Returns
        -------
        go.Figure
            Plotly 3D figure with the Pareto front.
        """
        obj_names = list(self.pareto_front.columns)
        if len(obj_names) < 3:
            raise ValueError('Need at least 3 objectives for 3D plot')

        obj_x = obj_x or obj_names[0]
        obj_y = obj_y or obj_names[1]
        obj_z = obj_z or obj_names[2]

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=self.pareto_front[obj_x],
            y=self.pareto_front[obj_y],
            z=self.pareto_front[obj_z],
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.7)
        ))

        fig.update_layout(
            title='Pareto Front (3D)',
            scene=dict(
                xaxis_title=obj_x,
                yaxis_title=obj_y,
                zaxis_title=obj_z
            ),
            width=800,
            height=600
        )

        return fig

    def plot_parallel_coordinates(self) -> go.Figure:
        """
        Create a parallel coordinates plot of Pareto-optimal solutions.

        Shows both parameters and objectives for each solution, useful for
        understanding trade-offs in high-dimensional Pareto fronts.

        Returns
        -------
        go.Figure
            Plotly parallel coordinates figure.
        """
        # Combine parameters and objectives
        combined = pd.concat([self.pareto_set, self.pareto_front], axis=1)

        # Normalize all columns to [0, 1] for visualization
        dims = []
        for col in combined.columns:
            vals = combined[col]
            dims.append(dict(
                label=col,
                values=vals,
                range=[vals.min(), vals.max()]
            ))

        # Color by first objective
        first_obj = self.pareto_front.columns[0]

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=self.pareto_front[first_obj],
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(title=first_obj)
            ),
            dimensions=dims
        ))

        fig.update_layout(
            title='Parallel Coordinates: Pareto Set',
            width=900,
            height=500
        )

        return fig

    def plot_convergence(self, metric: str = 'n_nds') -> go.Figure:
        """
        Plot convergence history.

        Parameters
        ----------
        metric : str, optional
            Metric to plot: ``'n_nds'`` (number of non-dominated solutions),
            ``'F_min'``, ``'F_max'``, or ``'F_mean'``. Default is ``'n_nds'``.

        Returns
        -------
        go.Figure
            Plotly figure showing convergence.
        """
        if not self.history:
            raise ValueError('No convergence history available')

        df = pd.DataFrame(self.history)
        fig = go.Figure()

        if metric == 'n_nds':
            fig.add_trace(go.Scatter(
                x=df['n_gen'],
                y=df['n_nds'],
                mode='lines+markers',
                name='Non-dominated Solutions'
            ))
            fig.update_layout(yaxis_title='Number of Non-dominated Solutions')
        elif metric in ('F_min', 'F_max', 'F_mean'):
            # Plot for each objective
            obj_names = [obj.name for obj in self.objectives]
            colors = sample_colorscale('Turbo', np.linspace(0, 1, len(obj_names)))

            for i, obj_name in enumerate(obj_names):
                values = [h[metric][i] for h in self.history]
                fig.add_trace(go.Scatter(
                    x=df['n_gen'],
                    y=values,
                    mode='lines+markers',
                    name=obj_name,
                    line=dict(color=colors[i])
                ))
            fig.update_layout(yaxis_title=metric.replace('_', ' ').title())

        fig.update_layout(
            title=f'Convergence: {metric}',
            xaxis_title='Generation',
            width=700,
            height=400
        )

        return fig

    def dashboard(self):
        """
        Display an interactive Jupyter dashboard for exploring results.

        Features:
        - Pareto front visualization (2D scatter or parallel coordinates)
        - Solution selection and inspection
        - Convergence plots
        - Parameter value display
        """
        n_obj = len(self.objectives)
        obj_names = [obj.name for obj in self.objectives]

        # Widgets
        view_dropdown = widgets.Dropdown(
            options=['Pareto 2D', 'Pareto 3D', 'Parallel Coordinates', 'Convergence'],
            value='Pareto 2D' if n_obj >= 2 else 'Parallel Coordinates',
            description='View:'
        )

        obj_x_dropdown = widgets.Dropdown(
            options=obj_names,
            value=obj_names[0],
            description='X-Axis:'
        )

        obj_y_dropdown = widgets.Dropdown(
            options=obj_names,
            value=obj_names[1] if n_obj > 1 else obj_names[0],
            description='Y-Axis:'
        )

        obj_z_dropdown = widgets.Dropdown(
            options=obj_names,
            value=obj_names[2] if n_obj > 2 else obj_names[0],
            description='Z-Axis:'
        )

        solution_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.n_solutions - 1,
            description='Solution:',
            continuous_update=False
        )

        # Output areas
        plot_output = widgets.Output()
        info_output = widgets.Output()

        def update_plot(*args):
            view = view_dropdown.value
            with plot_output:
                plot_output.clear_output(wait=True)
                try:
                    if view == 'Pareto 2D':
                        fig = self.plot_pareto_2d(obj_x_dropdown.value, obj_y_dropdown.value)
                    elif view == 'Pareto 3D':
                        fig = self.plot_pareto_3d(
                            obj_x_dropdown.value,
                            obj_y_dropdown.value,
                            obj_z_dropdown.value
                        )
                    elif view == 'Parallel Coordinates':
                        fig = self.plot_parallel_coordinates()
                    elif view == 'Convergence':
                        fig = self.plot_convergence()
                    display(fig)
                except Exception as e:
                    print(f'Error: {e}')

        def update_info(*args):
            idx = solution_slider.value
            with info_output:
                info_output.clear_output(wait=True)
                print(f'Solution {idx}')
                print('\nParameters:')
                for p in self.param_names:
                    print(f'  {p}: {self.pareto_set.iloc[idx][p]:.6g}')
                print('\nObjectives:')
                for obj in self.objectives:
                    print(f'  {obj.name}: {self.pareto_front.iloc[idx][obj.name]:.6g}')

        def on_view_change(change):
            if change['new'] == 'Pareto 3D':
                axis_controls.children = [obj_x_dropdown, obj_y_dropdown, obj_z_dropdown]
            elif change['new'] in ('Pareto 2D',):
                axis_controls.children = [obj_x_dropdown, obj_y_dropdown]
            else:
                axis_controls.children = []
            update_plot()

        # Connect callbacks
        view_dropdown.observe(on_view_change, names='value')
        obj_x_dropdown.observe(update_plot, names='value')
        obj_y_dropdown.observe(update_plot, names='value')
        obj_z_dropdown.observe(update_plot, names='value')
        solution_slider.observe(update_info, names='value')

        # Layout
        axis_controls = widgets.HBox([obj_x_dropdown, obj_y_dropdown])
        controls = widgets.VBox([view_dropdown, axis_controls, solution_slider])
        main_area = widgets.HBox([plot_output, info_output])

        # Initialize
        update_plot()
        update_info()

        display(widgets.VBox([controls, main_area]))

    def save_html(self, filename: str):
        """
        Export results to an interactive HTML file.

        Parameters
        ----------
        filename : str
            Path to output HTML file.
        """
        import plotly.io as pio

        n_obj = len(self.objectives)

        # Create main Pareto plot
        if n_obj >= 3:
            fig_pareto = self.plot_pareto_3d()
        elif n_obj >= 2:
            fig_pareto = self.plot_pareto_2d()
        else:
            fig_pareto = self.plot_parallel_coordinates()

        fig_parallel = self.plot_parallel_coordinates()
        fig_convergence = self.plot_convergence() if self.history else None

        # Generate HTML
        div_pareto = pio.to_html(fig_pareto, full_html=False, include_plotlyjs='cdn', div_id='pareto_plot')
        div_parallel = pio.to_html(fig_parallel, full_html=False, include_plotlyjs=False, div_id='parallel_plot')
        div_convergence = pio.to_html(fig_convergence, full_html=False, include_plotlyjs=False, div_id='convergence_plot') if fig_convergence else ''

        # Summary table
        summary_html = self.summary().to_html(classes='table', float_format='%.4g')

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Optimization Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .table {{ border-collapse: collapse; margin: 10px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                .table th {{ background-color: #f4f4f4; }}
                .plot-container {{ margin: 20px 0; }}
                .stats {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Multi-Objective Optimization Report</h1>

            <div class="section stats">
                <h2>Summary Statistics</h2>
                <p><strong>Pareto Solutions:</strong> {self.n_solutions}</p>
                <p><strong>Function Evaluations:</strong> {self.n_evals}</p>
                {summary_html}
            </div>

            <div class="section">
                <h2>Pareto Front</h2>
                <div class="plot-container">{div_pareto}</div>
            </div>

            <div class="section">
                <h2>Parallel Coordinates</h2>
                <div class="plot-container">{div_parallel}</div>
            </div>

            {'<div class="section"><h2>Convergence</h2><div class="plot-container">' + div_convergence + '</div></div>' if div_convergence else ''}
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)


def _get_default_ref_dirs(n_obj: int, pop_size: int, **kw) -> np.ndarray:
    """
    Generate default reference directions for many-objective algorithms.

    For 5 or fewer objectives, uses Das-Dennis with adaptive partitions.
    For 6+ objectives, uses energy-based method with exact point count
    to avoid combinatorial explosion.

    Parameters
    ----------
    n_obj : int
        Number of objectives.
    pop_size : int
        Population size (used as default n_points for energy method).
    **kw : dict
        Optional overrides:

        - ``ref_dirs``: Pre-computed reference directions (returned as-is)
        - ``ref_dirs_method``: ``'das-dennis'``, ``'energy'``, or ``'auto'``
        - ``n_ref_points``: Number of reference points for energy method
        - ``n_partitions``: Number of partitions for Das-Dennis method

    Returns
    -------
    np.ndarray
        Reference directions with shape ``(n_points, n_obj)``.
    """
    # Allow user to pass pre-computed ref_dirs
    if 'ref_dirs' in kw:
        return kw.pop('ref_dirs')

    method = kw.pop('ref_dirs_method', 'auto')
    n_points = kw.pop('n_ref_points', None)
    n_partitions = kw.pop('n_partitions', None)

    if method == 'auto':
        # Use Das-Dennis for few objectives, energy for many
        method = 'das-dennis' if n_obj <= 5 else 'energy'

    if method == 'das-dennis':
        # Adaptive partitions to keep ref_dirs count reasonable
        if n_partitions is None:
            n_partitions = max(2, 8 - n_obj)  # e.g., 6 for 2 obj, 3 for 5 obj
        return get_reference_directions('das-dennis', n_obj, n_partitions=n_partitions)
    elif method == 'energy':
        # Energy method gives exact control over number of points
        if n_points is None:
            n_points = pop_size
        return get_reference_directions('energy', n_obj, n_points=n_points)
    else:
        # Pass through to pymoo for other methods (layer-energy, etc.)
        return get_reference_directions(method, n_obj, n_points=n_points or pop_size)


def _make_nsga3(n_obj: int, pop_size: int, **kw):
    """Create NSGA3 with smart reference direction defaults."""
    ref_dirs = _get_default_ref_dirs(n_obj, pop_size, **kw)
    return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **kw)


def _make_moead(n_obj: int, pop_size: int, **kw):
    """Create MOEAD with smart reference direction defaults."""
    ref_dirs = _get_default_ref_dirs(n_obj, pop_size, **kw)
    n_neighbors = kw.pop('n_neighbors', 15)
    prob_neighbor_mating = kw.pop('prob_neighbor_mating', 0.7)
    return MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
        **kw
    )


def _make_ctaea(n_obj: int, pop_size: int, **kw):
    """Create CTAEA with smart reference direction defaults."""
    ref_dirs = _get_default_ref_dirs(n_obj, pop_size, **kw)
    return CTAEA(ref_dirs=ref_dirs, **kw)


# Algorithm factory
ALGORITHMS = {
    'NSGA2': lambda n_obj, pop_size, **kw: NSGA2(pop_size=pop_size, **kw),
    'NSGA3': _make_nsga3,
    'MOEAD': _make_moead,
    'CTAEA': _make_ctaea,
    'AGEMOEA': lambda n_obj, pop_size, **kw: AGEMOEA(pop_size=pop_size, **kw),
    'AGEMOEA2': lambda n_obj, pop_size, **kw: AGEMOEA2(pop_size=pop_size, **kw),
    'SMSEMOA': lambda n_obj, pop_size, **kw: SMSEMOA(pop_size=pop_size, **kw),
}


def run_optimization(
    quasar: Quasar,
    objectives: List[ObjectiveConfig],
    bounds: Dict[str, Tuple[float, float]],
    constraints: Optional[List[ConstraintConfig]] = None,
    algorithm: str = 'auto',
    pop_size: int = 100,
    n_gen: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True,
    **algorithm_kwargs
) -> OptimizationResult:
    """
    Run multi-objective optimization on a trained Quasar model.

    This is the main entry point for optimization. It creates the problem,
    selects an appropriate algorithm, and runs the optimization.

    Parameters
    ----------
    quasar : Quasar
        A trained Quasar instance.
    objectives : List[ObjectiveConfig]
        List of objectives to optimize.
    bounds : Dict[str, Tuple[float, float]]
        Parameter bounds.
    constraints : List[ConstraintConfig], optional
        List of constraints.
    algorithm : str, optional
        Algorithm to use. Options:

        - ``'auto'``: Automatically select based on number of objectives
          (NSGA2 for ≤3, NSGA3 for >3)
        - ``'NSGA2'``: Classic NSGA-II, best for 2-3 objectives
        - ``'NSGA3'``: Reference-direction based, best for 3-15 objectives
        - ``'MOEAD'``: Decomposition-based, good for many objectives
        - ``'CTAEA'``: Constrained Two-Archive EA, good with constraints
        - ``'AGEMOEA'``: Adaptive Geometry Estimation MOEA
        - ``'AGEMOEA2'``: Improved AGE-MOEA
        - ``'SMSEMOA'``: S-Metric Selection EMOA

        Default is ``'auto'``.

    pop_size : int, optional
        Population size. Default is 100.
    n_gen : int, optional
        Number of generations. Default is 100.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print progress. Default is True.
    **algorithm_kwargs
        Additional keyword arguments passed to the algorithm constructor.
        For reference-direction algorithms (NSGA3, MOEAD, CTAEA), you can pass:

        - ``ref_dirs``: Pre-computed reference directions array
        - ``ref_dirs_method``: ``'das-dennis'``, ``'energy'``, or ``'auto'`` (default)
        - ``n_ref_points``: Number of points for ``'energy'`` method
        - ``n_partitions``: Number of partitions for ``'das-dennis'`` method

        By default, ``'auto'`` uses Das-Dennis for ≤5 objectives and energy-based
        for 6+ objectives to avoid combinatorial explosion.

    Returns
    -------
    OptimizationResult
        Object containing Pareto front, Pareto set, and visualization methods.

    Examples
    --------
    Basic usage with auto-selected algorithm:

    >>> result = run_optimization(quasar, objectives, bounds)

    Many-objective optimization (6+ objectives) - uses energy method by default:

    >>> result = run_optimization(
    ...     quasar, objectives, bounds,
    ...     pop_size=200,  # Also sets n_ref_points=200 by default
    ...     n_gen=150
    ... )

    Override reference direction generation:

    >>> result = run_optimization(
    ...     quasar, objectives, bounds,
    ...     algorithm='NSGA3',
    ...     n_ref_points=150,  # Exact number of reference directions
    ...     ref_dirs_method='energy'
    ... )

    Use pre-computed reference directions:

    >>> from pymoo.util.ref_dirs import get_reference_directions
    >>> ref_dirs = get_reference_directions('energy', n_obj=12, n_points=100)
    >>> result = run_optimization(
    ...     quasar, objectives, bounds,
    ...     algorithm='NSGA3',
    ...     ref_dirs=ref_dirs
    ... )
    """
    # Create problem
    problem = QuasarProblem(quasar, objectives, bounds, constraints)

    # Select algorithm
    n_obj = len(objectives)
    if algorithm == 'auto':
        algorithm = 'NSGA2' if n_obj <= 3 else 'NSGA3'

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Options: {list(ALGORITHMS.keys())}")

    algo = ALGORITHMS[algorithm](n_obj, pop_size, **algorithm_kwargs)

    # Setup callback
    callback = ConvergenceCallback()

    # Run optimization
    result = minimize(
        problem,
        algo,
        ('n_gen', n_gen),
        seed=seed,
        verbose=verbose,
        callback=callback
    )

    return OptimizationResult(
        pymoo_result=result,
        objectives=objectives,
        constraints=constraints or [],
        param_names=problem.param_names,
        history=callback.history
    )
