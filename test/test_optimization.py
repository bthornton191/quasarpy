"""
Tests for the optimization module.

Unit tests use mocked Quasar predictions to test optimization logic
without requiring the Quasar engine.

Integration tests require ODYSSEE_CAE_INSTALLDIR or ODYSSEE_SOLVER_INSTALLDIR.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from quasarpy.optimization import (
    ObjectiveConfig,
    ConstraintConfig,
    OptimizationResult,
    QuasarProblem,
    run_optimization,
    AGGREGATIONS
)


# -----------------------------------------------------------------------------
# Unit Tests: ObjectiveConfig
# -----------------------------------------------------------------------------

class TestObjectiveConfig:
    """Tests for ObjectiveConfig dataclass."""

    def test_valid_directions(self):
        """Test that valid directions are accepted."""
        obj_min = ObjectiveConfig(name='test', dataset_name='ds', direction='minimize')
        obj_max = ObjectiveConfig(name='test', dataset_name='ds', direction='maximize')
        assert obj_min.direction == 'minimize'
        assert obj_max.direction == 'maximize'

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match='direction must be'):
            ObjectiveConfig(name='test', dataset_name='ds', direction='min')

    def test_aggregate_scalar(self):
        """Test aggregation for scalar datasets (single column)."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='scalar')
        y = np.array([[1.0], [2.0], [3.0]])  # 3 samples, 1 point
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_aggregate_max(self):
        """Test max aggregation for curve datasets."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='max')
        y = np.array([[1, 5, 3], [2, 8, 4], [0, 1, 9]])  # 3 samples, 3 points
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [5, 8, 9])

    def test_aggregate_min(self):
        """Test min aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='min')
        y = np.array([[1, 5, 3], [2, 8, 4], [0, 1, 9]])
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [1, 2, 0])

    def test_aggregate_mean(self):
        """Test mean aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='mean')
        y = np.array([[1.0, 5.0, 3.0], [2.0, 8.0, 5.0]])
        result = obj.aggregate(y)
        np.testing.assert_array_almost_equal(result, [3.0, 5.0])

    def test_aggregate_integral(self):
        """Test integral (trapz) aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='integral')
        y = np.array([[1.0, 1.0, 1.0]])  # Constant function = area 2
        result = obj.aggregate(y)
        np.testing.assert_array_almost_equal(result, [2.0])

    def test_aggregate_final(self):
        """Test final value aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='final')
        y = np.array([[1, 2, 3], [4, 5, 6]])
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [3, 6])

    def test_aggregate_initial(self):
        """Test initial value aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='initial')
        y = np.array([[1, 2, 3], [4, 5, 6]])
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [1, 4])

    def test_aggregate_peak(self):
        """Test peak (max abs) aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='peak')
        y = np.array([[-10, 5, 3], [2, -8, 4]])
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [10, 8])

    def test_aggregate_range(self):
        """Test range aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='range')
        y = np.array([[1, 5, 3], [2, 10, 4]])
        result = obj.aggregate(y)
        np.testing.assert_array_equal(result, [4, 8])  # 5-1, 10-2

    def test_aggregate_std(self):
        """Test std aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='std')
        y = np.array([[1.0, 2.0, 3.0]])
        result = obj.aggregate(y)
        assert result[0] == pytest.approx(np.std([1, 2, 3]))

    def test_aggregate_rms(self):
        """Test RMS aggregation."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='rms')
        y = np.array([[3.0, 4.0]])  # sqrt((9+16)/2) = sqrt(12.5)
        result = obj.aggregate(y)
        assert result[0] == pytest.approx(np.sqrt(12.5))

    def test_aggregate_callable(self):
        """Test custom callable aggregation."""
        obj = ObjectiveConfig(
            name='test',
            dataset_name='ds',
            aggregation=lambda y: np.percentile(y, 90, axis=1)
        )
        y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        result = obj.aggregate(y)
        assert result[0] == pytest.approx(9.1)

    def test_aggregate_invalid_raises(self):
        """Test that invalid aggregation string raises ValueError."""
        obj = ObjectiveConfig(name='test', dataset_name='ds', aggregation='invalid')
        with pytest.raises(ValueError, match='Unknown aggregation'):
            obj.aggregate(np.array([[1, 2, 3]]))


# -----------------------------------------------------------------------------
# Unit Tests: ConstraintConfig
# -----------------------------------------------------------------------------

class TestConstraintConfig:
    """Tests for ConstraintConfig dataclass."""

    def test_le_constraint(self):
        """Test <= constraint: value - threshold <= 0."""
        c = ConstraintConfig(name='c', dataset_name='ds', constraint_type='<=', threshold=10.0)
        values = np.array([5.0, 10.0, 15.0])
        result = c.evaluate(values)
        np.testing.assert_array_equal(result, [-5.0, 0.0, 5.0])

    def test_ge_constraint(self):
        """Test >= constraint: threshold - value <= 0."""
        c = ConstraintConfig(name='c', dataset_name='ds', constraint_type='>=', threshold=10.0)
        values = np.array([5.0, 10.0, 15.0])
        result = c.evaluate(values)
        np.testing.assert_array_equal(result, [5.0, 0.0, -5.0])

    def test_eq_constraint(self):
        """Test == constraint: |value - threshold| - tolerance <= 0."""
        c = ConstraintConfig(
            name='c', dataset_name='ds', constraint_type='==',
            threshold=10.0, tolerance=0.5
        )
        values = np.array([10.0, 10.4, 10.6])
        result = c.evaluate(values)
        np.testing.assert_array_almost_equal(result, [-0.5, -0.1, 0.1])

    def test_range_constraint(self):
        """Test range constraint: returns 2 constraint values."""
        c = ConstraintConfig(
            name='c', dataset_name='ds', constraint_type='range',
            lower=5.0, upper=15.0
        )
        values = np.array([3.0, 10.0, 20.0])
        result = c.evaluate(values)
        # g_lower = lower - value, g_upper = value - upper
        expected = np.array([[2.0, -12.0], [-5.0, -5.0], [-15.0, 5.0]])
        np.testing.assert_array_equal(result, expected)

    def test_n_constraints_single(self):
        """Test n_constraints property for single constraint types."""
        c = ConstraintConfig(name='c', dataset_name='ds', constraint_type='<=', threshold=10.0)
        assert c.n_constraints == 1

    def test_n_constraints_range(self):
        """Test n_constraints property for range constraint."""
        c = ConstraintConfig(name='c', dataset_name='ds', constraint_type='range', lower=0, upper=10)
        assert c.n_constraints == 2

    def test_missing_threshold_raises(self):
        """Test that missing threshold raises for <= constraint."""
        with pytest.raises(ValueError, match='threshold required'):
            ConstraintConfig(name='c', dataset_name='ds', constraint_type='<=')

    def test_missing_bounds_raises(self):
        """Test that missing lower/upper raises for range constraint."""
        with pytest.raises(ValueError, match='lower and upper required'):
            ConstraintConfig(name='c', dataset_name='ds', constraint_type='range', lower=0)

    def test_invalid_type_raises(self):
        """Test that invalid constraint_type raises ValueError."""
        with pytest.raises(ValueError, match='constraint_type must be one of'):
            ConstraintConfig(name='c', dataset_name='ds', constraint_type='<')


# -----------------------------------------------------------------------------
# Unit Tests: QuasarProblem (with mocked Quasar)
# -----------------------------------------------------------------------------

class TestQuasarProblem:
    """Tests for QuasarProblem class with mocked Quasar."""

    @pytest.fixture
    def mock_quasar(self):
        """Create a mock Quasar that returns predictable values."""
        q = Mock()
        # predict() returns dict of DataFrames
        # For testing: dataset1 = 2*x1 + x2 (scalar), dataset2 = curves

        def mock_predict(x_df):
            n = len(x_df)
            return {
                'scalar_ds': pd.DataFrame(
                    2 * x_df['x1'].values + x_df['x2'].values
                ).T.T,  # Shape (n, 1)
                'curve_ds': pd.DataFrame(
                    np.tile(x_df['x1'].values.reshape(-1, 1), (1, 5)) * np.arange(1, 6)
                )  # Shape (n, 5)
            }
        q.predict = mock_predict
        return q

    def test_problem_dimensions(self, mock_quasar):
        """Test problem dimensions are set correctly."""
        objectives = [
            ObjectiveConfig('obj1', 'scalar_ds', 'scalar', 'minimize'),
            ObjectiveConfig('obj2', 'curve_ds', 'max', 'minimize'),
        ]
        bounds = {'x1': (0, 10), 'x2': (0, 5)}
        constraints = [
            ConstraintConfig('c1', 'scalar_ds', 'scalar', '<=', threshold=20)
        ]

        problem = QuasarProblem(mock_quasar, objectives, bounds, constraints)

        assert problem.n_var == 2
        assert problem.n_obj == 2
        assert problem.n_ieq_constr == 1
        np.testing.assert_array_equal(problem.xl, [0, 0])
        np.testing.assert_array_equal(problem.xu, [10, 5])

    def test_evaluate_objectives(self, mock_quasar):
        """Test objective evaluation."""
        objectives = [
            ObjectiveConfig('scalar', 'scalar_ds', 'scalar', 'minimize'),
            ObjectiveConfig('peak', 'curve_ds', 'max', 'minimize'),
        ]
        bounds = {'x1': (0, 10), 'x2': (0, 5)}

        problem = QuasarProblem(mock_quasar, objectives, bounds)

        X = np.array([[1.0, 2.0], [3.0, 1.0]])  # 2 samples
        out = {}
        problem._evaluate(X, out)

        # scalar_ds = 2*x1 + x2 = [4, 7]
        # curve_ds max for x1=1: max([1,2,3,4,5]) = 5
        # curve_ds max for x1=3: max([3,6,9,12,15]) = 15
        expected_F = np.array([[4.0, 5.0], [7.0, 15.0]])
        np.testing.assert_array_almost_equal(out['F'], expected_F)

    def test_evaluate_maximize_negation(self, mock_quasar):
        """Test that maximize objectives are negated."""
        objectives = [
            ObjectiveConfig('scalar', 'scalar_ds', 'scalar', 'maximize'),
        ]
        bounds = {'x1': (0, 10), 'x2': (0, 5)}

        problem = QuasarProblem(mock_quasar, objectives, bounds)

        X = np.array([[1.0, 2.0]])
        out = {}
        problem._evaluate(X, out)

        # scalar_ds = 2*1 + 2 = 4, negated for maximize = -4
        assert out['F'][0, 0] == -4.0

    def test_evaluate_constraints(self, mock_quasar):
        """Test constraint evaluation."""
        objectives = [ObjectiveConfig('obj', 'scalar_ds', 'scalar', 'minimize')]
        bounds = {'x1': (0, 10), 'x2': (0, 5)}
        constraints = [
            ConstraintConfig('c1', 'scalar_ds', 'scalar', '<=', threshold=5.0)
        ]

        problem = QuasarProblem(mock_quasar, objectives, bounds, constraints)

        X = np.array([[1.0, 2.0], [3.0, 1.0]])  # scalar_ds = [4, 7]
        out = {}
        problem._evaluate(X, out)

        # G = value - threshold = [4-5, 7-5] = [-1, 2]
        expected_G = np.array([[-1.0], [2.0]])
        np.testing.assert_array_almost_equal(out['G'], expected_G)


# -----------------------------------------------------------------------------
# Unit Tests: OptimizationResult
# -----------------------------------------------------------------------------

class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample OptimizationResult for testing."""
        # Mock pymoo result
        pymoo_result = Mock()
        pymoo_result.F = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 1.5]])
        pymoo_result.X = np.array([[0.5, 1.0], [1.0, 0.5], [0.7, 0.8]])
        pymoo_result.algorithm = Mock()
        pymoo_result.algorithm.evaluator = Mock()
        pymoo_result.algorithm.evaluator.n_eval = 1000

        objectives = [
            ObjectiveConfig('Obj1', 'ds1', 'max', 'minimize'),
            ObjectiveConfig('Obj2', 'ds2', 'min', 'maximize'),  # Maximized
        ]

        return OptimizationResult(
            pymoo_result=pymoo_result,
            objectives=objectives,
            constraints=[],
            param_names=['x1', 'x2'],
            history=[]
        )

    def test_pareto_front_shape(self, sample_result):
        """Test Pareto front DataFrame shape and columns."""
        assert sample_result.pareto_front.shape == (3, 2)
        assert list(sample_result.pareto_front.columns) == ['Obj1', 'Obj2']

    def test_pareto_set_shape(self, sample_result):
        """Test Pareto set DataFrame shape and columns."""
        assert sample_result.pareto_set.shape == (3, 2)
        assert list(sample_result.pareto_set.columns) == ['x1', 'x2']

    def test_maximize_unnegated(self, sample_result):
        """Test that maximized objectives are un-negated in Pareto front."""
        # Original F[:, 1] = [2, 1, 1.5], should be negated to [-2, -1, -1.5]
        np.testing.assert_array_almost_equal(
            sample_result.pareto_front['Obj2'].values,
            [-2.0, -1.0, -1.5]
        )

    def test_n_solutions(self, sample_result):
        """Test n_solutions property."""
        assert sample_result.n_solutions == 3

    def test_n_evals(self, sample_result):
        """Test n_evals property."""
        assert sample_result.n_evals == 1000

    def test_summary(self, sample_result):
        """Test summary method."""
        summary = sample_result.summary()
        assert 'Min' in summary.columns
        assert 'Max' in summary.columns
        assert 'Mean' in summary.columns
        assert 'Std' in summary.columns
        assert len(summary) == 2  # 2 objectives

    def test_get_solution(self, sample_result):
        """Test get_solution method."""
        sol = sample_result.get_solution(1)
        assert 'parameters' in sol
        assert 'objectives' in sol
        np.testing.assert_array_almost_equal(
            sol['parameters'].values[0],
            [1.0, 0.5]
        )


# -----------------------------------------------------------------------------
# Integration Tests (require Quasar installation)
# -----------------------------------------------------------------------------

@pytest.fixture
def quasar_available():
    """Check if Quasar is available."""
    import os
    return (
        'ODYSSEE_CAE_INSTALLDIR' in os.environ or
        'ODYSSEE_SOLVER_INSTALLDIR' in os.environ
    )


@pytest.mark.skipif(
    'ODYSSEE_CAE_INSTALLDIR' not in __import__('os').environ and
    'ODYSSEE_SOLVER_INSTALLDIR' not in __import__('os').environ,
    reason='Quasar installation not available'
)
class TestOptimizationIntegration:
    """Integration tests requiring Quasar installation."""

    @pytest.fixture
    def trained_quasar(self):
        """Create and train a Quasar instance with synthetic data."""
        from quasarpy import Quasar, DatasetConfig, KrigingConfig

        # Simple functions: y1 = 2*x1 + x2, y2 = x1 - 0.5*x2
        X = pd.DataFrame({
            'x1': [0.0, 1.0, 2.0, 0.5, 1.5],
            'x2': [0.0, 1.0, 0.0, 1.0, 0.5]
        })
        Y1 = pd.DataFrame((2 * X['x1'] + X['x2']).values.reshape(-1, 1))
        Y2 = pd.DataFrame((X['x1'] - 0.5 * X['x2']).values.reshape(-1, 1))

        q = Quasar(keep_work_dir=False)
        q.train(X, [
            DatasetConfig('output1', Y1, kriging_config=KrigingConfig(basis_function=2)),
            DatasetConfig('output2', Y2, kriging_config=KrigingConfig(basis_function=2))
        ])
        return q

    def test_basic_optimization(self, trained_quasar):
        """Test basic two-objective optimization."""
        objectives = [
            ObjectiveConfig('f1', 'output1', 'scalar', 'minimize'),
            ObjectiveConfig('f2', 'output2', 'scalar', 'maximize'),
        ]
        bounds = {'x1': (0, 2), 'x2': (0, 1)}

        result = trained_quasar.optimize(
            objectives, bounds,
            algorithm='NSGA2',
            pop_size=20,
            n_gen=10,
            seed=42,
            verbose=False
        )

        assert result.n_solutions > 0
        assert result.pareto_front.shape[1] == 2
        assert result.pareto_set.shape[1] == 2

    def test_constrained_optimization(self, trained_quasar):
        """Test optimization with constraints."""
        objectives = [
            ObjectiveConfig('f1', 'output1', 'scalar', 'minimize'),
        ]
        bounds = {'x1': (0, 2), 'x2': (0, 1)}
        constraints = [
            ConstraintConfig('c1', 'output2', 'scalar', '>=', threshold=0.5)
        ]

        result = trained_quasar.optimize(
            objectives, bounds,
            constraints=constraints,
            algorithm='NSGA2',
            pop_size=20,
            n_gen=10,
            seed=42,
            verbose=False
        )

        assert result.n_solutions > 0

    def test_html_export(self, trained_quasar, tmp_path):
        """Test HTML export."""
        objectives = [
            ObjectiveConfig('f1', 'output1', 'scalar', 'minimize'),
            ObjectiveConfig('f2', 'output2', 'scalar', 'maximize'),
        ]
        bounds = {'x1': (0, 2), 'x2': (0, 1)}

        result = trained_quasar.optimize(
            objectives, bounds,
            pop_size=20, n_gen=5, seed=42, verbose=False
        )

        html_path = tmp_path / 'optimization_report.html'
        result.save_html(str(html_path))

        assert html_path.exists()
        content = html_path.read_text()
        assert 'Pareto' in content
        assert 'plotly' in content.lower()
