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
    AggregatedObjectiveConfig,
    ConstraintConfig,
    OptimizationResult,
    QuasarProblem,
    run_optimization,
    AGGREGATIONS,
    CROSS_AGGREGATIONS,
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
        result = c.evaluate(values=values)
        np.testing.assert_array_equal(result, [-5.0, 0.0, 5.0])

    def test_ge_constraint(self):
        """Test >= constraint: threshold - value <= 0."""
        c = ConstraintConfig(name='c', dataset_name='ds', constraint_type='>=', threshold=10.0)
        values = np.array([5.0, 10.0, 15.0])
        result = c.evaluate(values=values)
        np.testing.assert_array_equal(result, [5.0, 0.0, -5.0])

    def test_eq_constraint(self):
        """Test == constraint: |value - threshold| - tolerance <= 0."""
        c = ConstraintConfig(
            name='c', dataset_name='ds', constraint_type='==',
            threshold=10.0, tolerance=0.5
        )
        values = np.array([10.0, 10.4, 10.6])
        result = c.evaluate(values=values)
        np.testing.assert_array_almost_equal(result, [-0.5, -0.1, 0.1])

    def test_range_constraint(self):
        """Test range constraint: returns 2 constraint values."""
        c = ConstraintConfig(
            name='c', dataset_name='ds', constraint_type='range',
            lower=5.0, upper=15.0
        )
        values = np.array([3.0, 10.0, 20.0])
        result = c.evaluate(values=values)
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

    def test_func_based_constraint(self):
        """Test function-based constraint evaluation."""
        def my_func(predictions, X):
            # x1 + x2 <= 10  =>  x1 + x2 - 10 <= 0
            return X['x1'].values + X['x2'].values - 10.0

        c = ConstraintConfig(name='input_sum', func=my_func)

        predictions = {}  # Not used in this constraint
        X = pd.DataFrame({'x1': [3.0, 5.0, 8.0], 'x2': [2.0, 5.0, 4.0]})

        result = c.evaluate(predictions=predictions, X=X)
        # 3+2-10=-5, 5+5-10=0, 8+4-10=2
        np.testing.assert_array_equal(result, [-5.0, 0.0, 2.0])

    def test_func_based_constraint_with_predictions(self):
        """Test function-based constraint using predictions."""
        def my_func(predictions, X):
            # stress / weight <= 100  =>  stress / weight - 100 <= 0
            stress = predictions['stress'].values[:, 0]
            weight = X['weight'].values
            return stress / weight - 100.0

        c = ConstraintConfig(name='stress_to_weight', func=my_func)

        predictions = {'stress': pd.DataFrame({'0': [500.0, 900.0, 1200.0]})}
        X = pd.DataFrame({'weight': [5.0, 10.0, 10.0]})

        result = c.evaluate(predictions=predictions, X=X)
        # 500/5-100=0, 900/10-100=-10, 1200/10-100=20
        np.testing.assert_array_equal(result, [0.0, -10.0, 20.0])

    def test_func_skips_threshold_validation(self):
        """Test that func-based constraints don't require dataset_name/threshold."""
        # Should not raise even without dataset_name/threshold
        c = ConstraintConfig(
            name='custom',
            func=lambda preds, X: X['x'].values - 5.0
        )
        assert c.func is not None
        assert c.dataset_name is None

    def test_missing_dataset_name_raises_when_no_func(self):
        """Test that dataset_name is required when func is not provided."""
        with pytest.raises(ValueError, match='dataset_name required'):
            ConstraintConfig(name='c', constraint_type='<=', threshold=10.0)


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


# -----------------------------------------------------------------------------
# Unit Tests: AggregatedObjectiveConfig
# -----------------------------------------------------------------------------

class TestAggregatedObjectiveConfig:
    """Tests for AggregatedObjectiveConfig dataclass."""

    def test_valid_directions(self):
        """Test that valid directions are accepted."""
        obj_min = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1', 'ds2'], direction='minimize'
        )
        obj_max = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1', 'ds2'], direction='maximize'
        )
        assert obj_min.direction == 'minimize'
        assert obj_max.direction == 'maximize'

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match='direction must be'):
            AggregatedObjectiveConfig(
                name='test', dataset_names=['ds1'], direction='min'
            )

    def test_empty_dataset_names_raises(self):
        """Test that empty dataset_names raises ValueError."""
        with pytest.raises(ValueError, match='must contain at least one dataset'):
            AggregatedObjectiveConfig(name='test', dataset_names=[])

    def test_aggregate_curve_scalar(self):
        """Test curve aggregation for scalar datasets."""
        obj = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1'], aggregation='scalar'
        )
        y = np.array([[1.0], [2.0], [3.0]])
        result = obj.aggregate_curve(y)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_aggregate_curve_max(self):
        """Test max curve aggregation."""
        obj = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1'], aggregation='max'
        )
        y = np.array([[1, 5, 3], [2, 8, 4]])
        result = obj.aggregate_curve(y)
        np.testing.assert_array_equal(result, [5, 8])

    def test_aggregate_curve_callable(self):
        """Test custom callable curve aggregation."""
        obj = AggregatedObjectiveConfig(
            name='test',
            dataset_names=['ds1'],
            aggregation=lambda y: np.median(y, axis=1)
        )
        y = np.array([[1, 2, 3], [4, 5, 6]])
        result = obj.aggregate_curve(y)
        np.testing.assert_array_equal(result, [2, 5])

    def test_cross_aggregation_max(self):
        """Test max cross-dataset aggregation."""
        obj = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1', 'ds2'], cross_aggregation='max'
        )
        values = np.array([[1, 5], [3, 2], [4, 4]])  # 3 samples, 2 datasets
        result = obj.aggregate_cross_dataset(values)
        np.testing.assert_array_equal(result, [5, 3, 4])

    def test_cross_aggregation_min(self):
        """Test min cross-dataset aggregation."""
        obj = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1', 'ds2'], cross_aggregation='min'
        )
        values = np.array([[1, 5], [3, 2], [4, 4]])
        result = obj.aggregate_cross_dataset(values)
        np.testing.assert_array_equal(result, [1, 2, 4])

    def test_cross_aggregation_mean(self):
        """Test mean cross-dataset aggregation."""
        obj = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1', 'ds2'], cross_aggregation='mean'
        )
        values = np.array([[1.0, 5.0], [3.0, 7.0]])
        result = obj.aggregate_cross_dataset(values)
        np.testing.assert_array_equal(result, [3.0, 5.0])

    def test_cross_aggregation_worst_minimize(self):
        """Test 'worst' resolves to 'max' when minimizing."""
        obj = AggregatedObjectiveConfig(
            name='test',
            dataset_names=['ds1', 'ds2'],
            cross_aggregation='worst',
            direction='minimize'
        )
        values = np.array([[1, 5], [3, 2]])
        result = obj.aggregate_cross_dataset(values)
        # Minimizing: worst = highest value
        np.testing.assert_array_equal(result, [5, 3])

    def test_cross_aggregation_worst_maximize(self):
        """Test 'worst' resolves to 'min' when maximizing."""
        obj = AggregatedObjectiveConfig(
            name='test',
            dataset_names=['ds1', 'ds2'],
            cross_aggregation='worst',
            direction='maximize'
        )
        values = np.array([[1, 5], [3, 2]])
        result = obj.aggregate_cross_dataset(values)
        # Maximizing: worst = lowest value
        np.testing.assert_array_equal(result, [1, 2])

    def test_cross_aggregation_callable(self):
        """Test custom callable cross-dataset aggregation."""
        obj = AggregatedObjectiveConfig(
            name='test',
            dataset_names=['ds1', 'ds2', 'ds3'],
            cross_aggregation=lambda y: np.std(y, axis=1)
        )
        values = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        result = obj.aggregate_cross_dataset(values)
        np.testing.assert_array_almost_equal(result, [np.std([1, 2, 3]), 0.0])

    def test_invalid_cross_aggregation_raises(self):
        """Test that invalid cross_aggregation string raises ValueError."""
        obj = AggregatedObjectiveConfig(
            name='test', dataset_names=['ds1'], cross_aggregation='invalid'
        )
        with pytest.raises(ValueError, match='Unknown cross_aggregation'):
            obj.aggregate_cross_dataset(np.array([[1, 2]]))


# -----------------------------------------------------------------------------
# Unit Tests: QuasarProblem with AggregatedObjectiveConfig
# -----------------------------------------------------------------------------

class TestQuasarProblemAggregated:
    """Tests for QuasarProblem with aggregated objectives."""

    @pytest.fixture
    def mock_quasar_multi_dataset(self):
        """Create a mock Quasar with 4 load case datasets."""
        mock_quasar = Mock()

        def mock_predict(x_df):
            n = len(x_df)
            # Simulate 4 load cases with different stress levels
            return {
                'stress_up': pd.DataFrame(
                    np.ones((n, 1)) * x_df['x1'].values.reshape(-1, 1) * 10
                ),
                'stress_down': pd.DataFrame(
                    np.ones((n, 1)) * x_df['x1'].values.reshape(-1, 1) * 8
                ),
                'stress_left': pd.DataFrame(
                    np.ones((n, 1)) * x_df['x1'].values.reshape(-1, 1) * 12
                ),
                'stress_right': pd.DataFrame(
                    np.ones((n, 1)) * x_df['x1'].values.reshape(-1, 1) * 9
                ),
                'cost': pd.DataFrame(
                    np.ones((n, 1)) * x_df['x2'].values.reshape(-1, 1) * 100
                ),
            }

        mock_quasar.predict = mock_predict
        return mock_quasar

    def test_aggregated_objective_evaluation(self, mock_quasar_multi_dataset):
        """Test that aggregated objectives are evaluated correctly."""
        objectives = [
            AggregatedObjectiveConfig(
                name='Worst Stress',
                dataset_names=['stress_up', 'stress_down', 'stress_left', 'stress_right'],
                aggregation='scalar',
                cross_aggregation='max',
                direction='minimize'
            ),
            ObjectiveConfig(
                name='Cost',
                dataset_name='cost',
                aggregation='scalar',
                direction='minimize'
            )
        ]
        bounds = {'x1': (0, 1), 'x2': (0, 1)}

        problem = QuasarProblem(mock_quasar_multi_dataset, objectives, bounds)

        # Evaluate
        X = np.array([[0.5, 0.3], [1.0, 0.5]])
        out = {}
        problem._evaluate(X, out)

        # x1=0.5: stress_up=5, down=4, left=6, right=4.5 -> max=6
        # x1=1.0: stress_up=10, down=8, left=12, right=9 -> max=12
        expected_stress = [6.0, 12.0]
        # x2=0.3: cost=30, x2=0.5: cost=50
        expected_cost = [30.0, 50.0]

        np.testing.assert_array_almost_equal(out['F'][:, 0], expected_stress)
        np.testing.assert_array_almost_equal(out['F'][:, 1], expected_cost)

    def test_worst_case_minimize(self, mock_quasar_multi_dataset):
        """Test worst-case with minimize direction."""
        objectives = [
            AggregatedObjectiveConfig(
                name='Worst Stress',
                dataset_names=['stress_up', 'stress_down', 'stress_left', 'stress_right'],
                aggregation='scalar',
                cross_aggregation='worst',  # Should resolve to 'max'
                direction='minimize'
            )
        ]
        bounds = {'x1': (0, 1), 'x2': (0, 1)}
        problem = QuasarProblem(mock_quasar_multi_dataset, objectives, bounds)

        X = np.array([[1.0, 0.0]])
        out = {}
        problem._evaluate(X, out)

        # stress_left = 12 is highest
        np.testing.assert_array_almost_equal(out['F'][:, 0], [12.0])

    def test_worst_case_maximize(self):
        """Test worst-case with maximize direction."""
        mock_quasar = Mock()

        def mock_predict(x_df):
            n = len(x_df)
            return {
                'quality_a': pd.DataFrame(np.ones((n, 1)) * 0.9),
                'quality_b': pd.DataFrame(np.ones((n, 1)) * 0.7),
                'quality_c': pd.DataFrame(np.ones((n, 1)) * 0.8),
            }

        mock_quasar.predict = mock_predict

        objectives = [
            AggregatedObjectiveConfig(
                name='Worst Quality',
                dataset_names=['quality_a', 'quality_b', 'quality_c'],
                aggregation='scalar',
                cross_aggregation='worst',  # Should resolve to 'min'
                direction='maximize'
            )
        ]
        bounds = {'x1': (0, 1)}
        problem = QuasarProblem(mock_quasar, objectives, bounds)

        X = np.array([[0.5]])
        out = {}
        problem._evaluate(X, out)

        # quality_b = 0.7 is lowest = worst when maximizing
        # Since maximize, objective is negated
        np.testing.assert_array_almost_equal(out['F'][:, 0], [-0.7])

    def test_mixed_objectives(self, mock_quasar_multi_dataset):
        """Test mixing standard and aggregated objectives."""
        objectives = [
            AggregatedObjectiveConfig(
                name='Worst Stress',
                dataset_names=['stress_up', 'stress_down'],
                aggregation='scalar',
                cross_aggregation='max',
                direction='minimize'
            ),
            ObjectiveConfig(
                name='Cost',
                dataset_name='cost',
                aggregation='scalar',
                direction='minimize'
            )
        ]
        bounds = {'x1': (0, 1), 'x2': (0, 1)}
        problem = QuasarProblem(mock_quasar_multi_dataset, objectives, bounds)

        assert problem.n_obj == 2

        X = np.array([[0.5, 0.4]])
        out = {}
        problem._evaluate(X, out)

        # stress_up=5, stress_down=4 -> max=5
        # cost=40
        np.testing.assert_array_almost_equal(out['F'][0], [5.0, 40.0])

    def test_curve_then_cross_aggregation(self):
        """Test curve aggregation followed by cross-dataset aggregation."""
        mock_quasar = Mock()

        def mock_predict(x_df):
            n = len(x_df)
            # Each dataset returns curves (n_samples, 3 points)
            return {
                'curve_a': pd.DataFrame(np.array([[1, 5, 3]] * n)),  # max=5
                'curve_b': pd.DataFrame(np.array([[2, 8, 1]] * n)),  # max=8
            }

        mock_quasar.predict = mock_predict

        objectives = [
            AggregatedObjectiveConfig(
                name='Worst Peak',
                dataset_names=['curve_a', 'curve_b'],
                aggregation='max',  # Curve -> scalar (take max of each curve)
                cross_aggregation='max',  # Cross-dataset (take max across datasets)
                direction='minimize'
            )
        ]
        bounds = {'x1': (0, 1)}
        problem = QuasarProblem(mock_quasar, objectives, bounds)

        X = np.array([[0.5]])
        out = {}
        problem._evaluate(X, out)

        # curve_a max=5, curve_b max=8 -> cross max=8
        np.testing.assert_array_almost_equal(out['F'][:, 0], [8.0])
