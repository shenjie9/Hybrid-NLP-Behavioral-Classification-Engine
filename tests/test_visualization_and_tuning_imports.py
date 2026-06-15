"""Smoke tests for visualization and tuning modules."""

from art_sense_ml.hyperparameter_tuning import prepare_data, tune_l2_lambda
from art_sense_ml.visualization import plot_numeric_feature_distributions


def test_visualization_imports() -> None:
    assert callable(plot_numeric_feature_distributions)


def test_hyperparameter_tuning_imports() -> None:
    assert callable(prepare_data)
    assert callable(tune_l2_lambda)
