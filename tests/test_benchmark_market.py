"""
Test suite for market-grade benchmark results.
Validates that all 5 synthetic benchmarks meet minimum quality thresholds.
"""
import json
from pathlib import Path
import pytest


def load_benchmark_results():
    """Load the merged 5-dataset benchmark results."""
    results_file = Path(__file__).parent.parent / "results" / "benchmark_market_real.json"
    assert results_file.exists(), f"Benchmark results file not found: {results_file}"

    with open(results_file) as f:
        results = json.load(f)

    assert isinstance(results, list), "Results should be a list of benchmarks"
    assert len(results) == 5, f"Expected 5 benchmarks, got {len(results)}"

    return results


class TestMarketBenchmarks:
    """Validate market benchmark quality."""

    @pytest.fixture(scope="module")
    def benchmarks(self):
        return load_benchmark_results()

    def test_all_datasets_present(self, benchmarks):
        """Ensure all 5 required datasets are present."""
        names = {b['name'] for b in benchmarks}
        expected = {
            'IoT Sensor Anomaly Detection',
            'Healthcare Risk Stratification',
            'Financial Fraud Detection',
            'Telecom Churn Prediction',
            'E-commerce Purchase Propensity',
        }
        assert names == expected, f"Missing or extra datasets: {names ^ expected}"

    def test_accuracy_above_minimum(self, benchmarks):
        """All benchmarks must achieve >90% accuracy (or >93% for healthcare)."""
        thresholds = {
            'Healthcare Risk Stratification': 0.93,
        }

        for benchmark in benchmarks:
            name = benchmark['name']
            min_acc = thresholds.get(name, 0.90)
            actual_acc = benchmark['accuracy']

            assert actual_acc >= min_acc, (
                f"{name}: accuracy {actual_acc:.2%} below threshold {min_acc:.2%}"
            )

    def test_flops_reduction_meaningful(self, benchmarks):
        """All benchmarks must achieve >20% FLOPs reduction (market-grade compression)."""
        for benchmark in benchmarks:
            flops_red = benchmark['flops_reduction']

            assert flops_red >= 20.0, (
                f"{benchmark['name']}: FLOPs reduction {flops_red:.1f}% is below 20%"
            )

    def test_runtime_reasonable(self, benchmarks):
        """All benchmarks must complete in <20 minutes (market-grade throughput)."""
        max_seconds = 20 * 60

        for benchmark in benchmarks:
            time_sec = benchmark['time_seconds']

            assert time_sec <= max_seconds, (
                f"{benchmark['name']}: took {time_sec/60:.1f} min, exceeds 20 min limit"
            )

    def test_metadata_complete(self, benchmarks):
        """Ensure all required metadata fields are present."""
        required_fields = {
            'name', 'domain', 'samples', 'features', 'classes',
            'accuracy', 'flops_reduction', 'time_seconds',
            'n_generations', 'n_pop', 'type'
        }

        for benchmark in benchmarks:
            missing = required_fields - set(benchmark.keys())
            assert not missing, (
                f"{benchmark.get('name', 'unknown')}: missing fields {missing}"
            )

    def test_type_is_synthetic_market(self, benchmarks):
        """All market benchmarks should be marked as synthetic_market."""
        for benchmark in benchmarks:
            assert benchmark['type'] == 'synthetic_market', (
                f"{benchmark['name']}: type is '{benchmark['type']}', expected 'synthetic_market'"
            )

    def test_parameters_consistent(self, benchmarks):
        """All market benchmarks should use consistent NAS parameters."""
        for benchmark in benchmarks:
            assert benchmark['n_generations'] == 15, (
                f"{benchmark['name']}: n_generations={benchmark['n_generations']}, expected 15"
            )
            assert benchmark['n_pop'] == 12, (
                f"{benchmark['name']}: n_pop={benchmark['n_pop']}, expected 12"
            )

    def test_samples_and_features_in_range(self, benchmarks):
        """Validate that sample and feature counts are market-realistic."""
        expected_ranges = {
            'IoT Sensor Anomaly Detection': {'samples': 50000, 'features': 42},
            'Healthcare Risk Stratification': {'samples': 25000, 'features': 89},
            'Financial Fraud Detection': {'samples': 100000, 'features': 56},
            'Telecom Churn Prediction': {'samples': 35000, 'features': 67},
            'E-commerce Purchase Propensity': {'samples': 80000, 'features': 48},
        }

        for benchmark in benchmarks:
            name = benchmark['name']
            expected = expected_ranges.get(name, {})

            if expected:
                assert benchmark['samples'] == expected['samples'], (
                    f"{name}: samples mismatch"
                )
                assert benchmark['features'] == expected['features'], (
                    f"{name}: features mismatch"
                )

    def test_accuracy_vs_flops_tradeoff_reasonable(self, benchmarks):
        """Verify that compression didn't come at extreme accuracy cost."""
        for benchmark in benchmarks:
            # Market-grade benchmarks should maintain >90% accuracy even with >20% compression
            acc = benchmark['accuracy']
            flops_red = benchmark['flops_reduction']

            # Rough heuristic: deeper compression allowed with higher baseline accuracy
            if acc >= 0.99:
                # Near-perfect: can afford aggressive compression
                assert flops_red >= 20.0
            elif acc >= 0.94:
                # High: expect decent compression
                assert flops_red >= 20.0
            elif acc >= 0.90:
                # Good: still should compress meaningfully
                assert flops_red >= 20.0
            else:
                pytest.fail(f"{benchmark['name']}: accuracy {acc:.2%} too low")


class TestMarketBenchmarkIntegration:
    """Validate that results integrate properly into the app."""

    def test_results_file_valid_json(self):
        """Ensure the JSON is valid and parseable."""
        results_file = Path(__file__).parent.parent / "results" / "benchmark_market_real.json"

        try:
            with open(results_file) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"benchmark_market_real.json is invalid JSON: {e}")

    def test_no_duplicate_dataset_names(self):
        """Ensure no duplicate dataset names across market benchmarks."""
        benchmarks = load_benchmark_results()
        names = [b['name'] for b in benchmarks]

        duplicates = [name for name in names if names.count(name) > 1]
        assert not duplicates, f"Duplicate dataset names: {set(duplicates)}"

    def test_class_distribution_reasonable(self):
        """Verify multiclass setup is reasonable."""
        benchmarks = load_benchmark_results()

        for benchmark in benchmarks:
            n_classes = benchmark['classes']
            n_samples = benchmark['samples']

            # Sanity check: at least a few samples per class
            samples_per_class = n_samples / n_classes
            assert samples_per_class >= 100, (
                f"{benchmark['name']}: only {samples_per_class:.0f} samples/class, too few"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
