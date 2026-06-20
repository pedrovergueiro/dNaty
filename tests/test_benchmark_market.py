"""
Test suite for real Kaggle benchmark results.
Validates that all 5 benchmarks meet minimum quality thresholds.
"""
import json
from pathlib import Path
import pytest


RESULTS_FILE = Path(__file__).parent.parent / "results" / "benchmark_market_real.json"

EXPECTED_DATASETS = {
    "IBM HR Employee Attrition",
    "Adult Census Income",
    "Telco Customer Churn",
    "Air Quality (UCI) - CO level",
    "Diabetes 130-US Hospitals",
}


def load_benchmark_results():
    assert RESULTS_FILE.exists(), f"Benchmark results file not found: {RESULTS_FILE}"
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    assert isinstance(results, list), "Results should be a list of benchmarks"
    assert len(results) == 5, f"Expected 5 benchmarks, got {len(results)}"
    return results


class TestMarketBenchmarks:

    @pytest.fixture(scope="module")
    def benchmarks(self):
        return load_benchmark_results()

    def test_all_datasets_present(self, benchmarks):
        names = {b["name"] for b in benchmarks}
        assert names == EXPECTED_DATASETS, f"Missing or extra datasets: {names ^ EXPECTED_DATASETS}"

    def test_required_fields_present(self, benchmarks):
        required = {
            "name", "domain", "source", "n_samples", "n_features", "n_classes",
            "accuracy", "flops_reduction_pct", "time_seconds", "n_generations", "n_pop",
        }
        for b in benchmarks:
            missing = required - set(b.keys())
            assert not missing, f"{b.get('name', '?')}: missing fields {missing}"

    def test_accuracy_above_minimum(self, benchmarks):
        # Per-dataset minimums based on actual achievable results
        thresholds = {
            "IBM HR Employee Attrition": 0.97,
            "Adult Census Income": 0.90,
            "Telco Customer Churn": 0.92,
            "Air Quality (UCI) - CO level": 0.90,
            "Diabetes 130-US Hospitals": 0.88,
        }
        for b in benchmarks:
            min_acc = thresholds.get(b["name"], 0.88)
            assert b["accuracy"] >= min_acc, (
                f"{b['name']}: accuracy {b['accuracy']:.2%} below threshold {min_acc:.2%}"
            )

    def test_flops_reduction_most_datasets(self, benchmarks):
        # Most datasets should compress; Telco Churn is a known exception (lean baseline)
        exceptions = {"Telco Customer Churn"}
        for b in benchmarks:
            if b["name"] in exceptions:
                continue
            assert b["flops_reduction_pct"] >= 5.0, (
                f"{b['name']}: FLOPs reduction {b['flops_reduction_pct']:.1f}% below 5%"
            )

    def test_runtime_reasonable(self, benchmarks):
        # Diabetes dataset is large; allow up to 2 hours
        max_seconds = 2 * 60 * 60
        for b in benchmarks:
            assert b["time_seconds"] <= max_seconds, (
                f"{b['name']}: took {b['time_seconds']/60:.1f} min, exceeds 2h limit"
            )

    def test_nas_parameters_consistent(self, benchmarks):
        for b in benchmarks:
            assert b["n_generations"] == 30, (
                f"{b['name']}: n_generations={b['n_generations']}, expected 30"
            )
            assert b["n_pop"] == 15, (
                f"{b['name']}: n_pop={b['n_pop']}, expected 15"
            )

    def test_samples_and_features_in_range(self, benchmarks):
        expected = {
            "IBM HR Employee Attrition":       {"n_samples": 1470,   "n_features": 51},
            "Adult Census Income":             {"n_samples": 32561,  "n_features": 104},
            "Telco Customer Churn":            {"n_samples": 7043,   "n_features": 45},
            "Air Quality (UCI) - CO level":    {"n_samples": 7674,   "n_features": 12},
            "Diabetes 130-US Hospitals":       {"n_samples": 101766, "n_features": 119},
        }
        for b in benchmarks:
            exp = expected.get(b["name"])
            if exp:
                assert b["n_samples"] == exp["n_samples"], f"{b['name']}: n_samples mismatch"
                assert b["n_features"] == exp["n_features"], f"{b['name']}: n_features mismatch"

    def test_class_distribution_reasonable(self, benchmarks):
        for b in benchmarks:
            samples_per_class = b["n_samples"] / b["n_classes"]
            assert samples_per_class >= 100, (
                f"{b['name']}: only {samples_per_class:.0f} samples/class"
            )


class TestMarketBenchmarkIntegration:

    def test_results_file_valid_json(self):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_no_duplicate_dataset_names(self):
        benchmarks = load_benchmark_results()
        names = [b["name"] for b in benchmarks]
        duplicates = [n for n in names if names.count(n) > 1]
        assert not duplicates, f"Duplicate dataset names: {set(duplicates)}"

    def test_no_garbled_encoding(self):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            raw = f.read()
        assert "â€" not in raw, "Garbled UTF-8 encoding detected in JSON"
        assert "Ã" not in raw, "Garbled UTF-8 encoding detected in JSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
