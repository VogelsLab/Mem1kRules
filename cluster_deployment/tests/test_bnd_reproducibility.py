"""Regression tests for the corrected BND command interface."""

import importlib.util
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


CLUSTER_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_PATH = CLUSTER_ROOT / "synapsbi" / "simulator" / "simulator.py"


def load_simulator_module():
    """Load simulator.py without requiring the optional analysis dependencies."""
    torch = types.ModuleType("torch")
    synapsbi = types.ModuleType("synapsbi")
    utils = types.ModuleType("synapsbi.utils")
    utils.read_monitor_spiketime_files = lambda **kwargs: None
    utils.read_monitor_weights_files = lambda **kwargs: None

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("synapsbi", synapsbi)
    sys.modules.setdefault("synapsbi.utils", utils)

    spec = importlib.util.spec_from_file_location("bnd_simulator_module", SIMULATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_simulator_module()


def simulator_params():
    return {
        "auryn_sim_dir": "/opt/auryn/",
        "name": "sim_BND_IF_EEEIIEII_6pPol",
        "NE": 4096,
        "NI": 1024,
        "tau_ampa": 0.005,
        "tau_gaba": 0.01,
        "tau_nmda": 0.1,
        "ampa_nmda_ratio": 0.3,
        "wmax": 20,
        "eta": 0.01,
        "wee": 0.1,
        "wei": 0.1,
        "wie": 1,
        "wii": 1,
        "sparseness": 0.1,
        "N_input": 10000,
        "radius": 8,
        "N_active_input": 500,
        "active_input_rate": 60,
        "rate_poisson": 10,
        "weight_poisson": 0.075,
        "ontime_train": 1,
        "offtime_train": 0.01,
        "ontime_test": 1,
        "offtime_test": 3,
        "max_rate_checker": 100,
        "tau_checker": 1,
        "lpt": 30,
        "lt": 10,
        "workdir": "/tmp/bnd",
        "n_recorded": 4096,
        "record_i": True,
        "n_recorded_i": 500,
    }


class BNDReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.simulator = MODULE.Simulator_BND_IF_EEEIIEII_6pPol(
            simulator_params()
        )
        self.theta = list(range(24))

    def test_formats_exactly_four_six_parameter_rules(self):
        rule = self.simulator.format_rule(self.theta)
        self.assertIn("--ruleEE a0a1a2a3a4a5a", rule)
        self.assertIn("--ruleEI a6a7a8a9a10a11a", rule)
        self.assertIn("--ruleIE a12a13a14a15a16a17a", rule)
        self.assertIn("--ruleII a18a19a20a21a22a23a", rule)

    def test_legacy_nuisance_value_is_ignored(self):
        self.assertEqual(
            self.simulator.format_rule(self.theta),
            self.simulator.format_rule(self.theta + [999]),
        )

    def test_rejects_invalid_rule_lengths(self):
        for length in (0, 23, 26):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    self.simulator.format_rule(range(length))

    def test_command_separates_output_id_and_seed(self):
        command = self.simulator.format_command(self.theta, "rule-id", 202)
        self.assertIn("--ID rule-id", command)
        self.assertIn("--seed 202", command)
        self.assertIn("--N_input 10000", command)
        self.assertNotIn("--N_inputs", command)

    def test_seed_is_optional_for_id_hash_fallback(self):
        command = self.simulator.format_command(self.theta, "rule-id")
        self.assertNotIn("--seed", command)

    def test_rejects_seed_outside_unsigned_32_bit_range(self):
        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    self.simulator.format_command(self.theta, "rule-id", seed)

    def test_parameter_file_generation_uses_validated_bnd_command(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "params.txt"
            with redirect_stdout(io.StringIO()):
                MODULE.make_param_files_cluster(
                    self.simulator,
                    [self.theta + [999]],
                    ["rule-id"],
                    destination,
                    simulation_seeds=[202],
                )
            command = destination.read_text().strip()
        self.assertIn("--ID rule-id", command)
        self.assertIn("--seed 202", command)
        self.assertNotIn("999", command)


if __name__ == "__main__":
    unittest.main()
