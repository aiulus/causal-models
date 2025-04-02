import os
import tempfile
import json
import csv
import pytest
import subprocess
from utils import io


def test_configuration_loader():
    config = io.config
    assert "PATH_SCMs" in config
    assert "PATH_DATA" in config
    assert isinstance(config["COEFFICIENTS"], list)


def test_parse_and_save_scm(tmp_path):
    # Minimal SCM JSON content
    scm_data = {
        "nodes": ["X1", "X2"],
        "edges": [["X1", "X2"]],
        "functions": {"X1": "lambda _: 0", "X2": "lambda X1: X1"},
        "noise": {"X1": ("gaussian", [0, 1]), "X2": ("gaussian", [0, 1])}
    }

    file_path = tmp_path / "test_scm.json"
    with open(file_path, 'w') as f:
        json.dump(scm_data, f)

    nodes, G, F, N = io.parse_scm(str(file_path))
    assert set(nodes) == {"X1", "X2"}
    assert ("X1", "X2") in G.edges
    assert "X2" in F
    assert "X1" in N


def test_save_and_load_csv(tmp_path):
    data = {"X1": [1.0, 2.0], "X2": [3.0, 4.0]}
    file_path = tmp_path / "data.csv"
    io.save_to_csv(data, str(file_path))

    with open(file_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0][0] == "X1"
    assert "1.0" in rows[0][1]


def test_parse_interventions_atomic():
    atomic = ["(X1, 5)", "(X2, 2.5)"]
    parsed = io.parse_interventions(atomic)
    assert parsed == {"X1": "5", "X2": "2.5"}


def test_parse_interventions_extended_dict():
    extended = {
        "X1": 10,
        "X2": {
            "function": "lambda X1: X1 + 1",
            "noise": ("gaussian", [0, 1])
        }
    }
    parsed = io.parse_interventions(extended)
    assert isinstance(parsed["X2"], dict)
    assert parsed["X2"]["function"].startswith("lambda")


def test_scm_args_to_filename():
    class Args:
        n = 3
        graph_type = "chain"
        funct_type = "linear"
        noise_types = ["N(0,1)"]

    path = io.scm_args_to_filename(Args(), "json", "outputs/scms")
    assert path.endswith(".json")
    assert "chain_graph" in path


SCM_FILENAME = "SCM_n3_chain_graph_linear_functions_N(0,1).json"


def test_generate_scm_cli(tmp_path):
    # Run the generate_scm CLI script
    cmd = [
        "python", "scripts/generate_scm.py",
        "--graph_type", "chain",
        "--n", "3",
        "--funct_type", "linear",
        "--noise_types", "N(0,1)",
        "--save"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    print(result.stdout)

    # Check that file was created in the configured output path
    expected_path = os.path.join("outputs", "scms", SCM_FILENAME)
    assert os.path.exists(expected_path)

    # Check JSON structure
    with open(expected_path, "r") as f:
        scm_data = json.load(f)
    assert "nodes" in scm_data and "functions" in scm_data


def test_sample_data_l1_cli(tmp_path):
    cmd = [
        "python", "scripts/sample_data.py",
        "--file_name", SCM_FILENAME,
        "--mode", "l1",
        "--n_samples", "10",
        "--save"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    print(result.stdout)

    expected_csv = SCM_FILENAME.replace(".json", ".csv")
    expected_path = os.path.join("outputs", "data", expected_csv)
    assert os.path.exists(expected_path)


def test_sample_data_l2_cli(tmp_path):
    cmd = [
        "python", "scripts/sample_data.py",
        "--file_name", SCM_FILENAME,
        "--mode", "l2",
        "--n_samples", "10",
        "--do", "(X1,5)",
        "--save"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    print(result.stdout)

    expected_csv = SCM_FILENAME.replace(".json", ".csv")
    expected_path = os.path.join("outputs", "data", expected_csv)
    assert os.path.exists(expected_path)
