import subprocess
import os
import pytest
import json

SCM_FILENAME = "SCM_n3_chain_graph_linear_functions_N(0,1).json"
SCM_FILE = os.path.join("outputs", "scms", SCM_FILENAME)
DATA_DIR = "data"
SCM_DIR = "scms"

@pytest.mark.parametrize("args,expected_file", [
    (
        ["scripts/generate_scm.py",
         "--graph_type", "chain", "--n", "3",
         "--funct_type", "linear", "--noise_types", "N(0,1)", "--save"],
        os.path.join("config", "global_variables.json")
    )
])
def test_generate_scm_file(tmp_path, args, expected_file):
    # Ensure config is available
    assert os.path.exists(expected_file)

    result = subprocess.run(["python"] + args, capture_output=True, text=True)
    assert result.returncode == 0
    assert f"SCM saved to" in result.stdout


def test_sample_l1_data(tmp_path):
    # Ensure SCM file exists
    if not os.path.exists(SCM_FILE):
        subprocess.run([
            "python", "scripts/generate_scm.py",
            "--graph_type", "chain", "--n", "3",
            "--funct_type", "linear", "--noise_types", "N(0,1)", "--save"
        ], check=True)

    cmd = [
        "python", "scripts/sample_data.py",
        "--file_name", SCM_FILE,
        "--mode", "l1",
        "--n_samples", "10",
        "--save"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, result.stderr)
    assert result.returncode == 0


def test_sample_l2_do(tmp_path):
    # Ensure SCM file exists
    if not os.path.exists(SCM_FILE):
        subprocess.run([
            "python", "scripts/generate_scm.py",
            "--graph_type", "chain", "--n", "3",
            "--funct_type", "linear", "--noise_types", "N(0,1)", "--save"
        ], check=True)

    cmd = [
        "python", "scripts/sample_data.py",
        "--file_name", SCM_FILE,
        "--mode", "l2",
        "--n_samples", "10",
        "--do", "(X1,5)",
        "--save"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
