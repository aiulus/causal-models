import subprocess
import os
import pytest

GRAPH_TYPES = ["chain", "parallel", "random"]

@pytest.mark.parametrize("graph_type", GRAPH_TYPES)
def test_scm_generation_topologies(graph_type):
    args = [
        "python", "scripts/generate_scm.py",
        "--graph_type", graph_type,
        "--n", "4",
        "--funct_type", "linear",
        "--noise_types", "N(0,1)",
        "--save"
    ]

    if graph_type == "random":
        args += ["--p", "0.6", "--pa_n", "2"]

    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0



