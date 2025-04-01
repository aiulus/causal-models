import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli import scm_cli

if __name__ == "__main__":
    scm_cli.main()
