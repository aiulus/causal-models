import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli import sample_cli

if __name__ == "__main__":
    sample_cli.main()
