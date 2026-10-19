"""Run the registered RSRS study; supports installed package or source checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfquant.rsrs_study import main

if __name__ == "__main__":
    main()
