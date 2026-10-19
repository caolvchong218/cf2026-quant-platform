"""Replay preserved CogAlpha proposals or run the registered bounded study."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfquant.cogalpha_study import main

if __name__ == "__main__":
    main()
