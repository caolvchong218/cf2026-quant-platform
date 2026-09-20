from pathlib import Path
import argparse
from cfquant.data import download_project

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file")
    parser.add_argument("--assets", type=int, default=60)
    args = parser.parse_args()
    download_project(Path(__file__).resolve().parents[1], args.token_file, args.assets)
