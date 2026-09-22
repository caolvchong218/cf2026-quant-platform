from pathlib import Path
from cfquant.benchmark_research import run_benchmark_research

if __name__ == '__main__':
    run_benchmark_research(Path(__file__).resolve().parents[1])
