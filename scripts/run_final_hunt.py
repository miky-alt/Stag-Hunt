"""Launch the final Hunt ablations: python scripts\\run_final_hunt.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_ablations import run_environment  # noqa: E402

if __name__ == '__main__':
    run_environment('hunt')
