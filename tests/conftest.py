import sys
from pathlib import Path

# Algorithm packages live at the repo root (kmeans/, fft/, ...), as Home.py expects.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
