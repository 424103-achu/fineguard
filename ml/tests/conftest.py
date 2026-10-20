import sys
from pathlib import Path

# Make `src` and `service` importable when pytest runs from ml/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
