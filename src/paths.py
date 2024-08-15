import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
RESULT_DIR = ROOT_DIR / "result"