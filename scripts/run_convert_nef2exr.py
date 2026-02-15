import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.nef_2_exr import convert_all

if __name__ == "__main__":
    convert_all()