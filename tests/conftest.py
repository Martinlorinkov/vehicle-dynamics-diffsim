import os
import sys

# lets pytest find vdsim without installing anything
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
