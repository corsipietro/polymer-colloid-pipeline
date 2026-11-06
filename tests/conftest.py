"""Make the ``pipeline`` scripts importable from the tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
