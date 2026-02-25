
import os
import sys

print("Setting HF_HOME...")
os.environ["HF_HOME"] = "c:/1 Sanjay/Political_SA/.cache"
os.environ["TRANSFORMERS_VERBOSITY"] = "detail"

print("Starting Import Test V2")
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    print("Importing AutoTokenizer...")
    from transformers import AutoTokenizer
    print("AutoTokenizer imported.")
except ImportError as e:
    print(f"Failed to import AutoTokenizer: {e}")

try:
    print("Importing AutoModel...")
    from transformers import AutoModel
    print("AutoModel imported.")
except ImportError as e:
    print(f"Failed to import AutoModel: {e}")

print("Done.")
