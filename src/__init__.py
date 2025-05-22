import importlib
import pkgutil
from pathlib import Path

# Get the models directory path
models_dir = Path(__file__).parent / 'infrastructure' / 'models'

# Dictionary to store imported adapters
adapters = {}

# Iterate through all subdirectories in models
for _, name, is_pkg in pkgutil.iter_modules([str(models_dir)]):
    if is_pkg and name not in ['__pycache__']:
        try:
            # Try to import the adapter module from each package
            module = importlib.import_module(f"src.infrastructure.models.{name}.adapter")
            # Store the adapter in the dictionary
            adapters[name] = module
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Warning: Could not import adapter from {name}: {e}")
