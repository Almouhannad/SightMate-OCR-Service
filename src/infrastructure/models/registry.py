from typing import Type
from src.domain.ports import OCRPort

_ADAPTERS: dict[str, Type[OCRPort]] = {}

def register_adapter(name: str):
    """
    Decorator to register an OCR adapter implementation.
    
    Args:
        name: Unique identifier for the adapter
        
    Returns:
        Decorator function that registers the adapter class
    """
    def decorator(cls: Type[OCRPort]):
        _ADAPTERS[name] = cls
        return cls
    return decorator

def get_adapter(name: str) -> Type[OCRPort]:
    """
    Get an OCR adapter class by name.
    
    Args:
        name: Name of the registered adapter
        
    Returns:
        The adapter class
        
    Raises:
        ValueError: If no adapter is registered with the given name
    """
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise ValueError(f"No OCR adapter registered under name {name!r}")

def list_available_adapters() -> list[str]:
    """
    Get a list of all registered adapter names.
    
    Returns:
        List of registered adapter names
    """
    return list(_ADAPTERS.keys()) 