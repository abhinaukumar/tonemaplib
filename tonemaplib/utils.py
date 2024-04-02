import os
from types import ModuleType
from typing import Tuple, List, Dict


# Borrowed from Netflix/vmaf.
def import_python_file(filepath: str) -> ModuleType:
    '''
    Import a python file as a module.

    Args:
        filepath: Path to Python file.

    Returns:
        ModuleType: Loaded module.
    '''
    filename = os.path.basename(filepath).rsplit('.', 1)[0]
    try:
        from importlib.machinery import SourceFileLoader
        ret = SourceFileLoader(filename, filepath).load_module()
    except ImportError:
        import imp
        ret = imp.load_source(filename, filepath)
    return ret


def load_args(filepath: str) -> Tuple[List, Dict]:
    '''
    Load arguments from a Python file having the args, kwargs attributes.

    Args:
        filepath: Path to Python file containing arguments.

    Returns:
        Tuple[List, Dict]: Positional and keyword arguments respectively.
    '''
    if filepath is None:
        return [], {}
    args_module = import_python_file(filepath)
    args = args_module.args if hasattr(args_module, 'args') else []
    kwargs = args_module.kwargs if hasattr(args_module, 'kwargs') else {} 
    if not isinstance(args, list):
        raise TypeError('args must be of type List.')
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be of type Dict.')
    return args, kwargs
