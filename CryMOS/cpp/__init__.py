# TODO for now
available = False

try:
    import CryMOS.cpp.mod as mod
    available = True
except ImportError:
    from .build import import_inplace

    try:
        mod = import_inplace()
        available = True
    except Exception as e:
        from warnings import warn
        warn("Could not build C++ extension\n " + e, ImportWarning)

if available:
    from .mod import *
