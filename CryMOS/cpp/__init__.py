# TODO for now
available = False
mod = None

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
    PotLoop = mod.PotLoop
    PotLoopFD = mod.PotLoopFD
    PotLoopGildenblat = mod.PotLoopGildenblat
    test_fdint = mod.test_fdint
