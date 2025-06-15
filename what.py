import warnings
UserWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from . import what  # noqa: F401
    from . import what2  # noqa: F401
    from . import what3  # noqa: F401
    from . import what4  # noqa: F401
    from . import what5  # noqa: F401
    from . import what6  # noqa: F401
    from . import what7  # noqa: F401

def what():
    print("what")   
    return 0


if __name__ == "__main__":
    what()
    print("This is the main module of the 'what' package.")