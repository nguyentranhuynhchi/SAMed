"""Fallback dummy icecream module to prevent crashes when icecream is not installed."""
try:
    from icecream import ic
except ImportError:
    def ic(*args):
        if not args:
            return None
        return args[0] if len(args) == 1 else args
