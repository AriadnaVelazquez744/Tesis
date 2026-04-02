"""Optional-debug shims so we do not require `ipdb` at runtime."""


def set_trace(*_args, **_kwargs):
    """No-op stand-in for `ipdb.set_trace` (debug breakpoints)."""
