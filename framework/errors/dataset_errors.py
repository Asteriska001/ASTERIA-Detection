class BenchmarkInitError(OSError, ValueError):
    """Base class for errors raised if a benchmark fails to initialize."""


class DatasetInitError(OSError):
    """Base class for errors raised if a dataset fails to initialize."""

class DatasetNotMatch(OSError):
    """Base class for errors raised if a dataset fails to match a model."""