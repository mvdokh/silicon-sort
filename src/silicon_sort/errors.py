"""Shared exceptions."""


class SiliconSortError(Exception):
    """Base error for expected pipeline failures."""


class GeometryError(SiliconSortError):
    """Probe geometry file could not be parsed."""


class RecordingError(SiliconSortError):
    """Binary recording metadata is missing or inconsistent."""


class Cancelled(SiliconSortError):
    """The user cancelled a running stage."""
