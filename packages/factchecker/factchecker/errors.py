"""The failures this package raises."""


class InputValidationError(Exception):
    """The payload does not satisfy the contract."""


class AuthenticationFailed(Exception):  # noqa: N818 — the design spec fixes this name
    """A credential was rejected by an upstream service."""
