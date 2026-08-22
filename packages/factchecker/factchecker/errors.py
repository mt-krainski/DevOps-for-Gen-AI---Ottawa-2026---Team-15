"""The failures this package raises."""


class InputValidationError(Exception):
    """The payload does not satisfy the contract."""


class AuthenticationFailed(Exception):  # noqa: N818 — the README publishes this name
    """A credential was rejected by an upstream service.

    The name carries no `Error` suffix because the checker seam publishes it: a
    checking agent raises this class, by this name, to end a run.
    """
