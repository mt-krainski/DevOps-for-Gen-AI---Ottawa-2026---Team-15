"""The failures this package raises."""


class InputValidationError(Exception):
    """The payload does not satisfy the contract."""


class AuthenticationFailed(Exception):  # noqa: N818 — the README publishes this name
    """A credential was rejected by an upstream service.

    The name carries no `Error` suffix because the checker seam publishes it: a
    checking agent raises this class, by this name, to end a run.
    """


class McpCallError(Exception):
    """A call to the Bright Data MCP server failed, and no retry will change that.

    The message says what the server did in this package's own words. Nothing an
    upstream library wrote reaches it, because an upstream message names the request
    URL and the Bright Data token travels inside that URL.
    """
