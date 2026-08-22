"""The failures this package raises."""


class InputValidationError(Exception):
    """The payload does not satisfy the contract."""


class AuthenticationFailed(Exception):  # noqa: N818 — the README publishes this name
    """A credential was rejected by an upstream service.

    The name carries no `Error` suffix because the checker seam publishes it: a
    checking agent raises this class, by this name, to end a run.
    """


class CheckFailed(Exception):  # noqa: N818 — the README publishes this name
    """A check failed, and the checker names the failure itself.

    A checker raises this rather than let the orchestrator assign `check_failed` to
    every exception alike. The kind reaches the output payload, so it is written for
    the person reading that payload.

    The name carries no `Error` suffix for the same reason `AuthenticationFailed`
    carries none: the checker seam publishes it.
    """

    def __init__(self, kind: str, message: str) -> None:
        """Name the failure and describe it.

        Args:
            kind: What sort of failure this is, as the output payload reports it.
            message: What went wrong, in this package's own words. Nothing an
                upstream library wrote belongs here: the message reaches a
                user-visible artifact, and an upstream message can name the Bright
                Data endpoint, which is the credential.
        """
        super().__init__(message)
        self.kind = kind
        self.message = message


class McpCallError(Exception):
    """A call to the Bright Data MCP server failed, and no retry will change that.

    The message says what the server did in this package's own words. Nothing an
    upstream library wrote reaches it, because an upstream message names the request
    URL and the Bright Data token travels inside that URL.
    """
