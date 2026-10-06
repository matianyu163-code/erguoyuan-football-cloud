"""Classified transport errors for fail-closed providers."""


class NetworkError(RuntimeError):
    """Base error retaining a machine-readable failure category."""

    code = "NETWORK_ERROR"

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class SourceNotConfigured(NetworkError):
    code = "SOURCE_NOT_CONFIGURED"


class NetworkDependencyUnavailable(NetworkError):
    code = "NETWORK_CLIENT_DEPENDENCY_MISSING"


class RetryExhausted(NetworkError):
    """Bounded retries exhausted, retaining the attempt count for audit."""

    code = "RETRY_EXHAUSTED"

    def __init__(self, message: str, *, cause: Exception | None = None,
                 attempts: int | None = None) -> None:
        super().__init__(message, cause=cause)
        self.attempts = attempts


class InvalidNetworkResponse(NetworkError):
    code = "INVALID_RESPONSE"
