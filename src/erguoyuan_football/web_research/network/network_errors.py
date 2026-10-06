"""Research-facing error categories; transport stays in network/CoreNetworkClient."""

from __future__ import annotations


class NetworkError(RuntimeError):
    """Base classified research connection error."""


class NetworkUnavailableError(NetworkError):
    """No usable registered network path is available."""


class ProviderUnavailableError(NetworkError):
    """No configured provider can serve this task."""


class ProviderTimeoutError(NetworkError):
    """A bounded request timed out."""


class ProviderAuthenticationError(NetworkError):
    """Registered provider rejected configured authentication."""


class ProviderRateLimitError(NetworkError):
    """Registered provider exhausted its bounded rate-limit retry budget."""


class InvalidProviderResponseError(NetworkError):
    """Payload failed source, timestamp or type validation."""
