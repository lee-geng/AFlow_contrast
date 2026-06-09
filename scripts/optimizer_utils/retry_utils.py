RETRYABLE_ERROR_KEYWORDS = (
    "connection error",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "service unavailable",
    "server error",
    "bad gateway",
    "gateway timeout",
    "apiconnectionerror",
    "apitimeouterror",
    "server disconnected",
)


def is_retryable_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in RETRYABLE_ERROR_KEYWORDS)
