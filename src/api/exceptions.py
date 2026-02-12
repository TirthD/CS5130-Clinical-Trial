class ClinicalTrialsAPIError(Exception):
    """Base exception for all API errors."""

    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(ClinicalTrialsAPIError):
    """Raised when API rate limit (50 req/min) is exceeded."""

    def __init__(self):
        super().__init__(
            "Rate limit exceeded. The API allows ~50 requests per minute.",
            status_code=429,
        )


class StudyNotFoundError(ClinicalTrialsAPIError):
    """Raised when a specific NCT ID is not found."""

    def __init__(self, nct_id: str):
        self.nct_id = nct_id
        super().__init__(
            f"Study not found: {nct_id}",
            status_code=404,
        )


class InvalidParameterError(ClinicalTrialsAPIError):
    """Raised when an invalid query parameter is passed."""

    def __init__(self, message: str):
        super().__init__(
            f"Invalid parameter: {message}",
            status_code=400,
        )