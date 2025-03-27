class ModelClientError(Exception):
    """Base exception for model client errors."""

    pass


class ModelValidationError(ModelClientError):
    """Exception raised for validation errors."""

    pass


class ModelRequestError(ModelClientError):
    """Exception raised for request errors."""

    pass


class ModelResponseError(ModelClientError):
    """Exception raised for response errors."""

    pass
