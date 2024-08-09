import sys
from traceback import format_exception


class OptimagicError(Exception):
    """Base exception for optimagic which should be inherited by all exceptions."""


class TableExistsError(OptimagicError):
    """Exception for database tables that should not exist but do."""


class InvalidFunctionError(OptimagicError):
    """Exception for invalid user provided functions.

    This includes user functions that do not comply with interfaces, raise errors or
    produce NaNs.

    """


class UserFunctionRuntimeError(OptimagicError):
    """Exception that is raised when user provided functions raise errors."""


class MissingInputError(OptimagicError):
    """Exception for missing user provided input."""


class AliasError(OptimagicError):
    """Exception for aliasing errors."""


class InvalidKwargsError(OptimagicError):
    """Exception for invalid user provided keyword arguments."""


class InvalidParamsError(OptimagicError):
    """Exception for invalid user provided parameters."""


class InvalidConstraintError(OptimagicError):
    """Exception for invalid user provided constraints."""


class InvalidBoundsError(OptimagicError):
    """Exception for invalid user provided bounds."""


class InvalidScalingError(OptimagicError):
    """Exception for invalid user provided scaling."""


class InvalidMultistartError(OptimagicError):
    """Exception for invalid user provided multistart options."""


class InvalidNumdiffOptionsError(OptimagicError):
    """Exception for invalid user provided numdiff options."""


class NotInstalledError(OptimagicError):
    """Exception when optional dependencies are needed but not installed."""


class NotAvailableError(OptimagicError):
    """Exception when something is not available, e.g. because a calculation failed."""


class StopOptimizationError(OptimagicError):
    def __init__(self, message, current_status):
        super().__init__(message)
        self.message = message
        self.current_status = current_status

    def __reduce__(self):
        """Taken from here: https://tinyurl.com/y6eeys2f."""
        return (StopOptimizationError, (self.message, self.current_status))


def get_traceback():
    tb = format_exception(*sys.exc_info())
    if isinstance(tb, list):
        tb = "".join(tb)
    return tb


INVALID_INFERENCE_MSG = (
    "Taking the inverse of the information matrix failed. Only ever use this "
    "covariance matrix or standard errors based on it for diagnostic purposes, not for "
    "drawing conclusions."
)


INVALID_SENSITIVITY_MSG = (
    "Taking inverse failed during the calculation of sensitvity measures. Interpret "
    "them with caution."
)
