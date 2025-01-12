import sys
import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :returns: A formatted error message string.
    """
    # function provided by the sys module that retrieves information about the most recent exception.
    # _: Discards the first two values (type and instance of the exception),
    # exc_tb: Holds the traceback object, which contains detailed information about the error's location.
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception occurred from the traceback object
    # exc_tb.tb_frame: Represents the current stack frame where the exception occurred.
    # tb_frame.f_code: Contains the code object of the current stack frame.
    # f_code.co_filename: Provides the file name (full path) where the error occurred.
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatted error message string with file name, line number, and the actual error
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {str(error)}"

    # Log the error for better tracking
    logging.error(error_message)

    return error_message


class MyException(Exception):
    """
    Custom exception class for handling errors in the Vehicle Insurance Project.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the Custom Exception class with a detailed error message.

        :param error_message: string that describes the nature of the error..
        :param error_detail: reference to the sys module, which is used for extracting traceback information.
        """
        # Call the base class constructor with the error message
        super().__init__(error_message)

        # Format the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message
