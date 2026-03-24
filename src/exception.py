import sys
from src.logger import logging 

import sys

def error_message_detail(error, error_detail: sys):
    # Only takes the traceback object (last element).
    _, _, exc_tb = error_detail.exc_info()
    # Get the filename where error occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Gets the line number of the error , file number and what error occured
    return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

print("✅ Loaded CustomException class from:", __file__)