import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def error_message_details(error_message, error_details:sys):

    _,_, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    return f"The error has been occured in script name : {file_name}, line no : {line_no}, error message : {error_message}"

class BostonHousingException(Exception):

    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message=error_message, error_details=error_details)

    def __str__(self):
        return self.error_message