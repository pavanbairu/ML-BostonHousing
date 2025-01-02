import logging
import os
from datetime import datetime

LOGFILE = f"{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs", LOGFILE)

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOGFILE)

print("file name :", LOGFILE)
print("log path :", LOG_PATH)
print("file log path :", LOG_FILE_PATH)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
