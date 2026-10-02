import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

