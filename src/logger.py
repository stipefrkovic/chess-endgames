import logging

# TODO add name to logger file

logger = logging.getLogger(__name__)

# Create handlers
terminal_handler = logging.StreamHandler()
file_handler = logging.FileHandler('src/log_files/file.log', mode='w')

terminal_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

terminal_format = logging.Formatter('%(levelname)s - %(message)s')
file_format = logging.Formatter('%(message)s')

terminal_handler.setFormatter(terminal_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(terminal_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG)
