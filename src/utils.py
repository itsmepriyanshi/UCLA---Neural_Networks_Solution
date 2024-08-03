import logging

def setup_logging():
    logging.basicConfig(filename='logs/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def log_error(error_message):
    logging.error(error_message)