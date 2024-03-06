import logging
import os
from datetime import date

def get_logger(file_name, log_mode = 'file', testing=False, task=None):
    logger = logging.getLogger('')
    dir_path = './logging'
    # Set the logging level
    curr_date = date.today()
    logger.setLevel(logging.INFO)
    if task !=None:
        dir_path = f"{dir_path}/{task}-{curr_date}"
    # Create a file handler and set its level
    handler = logging.StreamHandler()
    if testing:
        dir_path = dir_path + '/testing'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if log_mode == 'file':
        handler = logging.FileHandler(f'{dir_path}/{file_name}.log')
    
    handler.setLevel(logging.INFO)

    # Create a log message formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
        
    logger.addHandler(handler)
    
    return logger