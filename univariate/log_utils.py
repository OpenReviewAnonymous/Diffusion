import os
import logging
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger
    
    Parameters:
    - name: Logger name
    - log_file: Log file path, use default if None
    - level: Log level
    
    Returns:
    - logger: Configured logger
    """
    # If no log file path provided, use default path
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/{name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger 