import logging
import sys
from app.config import settings

def setup_logger(name: str = "StudentRepublicBot"):
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()