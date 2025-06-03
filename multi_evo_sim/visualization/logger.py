import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log(message):
    logger.info(message)
