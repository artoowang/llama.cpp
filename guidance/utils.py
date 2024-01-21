import colorlog
import logging

# This allows us to get the same logger from all modules, and the logger is
# only initialized once.
_logger_initialized: bool = False


def get_logger() -> logging.Logger:
    global _logger_initialized
    logger = logging.getLogger()

    if not _logger_initialized:
        logger.setLevel(logging.DEBUG)

        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s:%(name)s:%(reset)s %(message)s"
        ))
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler("root.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s:%(name)s: %(message)s"
        ))
        logger.addHandler(file_handler)

        logger.info("Logger is now initialized.")
        _logger_initialized = True

    return logger
