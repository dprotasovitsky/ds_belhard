import logging
from datetime import datetime
import os


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Настройка логгера с указанным именем и уровнем логирования"""

    # Создаем директорию для логов если ее нет
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Создаем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Настраиваем логгер
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Добавляем handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
