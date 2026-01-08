LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
        }
    },

    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "cimxml.log",
            "encoding": "utf-8"
        },
        "stream_handler": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
    },

    "loggers": {
        "cimxml_logger": {
            "level": "DEBUG",
            "handlers": ["file_handler", "stream_handler"],
            "propagate": False
        }
    }
}

