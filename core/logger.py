import logging

from datetime import datetime
from logging.handlers import WatchedFileHandler
from logging import StreamHandler

logger = logging.getLogger('scikit-tester')


class CustomFormatter(logging.Formatter):
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%06d" % (t, record.msecs)
        return s


def set_logger():
    if logger.hasHandlers():
        return

    logger.setLevel(logging.DEBUG)

    formatter = CustomFormatter(
        '[%(levelname)s %(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )

    file_handler = WatchedFileHandler('logs/output.log', encoding='utf8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info('Added logging handler: ' + str(file_handler))

    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.info('Added logging handler: ' + str(console_handler))

    logger.info('Set new logger up.')
    return
