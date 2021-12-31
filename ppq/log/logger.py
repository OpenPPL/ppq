import logging
import coloredlogs

ppq_logger = logging.getLogger('PPQ')
coloredlogs.install(fmt='%(levelname)s %(name)s %(asctime)s %(message)s', level=logging.DEBUG, logger=ppq_logger)
ppq_logger.setLevel(logging.INFO)