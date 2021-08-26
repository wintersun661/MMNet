import logging

import logging.handlers

import time
import os


class Logger():
    def __init__(self, file_path='./logs', suffix='default', time_stamp=False, LOG_LEVEL=logging.DEBUG,
                 FILE_LEVEL=logging.INFO, STREAM_LEVEL=logging.DEBUG,
                 LOG_FORMAT="%(asctime)s - %(levelname)s - %(message)s",
                 ):

        self.file_path = file_path
        self.suffix = suffix
        self.time_stamp = time_stamp

        self.logger_level = LOG_LEVEL
        self.file_level = FILE_LEVEL
        self.stream_level = STREAM_LEVEL

        self.log_format = LOG_FORMAT

    def createLogger(self):
        logger = logging.getLogger('activeLogger')

        logger.setLevel(self.logger_level)
        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)

        # set up logfile name and create instance.

        file_name = 'log_'+str(self.suffix)

        if self.time_stamp:
            curTime = int(round(time.time()*1000))
            curTimeStr = time.strftime(
                '%Y-%m-%d_%H:%M:%S', time.localtime(curTime/1000))

            file_name += '_'+str(curTimeStr)

        fileHandler = logging.FileHandler(
            os.path.join(self.file_path, file_name+'.log'))
        fileHandler.setFormatter(logging.Formatter(self.log_format))
        fileHandler.setLevel(self.file_level)

        # set up stream instance
        streamHandler = logging.StreamHandler(stream=None)
        streamHandler.setLevel(self.stream_level)
        streamHandler.setFormatter(logging.Formatter(self.log_format))

        # attach handlers to logger
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

        self.logger = logger

        return logger
