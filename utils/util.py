# -*- coding: utf-8 -*-
import os
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

'''
日志模块
1. 同时将日志打印到屏幕跟文件中
2. 默认值保留近30天日志文件
'''

def init_logger(log_name,log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if log_name not in Logger.manager.loggerDict:
        logger  = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(filename=os.path.join(log_path,"all.log"),when='D',backupCount = 30)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        console= logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        handler = TimedRotatingFileHandler(filename=os.path.join(log_path,"error.log"),when='D',backupCount= 30)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    logger = logging.getLogger(log_name)
    return logger

# 测试
if __name__ == "__main__":
    logger = init_logger('test','D:\\data')
    logger.info("test")
    logger.error("test2")