import logging as lg, os

def start(filename, level='info', append=False, toscreen=False):
    # Logger that allows restarting with a new log file name
    TOSCREEN = toscreen
    format="%(asctime)s - %(levelname)s - %(message)s"
    leveldict = {'debug':lg.DEBUG, 'info':lg.INFO, 'warning':lg.WARNING,
                 'error':lg.ERROR}
    filemode = 'a' if append else 'w'
    logger = lg.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = lg.FileHandler(filename, mode=filemode)
    handler.setFormatter(lg.Formatter(format))
    logger.addHandler(handler)
    if toscreen:
        handler = lg.StreamHandler()
        handler.setFormatter(lg.Formatter(format))
        logger.addHandler(handler)
    logger.setLevel(leveldict[level])    

def info(s):
    lg.info(s)
    
def debug(s):
    lg.debug(s)

def warning(s):
    lg.warning(s)
    
def log(s):
    info(s)

if __name__ == "__main__":
    import log
    log.start('x.log', level='info', append=True)
    log.info('info-message')
    log.debug('debug-message')
    log.warning('warning-message')
