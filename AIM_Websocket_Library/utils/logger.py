# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
Logger utility

Logs a string to stdout with various log levels that are color-coded.
------------------------------------------------------------------------------------
"""
import json

class ANSIColorCodes():
    """
    ANSI color codes for color formatting on terminals
    """
    BLACK   = '\x1b[30m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    BLUE    = '\x1b[34m'
    MAGENTA = '\x1b[35m'
    CYAN    = '\x1b[36m'
    WHITE   = '\x1b[37m'

class LogCmd():
    """
    Base class for formatting messages depending on log level
    """
    class Level():
        """
        Log levels
        """
        DEBUG = 0
        SUCCESS = 1
        INFO = 2
        WARN = 3
        ERROR = 4


    def __init__(self, message:str, level=Level.INFO) -> None:
        self.level = level
        self.message = message

    def getLevelAsStr(self, level:int):
        """
        Get the Log Level as a string
        """
        if level == LogCmd.Level.DEBUG:
            return '[DEBUG]'
        elif level == LogCmd.Level.ERROR:
            return '[ERROR]'
        elif level == LogCmd.Level.WARN:
            return '[WARN]'
        elif level == LogCmd.Level.INFO:
            return '[INFO]'
        elif level == LogCmd.Level.SUCCESS:
            return '[SUCCESS]'
        
    def colorText(self, level, message)->str:
        """
        Formats message with ANSI color code.
        """
        if level == LogCmd.Level.DEBUG:
            return f'{ANSIColorCodes.MAGENTA}{message}\x1b[0m'
        elif level == LogCmd.Level.ERROR:
            return f'{ANSIColorCodes.RED}{message}\x1b[0m'
        elif level == LogCmd.Level.WARN:
            return f'{ANSIColorCodes.YELLOW}{message}\x1b[0m'
        elif level == LogCmd.Level.INFO:
            return f'{ANSIColorCodes.BLUE}{message}\x1b[0m'
        elif level == LogCmd.Level.SUCCESS:
            return f'{ANSIColorCodes.GREEN}{message}\x1b[0m'
        else:
            return message

    def toJSON(self) -> str:
        """
        Return Log Command as JSON
        """
        return json.dumps(self.__dict__)
    
    def toStr(self) -> str:
        """
        Return Log Command as human-readable string 
        """
        levelStr = self.getLevelAsStr(self.level)
        return self.colorText(self.level, f'{levelStr}: {self.message}')
    
    def send(self, REGULAR_LOG=True):
        """
        Prints message to stdout, either as string or JSON message
        """
        result = ''
        if REGULAR_LOG:
            result = self.toStr()
            result += '\r\n' # DOS newline
        else:
            result = self.toJSON()
            result = result + '@' # use @ delimiter instead of newline

        print(result, end='') #don't add a newline

# ---------------------------------------------------------------------------- #
# Logger Class
class Logger():
    """
        Contains various functions to log text to console.
    """
    def __init__(self, NormalLog = True):
        self.NormalLog = NormalLog

    def success(self, message:str):
        LogCmd(message,LogCmd.Level.SUCCESS).send(self.NormalLog)

    def debug(self, message:str):
        LogCmd(message,LogCmd.Level.DEBUG).send(self.NormalLog)

    def info(self, message:str):
        LogCmd(message,LogCmd.Level.INFO).send(self.NormalLog)

    def warn(self, message:str):
        LogCmd(message,LogCmd.Level.WARN).send(self.NormalLog)

    def error(self, message:str):
        LogCmd(message,LogCmd.Level.ERROR).send(self.NormalLog)