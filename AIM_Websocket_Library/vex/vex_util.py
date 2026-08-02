# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
""" 
AIM WebSocket API - Util

This module defines various utility functions used in the AIM WebSocket API.
"""

import time
from .vex_types import TimeUnits

def sleep(duration: float, units=TimeUnits.MSEC):
    '''### delay the current thread for the provided number of seconds or milliseconds.

    #### Arguments:
        duration: The number of seconds or milliseconds to sleep for
        units:    The units of duration, optional, default is milliseconds

    #### Returns:
        None
    '''
    if units == TimeUnits.MSEC:
        time.sleep(duration / 1000)
    else:
        time.sleep(duration)

def wait(duration: float, units=TimeUnits.MSEC):
    '''### delay the current thread for the provided number of seconds or milliseconds.

    #### Arguments:
        duration: The number of seconds or milliseconds to sleep for
        units:    The units of duration, optional, default is milliseconds

    #### Returns:
        None
    '''
    if units == TimeUnits.MSEC:
        time.sleep(duration / 1000)
    else:
        time.sleep(duration)
