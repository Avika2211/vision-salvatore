# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
Provides AimExample base class, which parses arguments, tries to find the IP of AIM Robot, 
and initializes Robot instance
"""

import sys
import os
import socket
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #use of abspath is necessary if Python < 3.9
from utils.logger import Logger
from vex import Robot

class AimExampleBase:
    """
    AimExample is a class that provides functionality to connect to and initialize an AIM robot.
    Attributes:
        parser (argparse.ArgumentParser): Argument parser for command-line options.
        parent_name (str): Name of the parent class.
        child_name (str): Name of the current class.
        logger (Logger): Logger instance for logging messages.
        robot (Robot): Instance of the Robot class.
        args (argparse.Namespace): Parsed command-line arguments.
        aim_host (str): Hostname or IP address of the AIM robot.
    Methods:
        __init__(): Initializes the AimExample instance, sets up the argument parser, and initializes attributes.
        parse_args(): Parses command-line arguments and sets the aim_host attribute.
        init_robot(): Initializes the robot by parsing arguments and creating a Robot instance.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--host', type=str, help="host can be the hostname (e.g. AIM-XXXXXXX), IP address, or even domain name belonging to the specific AIM robot to connect to")
        # self.parser.add_argument('--hostname', type=str, help="Connect to AIM robot using its hostname.")
        self.parent_name = self.__class__.__bases__[0].__name__
        self.child_name = self.__class__.__name__
        self.logger = Logger(NormalLog=True)
        self.robot = None
        self.args = None
        self.aim_host = None
        self.logger.info("initializing: %s->%s" %(self.parent_name, self.child_name))

    def _parse_args(self):
        """
        Parses all arguments.  Called by init_robot(), which subclasses should invoke after any additional arguments
        are added.
        """
        self.args = self.parser.parse_args()

        if self.args.host:
            self.aim_host = self.args.host
        else:
            self.logger.warn("No host given; using the default IP address from vex/settings.json.")
            self.aim_host = ""


    def init_robot(self):
        """
        Initializes the robot by parsing the arguments and creating a Robot instance.

        Call this function after any additional arguments are added with self.parser.add_argument().
        
        This method first processes any command-line arguments. It then creates an instance of the `Robot` class using the
        ip address of the AIM robot. If the ip address is not provided, it assumes that the AIM robot is in AP mode and
        uses the default ip address.

        Returns:
            None
        """
        self._parse_args()
        self.robot = Robot(self.aim_host)
        # print out additional info (hostname, aliases, and ip address) 
        if self.robot:
            hostname = aliases = ip = None
            try:
                hostname,aliases, [ip] = socket.gethostbyaddr(self.robot.host) # Reverse DNS lookup
            except socket.herror as error:
                # print(f"AIM appears to be in AP mode, with address: {self.robot.host}")
                pass
            if hostname:
                self.logger.info(f"You may connect to AIM robot with any of the following:  hostname: {hostname}, alias-list: {aliases}, IP: {ip}")