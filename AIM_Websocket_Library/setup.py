# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
from setuptools import setup, find_packages
setup(
    name='vex',
    version='1.0.1',
    packages=find_packages(where='./vex'),
    package_dir={'': '.'},
    install_requires=[
        'websocket-client'
    ],
    include_package_data=True,
    description='VEX AIM WebSocket Python Client',
    author='Charlie Didear,James Pearman, Levi Pope, Raj Balasubramanian',
    author_email='charlie_didear@innovationfirst.com, james_pearman@innovationfirst.com, levi_pope@innovationfirst.com, raj@vex.com',
    url='https://github.com/VEX-Robotics/AIM_Python_API',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    license_files=['LICENSE'],
    python_requires='>=3.8',
   
)