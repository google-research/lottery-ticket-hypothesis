# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup file for lottery ticket hypothesis."""

from setuptools import setup

SHORT_DESCRIPTION = """
An implementation of the lottery ticket hypothesis experiment.""".strip()

DEPENDENCIES = [
    'six',
    'fire',
    'tensorflow',
    'keras',
    'numpy',
    'absl-py',
]

VERSION = '1'

setup(
    name='lottery-ticket',
    version=VERSION,
    description=SHORT_DESCRIPTION,

    author='Jonathan Frankle',
    author_email='jfrankle@google.com',
    license='Apache Software License',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
    ],

    keywords='lottery ticket hypothesis',

    packages=['lottery-ticket'],

    package_dir={'lottery-ticket': '.'},

    install_requires=DEPENDENCIES,
)
