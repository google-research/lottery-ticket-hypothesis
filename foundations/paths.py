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

"""Utilities for building paths to store data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def initial(parent_directory):
  """The path where the weights at the beginning of training are stored."""
  return os.path.join(parent_directory, 'initial')


def final(parent_directory):
  """The path where the weights at the end of training are stored."""
  return os.path.join(parent_directory, 'final')


def masks(parent_directory):
  """The path where the pruning masks are stored."""
  return os.path.join(parent_directory, 'masks')


def log(parent_directory, name):
  """The path where training/testing/validation logs are stored."""
  return os.path.join(parent_directory, '{}.log'.format(name))


def summaries(parent_directory):
  """The path where tensorflow summaries are stored."""
  return os.path.join(parent_directory, 'summaries')


def trial(parent_directory, trial_name):
  """The parent directory for a trial."""
  return os.path.join(parent_directory, 'trial{}'.format(trial_name))


def run(parent_directory,
        level,
        experiment_name,
        run_id=''):
  """The name for a particular training run.

  Args:
    parent_directory: The directory in which this directory should be created.
    level: The number of pruning iterations.
    experiment_name: The name of this specific experiment.
    run_id: (optional) The number of this run (if the same experiment is being
      run more than once).

  Returns:
    The path in which data about this run should be stored.
  """
  return os.path.join(parent_directory, str(level),
                      '{}{}'.format(experiment_name, run_id))
