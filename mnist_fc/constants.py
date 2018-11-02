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

"""Constants for fully-connected MNIST experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from lottery_ticket.foundations import paths
from lottery_ticket.mnist_fc import locations
import tensorflow as tf

HYPERPARAMETERS = {'layers': [(300, tf.nn.relu), (100, tf.nn.relu), (10, None)]}

MNIST_LOCATION = locations.MNIST_LOCATION

FASHIONMNIST_LOCATION = locations.FASHIONMNIST_LOCATION

OPTIMIZER_FN = functools.partial(tf.train.GradientDescentOptimizer, .1)

PRUNE_PERCENTS = {'layer0': .2, 'layer1': .2, 'layer2': .1}

TRAINING_LEN = ('iterations', 50000)

EXPERIMENT_PATH = locations.EXPERIMENT_PATH


def graph(category, filename):
  return os.path.join(EXPERIMENT_PATH, 'graphs', category, filename)


def initialization(level):
  return os.path.join(EXPERIMENT_PATH, 'weights', str(level), 'initialization')


def trial(trial_name):
  return paths.trial(EXPERIMENT_PATH, trial_name)


def run(trial_name, level, experiment_name='same_init', run_id=''):
  return paths.run(trial(trial_name), level, experiment_name, run_id)
