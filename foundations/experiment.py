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

"""Run the lottery ticket experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def experiment(make_dataset, make_model, train_model, prune_masks, iterations,
               presets=None):
  """Run the lottery ticket experiment for the specified number of iterations.

  Args:
    make_dataset: A function that, when called with no arguments, will create an
      object that descends from dataset_base.
    make_model: A function that, when called with four arguments (input_tensor,
      label_tensor, presets, masks), creates a model object that descends from
      model_base. Presets and masks are optional.
    train_model: A function that, when called with four arguments (session,
      pruning iteration number, dataset, model), trains the model using the
      dataset and returns the model's initial and final weights as dictionaries.
    prune_masks: A function that, when called with two arguments (dictionary of
      current masks, dictionary of final weights), returns a new dictionary of
      masks that have been pruned. Each dictionary key is the name of a tensor
      in the network; each value is a numpy array containing the values of the
      tensor (1/0 values for mask, weights for the dictionary of final weights).
    iterations: The number of pruning iterations to perform.
    presets: (optional) The presets to use for the first iteration of training.
      In the form of a dictionary where each key is the name of a tensor and
      each value is a numpy array of the values to which that tensor should
      be initialized.
  """

  # A helper function that trains the network once according to the behavior
  # determined internally by the train_model function.
  def train_once(iteration, presets=None, masks=None):
    tf.reset_default_graph()
    sess = tf.Session()
    dataset = make_dataset()
    input_tensor, label_tensor = dataset.placeholders
    model = make_model(input_tensor, label_tensor, presets=presets, masks=masks)
    return train_model(sess, iteration, dataset, model)

  # Run once normally.
  initial, final = train_once(0, presets=presets)

  # Create the initial masks with no weights pruned.
  masks = {}
  for k, v in initial.items():
    masks[k] = np.ones(v.shape)

  # Begin the training loop.
  for iteration in range(1, iterations + 1):
    # Prune the network.
    masks = prune_masks(masks, final)

    # Train the network again.
    _, final = train_once(iteration, presets=initial, masks=masks)
