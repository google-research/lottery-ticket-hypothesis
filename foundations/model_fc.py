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

"""A fully-connected neural network model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lottery_ticket.foundations import model_base
import tensorflow as tf


class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""

  def __init__(self,
               hyperparameters,
               input_placeholder,
               label_placeholder,
               presets=None,
               masks=None):
    """Creates a fully-connected network.

    Args:
      hyperparameters: A dictionary of hyperparameters for the network.
        For this class, a single hyperparameter is available: 'layers'. This
        key's value is a list of (# of units, activation function) tuples
        for each layer in order from input to output. If the activation
        function is None, then no activation will be used.
      input_placeholder: A placeholder for the network's input.
      label_placeholder: A placeholder for the network's expected output.
      presets: Preset initializations for the network as in model_base.py
      masks: Masks to prune the network as in model_base.py.
    """
    # Call parent constructor.
    super(ModelFc, self).__init__(presets=presets, masks=masks)

    # Build the network layer by layer.
    current_layer = input_placeholder
    for i, (units, activation) in enumerate(hyperparameters['layers']):
      current_layer = self.dense_layer(
          'layer{}'.format(i),
          current_layer,
          units,
          activation,
          kernel_initializer=tf.contrib.layers.xavier_initializer(
              uniform=False))

    # Compute the loss and accuracy.
    self.create_loss_and_accuracy(label_placeholder, current_layer)
