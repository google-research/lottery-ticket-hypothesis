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

"""The MNIST dataset."""

import keras
from lottery_ticket.foundations import dataset_base
from lottery_ticket.foundations import save_restore
import numpy as np


class DatasetMnist(dataset_base.DatasetBase):
  """The MNIST dataset."""

  def __init__(self,
               mnist_location,
               flatten=True,
               permute_labels=False,
               train_order_seed=None):
    """Create an MNIST dataset object.

    Args:
      mnist_location: The directory that contains MNIST as four npy files.
      flatten: Whether to convert the 28x28 MNIST images into a 1-dimensional
        vector with 784 entries.
      permute_labels: Whether to randomly permute the labels.
      train_order_seed: (optional) The random seed for shuffling the training
        set.
    """
    mnist = save_restore.restore_network(mnist_location)

    x_train = mnist['x_train']
    x_test = mnist['x_test']
    y_train = mnist['y_train']
    y_test = mnist['y_test']

    if permute_labels:
      # Reassign labels according to a random permutation of the labels.
      permutation = np.random.permutation(10)
      y_train = permutation[y_train]
      y_test = permutation[y_test]

    # Flatten x_train and x_test.
    if flatten:
      x_train = x_train.reshape((x_train.shape[0], -1))
      x_test = x_test.reshape((x_test.shape[0], -1))

    # Normalize x_train and x_test.
    x_train = keras.utils.normalize(x_train).astype(np.float32)
    x_test = keras.utils.normalize(x_test).astype(np.float32)

    # Convert y_train and y_test to one-hot.
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # Prepare the dataset.
    super(DatasetMnist, self).__init__(
        (x_train, y_train),
        64, (x_test, y_test),
        train_order_seed=train_order_seed)
