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

"""Root locations for fully-connected MNIST experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# MNIST is stored as a directory containing four npy files:
#   x_train.npy, x_test.npy, y_train.npy, y_test.npy
# See datasets/dataset_mnist.py for details.

# Originally from https://s3.amazonaws.com/img-datasets/mnist.npz
MNIST_LOCATION = 'data/mnist'

EXPERIMENT_PATH = 'mnist_fc_data'
