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

"""Run this script to download MNIST and FashionMNIST datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fire
from keras.datasets import mnist
from lottery_ticket.foundations import save_restore
from lottery_ticket.mnist_fc import locations


def download(location=locations.MNIST_LOCATION):
  d = {}
  (d['x_train'], d['y_train']), (d['x_test'], d['y_test']) = mnist.load_data()
  save_restore.save_network(location, d)


def main(unused_argv):
  fire.Fire(download)

if __name__ == '__main__':
  main(0)
