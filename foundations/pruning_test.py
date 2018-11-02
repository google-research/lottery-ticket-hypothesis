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

"""Tests for pruning.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lottery_ticket.foundations import pruning
import numpy as np
import tensorflow as tf


class PruningTest(tf.test.TestCase):

  def test_union(self):
    masks1 = {'layer0': np.array([[1, 0], [0, 1]])}
    masks2 = {'layer0': np.array([[0, 1], [0, 1]])}
    union_mask = pruning.union(masks1, masks2)
    self.assertTrue((union_mask['layer0'] == np.array([[1, 1], [0, 1]])).all())

  def test_intersect(self):
    masks1 = {'layer0': np.array([[1, 0], [0, 1]])}
    masks2 = {'layer0': np.array([[0, 1], [0, 1]])}
    intersect_mask = pruning.intersect(masks1, masks2)
    self.assertTrue(
        (intersect_mask['layer0'] == np.array([[0, 0], [0, 1]])).all())


if __name__ == '__main__':
  tf.test.main()
