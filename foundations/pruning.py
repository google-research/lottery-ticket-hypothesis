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

"""A collection of pruning heuristics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def prune_by_percent(percents, masks, final_weights):
  """Return new masks that involve pruning the smallest of the final weights.

  Args:
    percents: A dictionary determining the percent by which to prune each layer.
      Keys are layer names and values are floats between 0 and 1 (inclusive).
    masks: A dictionary containing the current masks. Keys are strings and
      values are numpy arrays with values in {0, 1}.
    final_weights: The weights at the end of the last training run. A
      dictionary whose keys are strings and whose values are numpy arrays.

  Returns:
    A dictionary containing the newly-pruned masks.
  """

  def prune_by_percent_once(percent, mask, final_weight):
    # Put the weights that aren't masked out in sorted order.
    sorted_weights = np.sort(np.abs(final_weight[mask == 1]))

    # Determine the cutoff for weights to be pruned.
    cutoff_index = np.round(percent * sorted_weights.size).astype(int)
    cutoff = sorted_weights[cutoff_index]

    # Prune all weights below the cutoff.
    return np.where(np.abs(final_weight) <= cutoff, np.zeros(mask.shape), mask)

  new_masks = {}
  for k, percent in percents.items():
    new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k])

  return new_masks
