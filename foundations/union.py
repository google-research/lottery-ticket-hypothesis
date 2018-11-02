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

"""Takes the union or intersection of masks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


def union(*masks):
  """Return new masks that are the per-layer union of the provided masks.

  Args:
    *masks: The set of masks to union.
  Returns:
    The union of the provided masks.
  """
  result = {}
  for mask in masks:
    for layer, values in six.iteritems(mask):
      if layer in result:
        result[layer] = result[layer] + values - result[layer] * values
      else:
        result[layer] = values
  return result


def intersect(*masks):
  """Return new masks that are the per-layer intersection of the provided masks.

  Args:
    *masks: The set of masks to intersect.
  Returns:
    The intersection of the provided masks.
  """
  result = {}
  for mask in masks:
    for layer, values in six.iteritems(mask):
      if layer in result:
        result[layer] *= values
      else:
        result[layer] = values
  return result
