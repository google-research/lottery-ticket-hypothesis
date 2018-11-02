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

"""Save and restore networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from lottery_ticket.foundations import paths
import numpy as np
import six
import tensorflow as tf


def save_network(filename, weights_dict):
  """Save the parameters of a neural network.

  weights_dict is a dictionary where each key is the name of a tensor and each
  value is a numpy array containing that tensor's weights. filename is created
  as a directory and each item of weights_dict is saved as a separate npy file
  within that directory.

  This function is useful for saving the weights of a network, the
  initialization of a network (in the same manner), or the masks used to prune a
  network.

  Args:
    filename: A directory in which the network should be saved.
    weights_dict: A dictionary where each key is the name of a tensor and each
      value is a numpy array. This is the dictionary of values that is to be
      saved.
  """
  if tf.gfile.Exists(filename):
    tf.gfile.DeleteRecursively(filename)
  tf.gfile.MakeDirs(filename)

  for k, v in weights_dict.items():
    with tf.gfile.FastGFile(os.path.join(filename, k + '.npy'), 'w') as fp:
      np.save(fp, v)


def restore_network(filename):
  """Loads a network in the form stored by save_network.

  The inverse operation of save_network.

  filename is the name of a directory containing many npy files. Each npy file
  is loaded into a numpy array and added to a dictionary. The dictionary key
  is the name of the file (without the .npy extension). This dictionary is
  returned.

  Args:
    filename: The name of the directory where the npy files are saved.

  Returns:
    A dictionary where each key is the name of a npy file within filename and
    each value is the corresponding numpy array stored in that file. This
    dictionary is of the same form as that passed to save_network.

  Raises:
    ValueError: If filename does not exist.
  """
  if not tf.gfile.Exists(filename):
    raise ValueError('Filename {} does not exist.'.format(filename))

  weights_dict = {}

  for basename in tf.gfile.ListDirectory(filename):
    name = basename.split('.')[0]
    with tf.gfile.FastGFile(os.path.join(filename, basename)) as fp:
      weights_dict[name] = np.load(fp)

  return weights_dict


def standardize(network, combine_fn=None):
  """Restore a network that has been provided in one of four possible forms.

  A network can be represented in one of four forms:
    * None, the absence of a network.
    * A dictionary where keys are names of tensors and values are numpy arrays
      of the values to be stored in those tensors.
    * The name of a directory containing npy files. The filenames become
      dictionary keys and the file contents become dictionary values.
    * A list of directory names and dictionaries in one of the aforementioned
      forms. Any directories are restored into dictionaries, after which
      combine_fn is applied to the list of dictionaries to combine it into
      a single dictionary.

  Args:
    network: A reference to a network in one of the forms described above.
    combine_fn: The function used to combine a list of dictionaries into a
      single dictionary. This argument is only required if network could be
      a list.

  Returns:
    A dictionary whose keys are tensor names and whose values are numpy arrays.
    This dictionary was derived from the dictionary, location, or location_list
    arguments.

  Raises:
    ValueError: If the network is of an unexpected type.
  """
  if isinstance(network, dict) or network is None:
    return network
  elif isinstance(network, six.string_types):
    return restore_network(network)
  elif isinstance(network, list):
    return combine_fn([standardize(n) for n in network])
  else:
    raise ValueError('network must be a dict, string path, None, or a list '
                     ' of those types.')


def read_log(directory, name='test', tail=0):
  """Reads logged data about the performance of a lottery ticket experiment.

  Args:
    directory: The directory where the log data for a particular experiment
      is stored.
    name: Whether to retrieve data from the "test", "train", or "validate"
      logs.
    tail: If nonzero, returns only the last tail entries in each run.

  Returns:
    A dictionary with three keys.
    'iteration' is a numpy array of the iterations at which data was collected.
    'loss' is a numpy array of loss values at the corresponding iteration.
    'accuracy' is a numpy array of accuracy values at the corresponding
      iteration.
  """
  output = {
      'iteration': [],
      'loss': [],
      'accuracy': [],
  }

  with tf.gfile.GFile(paths.log(directory, name)) as fp:
    reader = csv.reader(fp)
    for row in reader:
      output['iteration'].append(float(row[1]))
      output['loss'].append(float(row[3]))
      output['accuracy'].append(float(row[5]))

  output['iteration'] = np.array(output['iteration'][-tail:])
  output['loss'] = np.array(output['loss'][-tail:])
  output['accuracy'] = np.array(output['accuracy'][-tail:])

  return output


def write_log(data, directory, name='test'):
  """Writes data about the performance of a lottery ticket experiment.

  Input data takes the same form as data returned by read_data. Writes a file
  in the format read by read_data.

  Args:
    data: The data to be written to the file. Takes the same form as the data
      returned by read_data.
    directory: The directory where the log data for a particular experiment is
      to be stored.
    name: What to call the data file itself.
  """
  with tf.gfile.GFile(paths.log(directory, name), 'w') as fp:
    for loss, it, acc in zip(data['loss'], data['iteration'], data['accuracy']):
      fp.write(','.join(
          ('iteration',
           str(it), 'loss',
           str(loss), 'accuracy',
           str(acc))))
      fp.write('\n')
