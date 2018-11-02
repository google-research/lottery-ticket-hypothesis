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

"""The lottery ticket experiment for Lenet 300-100 trained on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from lottery_ticket.datasets import dataset_mnist
from lottery_ticket.foundations import experiment
from lottery_ticket.foundations import model_fc
from lottery_ticket.foundations import paths
from lottery_ticket.foundations import pruning
from lottery_ticket.foundations import save_restore
from lottery_ticket.foundations import trainer
from lottery_ticket.mnist_fc import constants


def train(output_dir,
          mnist_location=constants.MNIST_LOCATION,
          training_len=constants.TRAINING_LEN,
          iterations=30,
          experiment_name='same_init',
          presets=None,
          permute_labels=False,
          train_order_seed=None):
  """Perform the lottery ticket experiment.

  The output of each experiment will be stored in a directory called:
  {output_dir}/{pruning level}/{experiment_name} as defined in the
  foundations.paths module.

  Args:
    output_dir: Parent directory for all output files.
    mnist_location: The path to the NPZ file containing MNIST.
    training_len: How long to train on each iteration.
    iterations: How many iterative pruning steps to perform.
    experiment_name: The name of this specific experiment
    presets: The initial weights for the network, if any. Presets can come in
      one of three forms:
      * A dictionary of numpy arrays. Each dictionary key is the name of the
        corresponding tensor that is to be initialized. Each value is a numpy
        array containing the initializations.
      * The string name of a directory containing one file for each
        set of weights that is to be initialized (in the form of
        foundations.save_restore).
      * None, meaning the network should be randomly initialized.
    permute_labels: Whether to permute the labels on the dataset.
    train_order_seed: The random seed, if any, to be used to determine the
      order in which training examples are shuffled before being presented
      to the network.
  """
  # Define model and dataset functions.
  def make_dataset():
    return dataset_mnist.DatasetMnist(
        mnist_location,
        permute_labels=permute_labels,
        train_order_seed=train_order_seed)

  make_model = functools.partial(model_fc.ModelFc, constants.HYPERPARAMETERS)

  # Define a training function.
  def train_model(sess, level, dataset, model):
    params = {
        'test_interval': 100,
        'save_summaries': True,
        'save_network': True,
    }

    return trainer.train(
        sess,
        dataset,
        model,
        constants.OPTIMIZER_FN,
        training_len,
        output_dir=paths.run(output_dir, level, experiment_name),
        **params)

  # Define a pruning function.
  prune_masks = functools.partial(pruning.prune_by_percent,
                                  constants.PRUNE_PERCENTS)

  # Run the experiment.
  experiment.experiment(
      make_dataset,
      make_model,
      train_model,
      prune_masks,
      iterations,
      presets=save_restore.standardize(presets))
