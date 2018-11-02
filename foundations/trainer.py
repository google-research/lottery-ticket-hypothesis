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

"""A function that trains a network on a dataset."""

from lottery_ticket.foundations import paths
from lottery_ticket.foundations import save_restore
import tensorflow as tf


def train(sess, dataset, model, optimizer_fn, training_len, output_dir,
          **params):
  """Train a model on a dataset.

  Training continues until training_len iterations or epochs have taken place.

  Args:
    sess: A tensorflow session
    dataset: The dataset on which to train (a child of dataset_base.DatasetBase)
    model: The model to train (a child of model_base.ModelBase)
    optimizer_fn: A function that, when called, returns an instance of an
      optimizer object to be used to optimize the network.
    training_len: A tuple whose first value is the unit of measure
      ("epochs" or "iterations") and whose second value is the number of
      units for which the network should be trained.
    output_dir: The directory to which any output should be saved.
    **params: Other parameters.
      save_summaries is whether to save summary data.
      save_network is whether to save the network before and after training.
      test_interval is None if the test set should not be evaluated; otherwise,
        frequency (in iterations) at which the test set should be run.
      validate_interval is analogous to test_interval.

  Returns:
      A dictionary containing the weights before training and the weights after
      training.
  """
  # Create initial session parameters.
  optimize = optimizer_fn().minimize(model.loss)
  sess.run(tf.global_variables_initializer())
  initial_weights = model.get_current_weights(sess)

  train_handle = dataset.get_train_handle(sess)
  test_handle = dataset.get_test_handle(sess)
  validate_handle = dataset.get_validate_handle(sess)

  # Optional operations to perform before training.
  if params.get('save_summaries', False):
    writer = tf.summary.FileWriter(paths.summaries(output_dir))
    train_file = tf.gfile.GFile(paths.log(output_dir, 'train'), 'w')
    test_file = tf.gfile.GFile(paths.log(output_dir, 'test'), 'w')
    validate_file = tf.gfile.GFile(paths.log(output_dir, 'validate'), 'w')

  if params.get('save_network', False):
    save_restore.save_network(paths.initial(output_dir), initial_weights)
    save_restore.save_network(paths.masks(output_dir), model.masks)

  # Helper functions to collect and record summaries.
  def record_summaries(iteration, records, fp):
    """Records summaries obtained from evaluating the network.

    Args:
      iteration: The current training iteration as an integer.
      records: A list of records to be written.
      fp: A file to which the records should be logged in an easier-to-parse
        format than the tensorflow summary files.
    """
    if params.get('save_summaries', False):
      log = ['iteration', str(iteration)]
      for record in records:
        # Log to tensorflow summaries for tensorboard.
        writer.add_summary(record, iteration)

        # Log to text file for convenience.
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(record)
        value = summary_proto.value[0]
        log += [value.tag, str(value.simple_value)]
      fp.write(','.join(log) + '\n')

  def collect_test_summaries(iteration):
    if (params.get('save_summaries', False) and
        'test_interval' in params and
        iteration % params['test_interval'] == 0):
      sess.run(dataset.test_initializer)
      records = sess.run(model.test_summaries, {dataset.handle: test_handle})
      record_summaries(iteration, records, test_file)

  def collect_validate_summaries(iteration):
    if (params.get('save_summaries', False) and
        'validate_interval' in params and
        iteration % params['validate_interval'] == 0):
      sess.run(dataset.validate_initializer)
      records = sess.run(model.validate_summaries,
                         {dataset.handle: validate_handle})
      record_summaries(iteration, records, validate_file)

  # Train for the specified number of epochs. This behavior is encapsulated
  # in a function so that it is possible to break out of multiple loops
  # simultaneously.
  def training_loop():
    """The main training loop encapsulated in a function."""
    iteration = 0
    epoch = 0
    while True:
      sess.run(dataset.train_initializer)
      epoch += 1

      # End training if we have passed the epoch limit.
      if training_len[0] == 'epochs' and epoch > training_len[1]:
        return

      # One training epoch.
      while True:
        try:
          iteration += 1

          # End training if we have passed the iteration limit.
          if training_len[0] == 'iterations' and iteration > training_len[1]:
            return

          # Train.
          records = sess.run([optimize] + model.train_summaries,
                             {dataset.handle: train_handle})[1:]
          record_summaries(iteration, records, train_file)

          # Collect test and validation data if applicable.
          collect_test_summaries(iteration)
          collect_validate_summaries(iteration)

        # End of epoch handling.
        except tf.errors.OutOfRangeError:
          break

  # Run the training loop.
  training_loop()

  # Clean up.
  if params.get('save_summaries', False):
    train_file.close()
    test_file.close()
    validate_file.close()

  # Retrieve the final weights of the model.
  final_weights = model.get_current_weights(sess)
  if params.get('save_network', False):
    save_restore.save_network(paths.final(output_dir), final_weights)

  return initial_weights, final_weights
