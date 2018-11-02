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

"""Generates the argfile for the fashionmnist experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from lottery_ticket.foundations import paths
from lottery_ticket.mnist_fc import constants

FLAGS = flags.FLAGS

flags.DEFINE_enum('experiment', 'reinit', ['reinit', 'reuse', 'reuse_sign'],
                  'Which experiment to run.')


def main(argv):
  del argv  # Unused.
  line_format = ('--masks={masks} --output_dir={output_dir}')
  name = FLAGS.experiment

  for trial in range(1, 21):
    for level in range(0, 31):
      for run in range(1, 11):
        masks = paths.masks(constants.run(trial, level))
        output = constants.run(trial, level, name, run)
        result = line_format.format(masks=masks, output_dir=output)

        if FLAGS.experiment in ('reuse', 'reuse_sign'):
          result += (' --initialization_distribution=' +
                     constants.initialization(level))

        if FLAGS.experiment == 'reuse_sign':
          presets = paths.initial(constants.run(trial, level))
          result += ' --same_sign={}'.format(presets)

        print(result)


if __name__ == '__main__':
  app.run(main)
