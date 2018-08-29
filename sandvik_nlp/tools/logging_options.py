from __future__ import absolute_import

import logging
import sys

from apache_beam.options.pipeline_options import PipelineOptions


class LoggingOptions(PipelineOptions):
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }

    DEFAULT_LOG_LEVEL = 'INFO'

    @classmethod
    def _add_argparse_args(cls, parser):
        # Use add_value_provider_argument for arguments to be templatable
        # Use add_argument as usual for non-templatable arguments

        parser.add_argument(
            '--log_file',
            help='file to send logging output to')

        parser.add_argument(
            '--log_level',
            default=cls.DEFAULT_LOG_LEVEL,
            choices=list(cls.log_levels.keys()),
            help='logging level (default: %(default)s)')

        parser.add_argument(
            '--log_args',
            default=False,
            action='store_true',
            help='Output command line arguments to logging (useful for debugging then you cant get stdout')

    def configure_logging(self):
        args = self._flags or sys.argv
        logger = logging.getLogger()
        if self.log_file:
            logger.addHandler(logging.FileHandler(self.log_file))
        else:
            logging.basicConfig()
        logger.setLevel(self.log_levels[self.log_level])
        if self.log_args:
            logging.info('Running with these command line params')
            for arg in args:
                logging.info('   %s' % arg)
