import argparse
import sys

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.options.pipeline_options import SetupOptions

from sandvik_nlp.tools import flatten


def validate_options(args=None, option_classes=None):

    args = args or sys.argv
    option_classes = flatten(option_classes)

    help_flags = ['-h', '--help']
    help = any(flag in help_flags for flag in args)

    # first check to see if we are using the DirectRunner or the DataflowRunner
    # need to strip out any help params so that we don't exit too early
    nohelp_args = [arg for arg in sys.argv if arg not in help_flags]
    # Parse args just for StandardOptions and see which runner we are using
    local = StandardOptions(nohelp_args).runner in (None, 'DirectRunner')

    # make a new parser
    parser = argparse.ArgumentParser()

    # add args for all the options classes that we are using
    for opt in option_classes:
        opt._add_argparse_args(parser)
    StandardOptions._add_argparse_args(parser.add_argument_group('Dataflow Runner'))

    if help or not local:
        GoogleCloudOptions._add_argparse_args(parser.add_argument_group('Dataflow Runtime'))
        WorkerOptions._add_argparse_args(parser.add_argument_group('Dataflow Workers'))
        SetupOptions._add_argparse_args(parser.add_argument_group('Dataflow Setup'))

    # parse all args and trigger help if any required args are missing
    parser.parse_known_args(args)

    return PipelineOptions(args)
