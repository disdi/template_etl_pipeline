import sys

from sandvik_nlp.tools import validate_options
from sandvik_nlp.tools import LoggingOptions

from sandvik_nlp.options import TemplateOptions
from sandvik_nlp import pipeline


def run(args=None):
    options = validate_options(args=args, option_classes=[LoggingOptions, TemplateOptions])

    options.view_as(LoggingOptions).configure_logging()

    return pipeline.run(options)


if __name__ == '__main__':
    sys.exit(run(args=sys.argv))
