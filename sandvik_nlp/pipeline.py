import logging
import ujson

import apache_beam as beam
from apache_beam.runners import PipelineState

from sandvik_nlp.tools import MessageGenerator
from sandvik_nlp.tools import GenerateMessages
from sandvik_nlp.tools import JSONDictCoder

from sandvik_nlp.options import TemplateOptions
from sandvik_nlp.transform import AddField


def run(options):

    template_options = options.view_as(TemplateOptions)

    source = GenerateMessages(generator=MessageGenerator())
    sink = beam.io.WriteToText(file_path_prefix=template_options.dest, coder=JSONDictCoder())

    pipeline = beam.Pipeline(options=options)
    (
        pipeline
        | "generate" >> source
        | "tag" >> beam.ParDo(AddField(field=template_options.tag_field,
                                       value=template_options.tag_value))
        | "write" >> sink
    )

    result = pipeline.run()
    success_states = set([PipelineState.DONE])

    if template_options.wait:
        result.wait_until_finish()
    else:
        success_states.add(PipelineState.RUNNING)
        success_states.add(PipelineState.UNKNOWN)

    logging.info('returning with result.state=%s' % result.state)
    return 0 if result.state in success_states else 1
