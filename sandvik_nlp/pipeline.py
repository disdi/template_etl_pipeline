import logging
import ujson

import apache_beam as beam
from apache_beam.runners import PipelineState

from sandvik_nlp.tools import MessageGenerator
from sandvik_nlp.tools import GenerateMessages
from sandvik_nlp.tools import JSONDictCoder

from sandvik_nlp.options import TemplateOptions
from sandvik_nlp.transform import AddField

from sandvik_nlp.beam_utils import NoopCoder
from sandvik_nlp.beam_utils import CsvFileSource
from apache_beam.io.filesystem import CompressionTypes
import pandas as pd

class Train(beam.DoFn):
    def process(self, element):
        #print(element['sepal_width'])
        df = pd.DataFrame([element], columns=element.keys())
        print df

def run(options):

    template_options = options.view_as(TemplateOptions)


    source = beam.io.Read(CsvFileSource('/opt/project/sandvik_nlp/iris/iris.csv'))
    sink = beam.io.WriteToText(file_path_prefix=template_options.dest, coder=NoopCoder())

    pipeline = beam.Pipeline(options=options)
    (
        pipeline
        | "generate" >> source
        | 'Train Values' >> beam.ParDo(Train())
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

