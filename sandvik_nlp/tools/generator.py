from datetime import datetime
import pytz

import apache_beam as beam
from apache_beam import typehints

from sandvik_nlp.tools import JSONDict
from sandvik_nlp.tools import timestampFromDatetime

DEFAULT_START_TS = timestampFromDatetime(datetime(2017, 1, 1, 0, 0, 0, tzinfo=pytz.UTC))
HOUR_IN_SECONDS = 60 * 60


class MessageGenerator():
    def __init__(self, start_ts=DEFAULT_START_TS,
                 increment=HOUR_IN_SECONDS, count=72):
        self.start_ts = start_ts
        self.increment = increment
        self.count = count

    def __iter__(self):
        return self.messages()

    def messages(self):
        ts = self.start_ts
        for idx in xrange(self.count):
            yield JSONDict(mmsi=1, timestamp=ts, idx=idx)
            ts += self.increment

    def bigquery_schema(self):
        return "mmsi:INTEGER,timestamp:TIMESTAMP,idx:INTEGER"


@typehints.with_output_types(JSONDict)
class GenerateMessages(beam.PTransform):
    """generate simulated AIS messages for testing"""

    def __init__(self, generator=MessageGenerator(), **kwargs):
        self.messages = generator
        super(GenerateMessages,self).__init__(**kwargs)

    def expand(self, pcoll):
        return pcoll | beam.Create(self.messages)
