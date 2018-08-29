import ujson
import apache_beam as beam
from apache_beam import typehints
from apache_beam import PTransform


class JSONDictCoder(beam.coders.Coder):
    """A coder used for reading and writing json"""

    def encode(self, value):
        return ujson.dumps(value)

    def decode(self, value):
        return JSONDict(ujson.loads(value))

    def is_deterministic(self):
        return True


class JSONDict(dict):
    """A single AIS/VMS message stored as a python dict with string keys
    We need to make this a class so taht we can properly use the type coding system
    in Beam to JSON-encode messages as we move them around.
    """
    pass

beam.coders.registry.register_coder(JSONDict, JSONDictCoder)


class ReadAsJSONDict(PTransform):
    """
    We need this because providing a custom coder to BigQuerySource does not seem to work, and we need the
    output from that source to be a JSONDict so that our JSONCoder will get used to serialize the rows
    when we pass data from one node to another in Beam (because json is faster and better then pickle)
    """

    def __init__(self, source):
        self.source = source

    def expand(self, p):
        return (
            p | beam.io.Read(self.source)
            | beam.ParDo(JSONDictDoFn())
        )


@typehints.with_input_types(typehints.Dict)
@typehints.with_output_types(JSONDict)
class JSONDictDoFn(beam.DoFn):
    """converts a dict to a JSONDict"""

    def process(self, d):
        yield JSONDict(d)