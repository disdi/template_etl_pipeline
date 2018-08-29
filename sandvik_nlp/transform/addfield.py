import ujson
import six

import apache_beam as beam

class AddField(beam.DoFn):
    def __init__(self, field, value):
        self.field = field
        self.value = value


    def process(self, element):
        element[self.field] = self.value
        yield element
