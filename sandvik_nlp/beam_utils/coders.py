import apache_beam as beam

class NoopCoder(beam.coders.Coder):
  """ Implements coder interface. Returns data as is.
  """
  def encode(self, record):
    return record

  def decode(self, record):
    return record
