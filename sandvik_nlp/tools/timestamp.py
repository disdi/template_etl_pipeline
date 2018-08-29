from datetime import datetime
import udatetime
import pytz
from sandvik_nlp.tools import JSONDict

EPOCH = udatetime.utcfromtimestamp(0)

def timestampFromDatetime(dt):
    """Convert a datetime to a unix timestamp """
    return (dt - EPOCH).total_seconds()

