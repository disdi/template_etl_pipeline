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
import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import pickle

class Train(beam.DoFn):
    def process(self, element):
        #print(element['sepal_width'])
        df = pd.DataFrame([element], columns=element.keys())
        #print df
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = df[feature_cols]
        target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        species_map = {k:i for i, k in enumerate(target_names)}
        df['species'] = df['species'].map(species_map)
        y = df.species

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        with open('filename.pkl', 'wb') as f:
              pickle.dump(clf, f)

class PredictDoFn(beam.DoFn):

  def process(self, element):
    with open('filename.pkl', 'rb') as f:
          clf = pickle.load(f)
    df = pd.DataFrame([element], columns=element.keys())
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]

    result = clf.predict(X)
    return result

def run(options):

    template_options = options.view_as(TemplateOptions)


    source_train = beam.io.Read(CsvFileSource('/opt/project/sandvik_nlp/iris/iris_train.csv'))
    source_test = beam.io.Read(CsvFileSource('/opt/project/sandvik_nlp/iris/iris_test.csv'))
    sink = beam.io.WriteToText(file_path_prefix=template_options.dest, coder=NoopCoder())

    p = beam.Pipeline(options=options)
    data_process = (
           p
           | 'generate' >> source_train
           | 'train'>> beam.ParDo(Train())
    )
    predictions = (
           p
           | 'test' >> source_test
           | 'Prediction' >> beam.ParDo(PredictDoFn())
    )
    predictions | 'Write' >> sink

    result = p.run()
    success_states = set([PipelineState.DONE])

    if template_options.wait:
        result.wait_until_finish()
    else:
        success_states.add(PipelineState.RUNNING)
        success_states.add(PipelineState.UNKNOWN)

    logging.info('returning with result.state=%s' % result.state)
    return 0 if result.state in success_states else 1

