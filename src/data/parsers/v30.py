import json, os, numpy

from .WllDataParser import WllDataParser

def parse(labelmap, *args, **kwargs):
    '''

    Input:
        labelmap - dict mapping str label to float target output or None.
            All rows with target outputs of None are removed.
        *args, **kwargs - see WllDataParser.__init__

    Output:
        (see WllDataParser.parse)

    '''
    # Here we assume healthy aging is the only dataset with one label.
    Parser = [V30, V30_healthyaging][len(labelmap) == 1]
    return Parser(labelmap, *args, **kwargs).parse()

# === PRIVATE ===

NAME = "name"

class V30(WllDataParser):

    def __init__(self, labelmap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labelmap = labelmap

    def extract_Y(self, df):
        '''

        Input:
            df - pandas DataFrame, entire csv data.

        Output:
            Y - numpy array of size (N), the diagnosis value.

        '''
        ys = [json.loads(s) for s in df.diagnoses]
        assert all(map(self._check, ys))
        dg = [o[0][NAME] for o in ys]
        it = [self.labelmap[s] for s in dg]
        return numpy.array(it, dtype=numpy.float32)

    def _check(self, y):
        return len(y) == 1 and NAME in y[0]

class V30_healthyaging(V30):

    def extract_Y(self, df):
        '''

        Input:
            df - pandas DataFrame, entire csv data.

        Output:
            Y - numpy array of size (N), the diagnosis value.

        '''
        assert len(self.labelmap) == 1
        return numpy.zeros(df.shape[0]) + self.labelmap["hc"]
        
