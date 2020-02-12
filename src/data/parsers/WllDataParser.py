import pandas, os, numpy, sys

class WllDataParser:

    def __init__(self, datadir, fname, features):
        '''

        Input:
            datadir - str path to the directory containing all data files.
            fname - str file name of the data file of interest.
            features - list of str feature names.
        
        '''
        self.fname = os.path.join(datadir, fname)
        self.feats = features

    def parse(self):
        '''

        Output:
            X - numpy array of shape (N, D), the features.
            Y - numpy array of shape (N, *), the target outputs.
            U - numpy.array of shape (N), unique subject id per sample.

        '''
        sys.stderr.write("Loading: %s\n" % self.fname)
        df = pandas.read_csv(self.fname)
        X = df[self.feats].values
        Y = self.extract_Y(df)
        U = df.original_subject_id.values
        return self._filter(X, Y, U)

    # === PROTECTED ===

    def extract_Y(self, df):
        '''

        Input:
            df - pandas DataFrame, entire csv data.

        Output:
            Y - numpy array of size (N, *), the output targets only.

        '''
        raise NotImplementedError

    # === PRIVATE ===

    def _filter(self, X, Y, U):
        I = numpy.isfinite(Y)
        return X[I], Y[I], U[I]
