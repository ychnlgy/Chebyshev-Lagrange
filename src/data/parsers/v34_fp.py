import numpy, json

from .WllDataParser import WllDataParser

def parse(tminus_milestones, healthy_diag_year, *args, **kwargs):
    '''

    Input:
        tminus_milestones - list of int tminus diagnosis year, sorted
            by descending order. For example [10, 0] means that samples
            that were taken 10 years prior to formal AD diagnosis are
            treated as healthy (i.e. value 0), samples within 10 years
            of the diagnosis year have value 0.5 and samples after
            the diagnosis year have value 0.
        healthy_diag_year - int age at which healthy people are diagnosed
            with AD. For example, 150, which is an age that is far out
            of the range of diagnosis years in the dataset.
        *args, **kwargs - see WllDataParser.__init__

    Output:
        S and (see WllDataParser.parse) - S is the session id of each
            sample, since each sample is 5-utterances extracted from
            the original full speech.

    '''
    parser = V34_FamousPeople(healthy_diag_year, *args, **kwargs)
    X, T, U = parser.parse()
    S = parser.session
    Y = _apply_milestones(tminus_milestones, T)
    return S, X, Y, U

# === PRIVATE ===

DIAG_YEAR = "diagnosis_year"

class V34_FamousPeople(WllDataParser):

    def __init__(self, healthy_diag, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.healthy_diag = healthy_diag
        self.session = None

    def extract_Y(self, df):
        '''

        Input:
            df - pandas DataFrame, entire csv data.

        Output:
            Y - numpy array of size (N), the difference between
                the year of diagnosis and the year of the sample.
                Negative values mean the subject is diagnosed already.

        '''
        self._store_session_and_subject(df)
        ys = [json.loads(s) for s in df.diagnoses]
        assert all(map(self._check, ys))
        dg = [o[0][DIAG_YEAR] for o in ys]
        dg = [d if d is not None else self.healthy_diag for d in dg]
        diag = numpy.array(dg, dtype=numpy.int64)
        curr = df.sample_created_year.values
        diff = diag - curr
        return diff

    def _store_session_and_subject(self, df):
        self.session = df.session_id.values

    def _check(self, y):
        return len(y) == 1 and DIAG_YEAR in y[0]

def _apply_milestones(milestones, tminus):
    '''

    Description:
        Transforms continuous tminus values into discrete categories
        within the range [0, 1].

    '''
    Y = numpy.zeros_like(tminus)

    Y[tminus > milestones[0]] = 0
    Y[tminus <= milestones[-1]] = 1
    
    sep_n = len(milestones)-1
    
    if sep_n > 0:
        d = 1.0/sep_n
    
        for i, h in enumerate(milestones[:-1], 1):
            l = milestones[i]
            Y[(tminus > l) & (tminus <= h)] = (i-0.5)*d
    
    return Y
