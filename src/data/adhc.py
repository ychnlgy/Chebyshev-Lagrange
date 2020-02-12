from . import parsers

def get(datadir):
    features = parsers.featurelist.get(datadir)

    X_db_am, Y_db_am, U_db_am = parsers.v34.parse(
        labelmap = {
            "ad": 1,
            "mci": None,
            "dementia_vascular": None,
            "unknown": None,
            "hc": 0
        },
        datadir = datadir,
        fname = "db-34-am.csv",
        features = features
    )

    X_ha_am, Y_ha_am, U_ha_am = parsers.v34.parse(
        labelmap = {
            "hc": 0
        },
        datadir = datadir,
        fname = "ha-34-am.csv",
        features = features
    )

    X_ha_ca, Y_ha_ca, U_ha_ca = parsers.v34.parse(
        labelmap = {
            "hc": 0
        },
        datadir = datadir,
        fname = "ha-34-ca.csv",
        features = features
    )

    S_uf_am, X_uf_am, Y_uf_am, U_uf_am = parsers.v34_fp.parse(
        tminus_milestones = [20],
        healthy_diag_year = 9999,
        datadir = datadir,
        fname = "uf-34-am.csv",
        features = features
    )

    return (
        (X_db_am, Y_db_am, U_db_am),
        (X_ha_am, Y_ha_am, U_ha_am),
        (X_ha_ca, Y_ha_ca, U_ha_ca),
        (X_uf_am, Y_uf_am, U_uf_am, S_uf_am)
    )
