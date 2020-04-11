"""
"""
from halotools.utils import crossmatch
from astropy.table import Table
import numpy as np
import os

_dirname = "/Users/aphearin/work/DATA/MOCKS/UniverseMachine/paper1_published_mocks"
_basename = "sfr_catalog_1.002310_logmp_gt_10p5.npy"
FNAME = os.path.join(_dirname, _basename)


def load_umachine_z0(fname=FNAME, ssfr_q_loc=-11.8, ssfr_q_scale=0.5, seed=43):
    """These data can be downloaded from """
    mock = Table(np.load(fname))

    cenmask = mock["upid"] == -1
    mock["hostid"] = mock["upid"]
    mock["hostid"][cenmask] = mock["id"][cenmask]

    mhost = np.copy(mock["mp"])
    idxA, idxB = crossmatch(mock["upid"][~cenmask], mock["id"])
    mhost_sats = mhost[~cenmask]
    mhost_sats[idxA] = mock["mp"][idxB]

    mhost[~cenmask] = mhost_sats
    mock["logmhost"] = np.log10(mhost)
    mask = mock["sm"] > 10 ** 9.5
    mock = mock[mask]
    mock["logsm"] = np.log10(mock["sm"])
    mock["ssfr"] = mock["sfr"] / mock["sm"]
    mock.remove_column("sm")

    zeromask = mock["ssfr"] == 0
    nzero = np.count_nonzero(zeromask)
    rng = np.random.RandomState(seed)
    random_ssfr_quenched_gals = rng.normal(
        loc=ssfr_q_loc, scale=ssfr_q_scale, size=nzero
    )
    mock["ssfr"][zeromask] = 10 ** random_ssfr_quenched_gals
    mock["log_ssfr"] = np.log10(mock["ssfr"])
    mock.remove_column("ssfr")

    mock["x"] = np.mod(mock["pos"][:, 0], 250)
    mock["y"] = np.mod(mock["pos"][:, 1], 250)
    mock["z"] = np.mod(mock["pos"][:, 2], 250)
    cols_to_remove = [
        "flags",
        "uparent_dist",
        "pos",
        "vmp",
        "lvmp",
        "m",
        "descid",
        "v",
        "r",
        "rank1",
        "rank2",
        "ra",
        "rarank",
        "A_UV",
        "icl",
        "obs_sm",
        "obs_sfr",
        "obs_uv",
        "empty",
    ]
    for col in cols_to_remove:
        mock.remove_column(col)
    mock["logmpeak"] = np.log10(mock["mp"])
    mock.remove_column("mp")

    mock["neg_logmhost"] = -mock["logmhost"]
    mock.sort(("neg_logmhost", "hostid", "upid", "logmpeak"))
    mock.remove_column("neg_logmhost")

    return mock
