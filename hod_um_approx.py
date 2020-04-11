"""
"""
import numpy as np


def generate_um_hod_mock(upid, logsm, logmhalo, seed=43, satboost=False):
    """
    """
    ntot = upid.size

    cenmask = upid == -1

    qfrac_cens = quenched_fraction_centrals(logmhalo[cenmask])
    if satboost:
        qfrac_sats = cluster_boosted_quenched_fraction_satellites(
            logsm[~cenmask], logmhalo[~cenmask]
        )
    else:
        qfrac_sats = quenched_fraction_satellites(logsm[~cenmask], logmhalo[~cenmask])
    qfrac = np.zeros_like(logsm)
    qfrac[cenmask] = qfrac_cens
    qfrac[~cenmask] = qfrac_sats

    rng0 = np.random.RandomState(seed + 0)
    uran = rng0.uniform(0, 1, ntot)
    qmask = uran < qfrac

    loc_qcens = quenched_median_logssfr(logsm[qmask & cenmask])
    loc_qsats = quenched_median_logssfr(logsm[qmask & ~cenmask])
    loc_sfcens = main_sequence_median_logssfr(logsm[~qmask & cenmask])
    loc_sfsats = main_sequence_median_logssfr(logsm[~qmask & ~cenmask])

    rng1 = np.random.RandomState(seed + 1)
    ssfr_qcens = rng1.normal(loc=loc_qcens, scale=0.5)

    rng2 = np.random.RandomState(seed + 2)
    ssfr_qsats = rng2.normal(loc=loc_qsats, scale=0.5)

    rng3 = np.random.RandomState(seed + 3)
    ssfr_sfcens = rng3.normal(loc=loc_sfcens, scale=0.35)

    rng4 = np.random.RandomState(seed + 4)
    ssfr_sfsats = rng4.normal(loc=loc_sfsats, scale=0.35)

    log_ssfr_new = np.zeros(ntot)
    log_ssfr_new[qmask & cenmask] = ssfr_qcens
    log_ssfr_new[qmask & ~cenmask] = ssfr_qsats
    log_ssfr_new[~qmask & cenmask] = ssfr_sfcens
    log_ssfr_new[~qmask & ~cenmask] = ssfr_sfsats
    return log_ssfr_new


def main_sequence_median_logssfr(logsm, x0=10, y0=-10.15, m=-0.15):
    return y0 + m * (logsm - x0)


def quenched_median_logssfr(logsm):
    return np.zeros_like(logsm) - 11.9


def quenched_fraction_centrals(
    logmpeak, fqcens_x0=12.25, fqcens_k=2.5, fqcens_ymin=0.1, fqcens_ymax=0.925
):
    return _sigmoid(logmpeak, fqcens_x0, fqcens_k, fqcens_ymin, fqcens_ymax)


def quenched_fraction_satellites(logsm, logmhost):
    x0, k, ymin, ymax = _get_satmodel_params(logsm)
    return _sigmoid(logmhost, x0, k, ymin, ymax)


def cluster_boosted_quenched_fraction_satellites(
    logsm, logmhost, satboost_x0=14.25, satboost_k=20, satboost_ymin=0, boost_factor=2
):
    y0 = quenched_fraction_satellites(logsm, logmhost)
    maximum_allowable_boost = 1 - y0
    satboost_ymax = maximum_allowable_boost / boost_factor
    satboost = _sigmoid(logmhost, satboost_x0, satboost_k, satboost_ymin, satboost_ymax)
    return satboost + y0


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (x - x0)))


def _get_satmodel_params(logsm):
    x0 = _satmodel_x0(logsm)
    k = _satmodel_k(logsm)
    ymin = _satmodel_ymin(logsm)
    ymax = _satmodel_ymax(logsm)
    return x0, k, ymin, ymax


def _satmodel_x0(logsm, fq_x0_x0=9.95, fq_x0_k=4, fq_x0_ymin=12.9, fq_x0_ymax=12.3):
    """Calculate sigmoid x0 parameter for satellite with logsm"""
    return _sigmoid(logsm, fq_x0_x0, fq_x0_k, fq_x0_ymin, fq_x0_ymax)


def _satmodel_k(logsm, fq_k_x0=10.25, fq_k_k=5, fq_k_ymin=0.9, fq_k_ymax=2.9):
    """Calculate sigmoid k parameter for satellite with logsm"""
    return _sigmoid(logsm, fq_k_x0, fq_k_k, fq_k_ymin, fq_k_ymax)


def _satmodel_ymin(
    logsm, fq_ymin_x0=9.95, fq_ymin_k=2.5, fq_ymin_ymin=0.15, fq_ymin_ymax=0.53
):
    """Calculate sigmoid ymin parameter for satellite with logsm"""
    return _sigmoid(logsm, fq_ymin_x0, fq_ymin_k, fq_ymin_ymin, fq_ymin_ymax)


def _satmodel_ymax(
    logsm, fq_ymax_x0=10.335, fq_ymax_k=1.75, fq_ymax_ymin=0.36, fq_ymax_ymax=1
):
    """Calculate sigmoid ymax parameter for satellite with logsm"""
    return _sigmoid(logsm, fq_ymax_x0, fq_ymax_k, fq_ymax_ymin, fq_ymax_ymax)
