import sys

if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")

import os
import re

from astropy.table import Table, MaskedColumn, join, vstack
from astropy.time import Time

import lightkurve as lk


import numpy as np

# my helpers
import lightkurve_ext as lke

# def _read_a_kelt_tbl(path_or_url, kelt_id=None) -> lk.LightCurve:
#     tab = Table.read(path_or_url, format="ascii.ipac")
#     # follow lightkurve convention
#     tab.rename_column("TIME", "time")
#     tab.rename_column("MAG", "mag")
#     tab.rename_column("MAG_ERR", "mag_err")

#     tab["time"] = Time(tab["time"], format="jd", scale="utc")
#     tab["flux"] = tab["mag"]
#     tab["flux_err"] = tab["mag_err"]

#     lc = lk.LightCurve(data=tab)
#     lc.meta.update({
#         "OBJNAME": kelt_id,
#         "LABEL": kelt_id,
#         "FLUX_ORIGIN": "mag",
#         "FILEURL": path_or_url,
#     })

#     return lc


#
# KELT raw vs TFA
# https://web.archive.org/web/20241107072709/https://keltsurvey.org/lightcurves
# - Raw data are the direct products of the difference imaging pipeline.
# - If you are looking for large amplitude periodic variations, like Cepheids or RR Lyrae,
#   or punctuated variability at significant amplitude, like EBs or flares, we recommend using the raw data.
#   The typical RMS for the raw lightcurves is 1% to 3% for stars with 8 < V < 10.
# - We also have a detrending procedure using the TFA algorithm from Kovács et al, which removes common trends to the data analogous to the Kepler CBVs.
# - That is better for looking for small-amplitude variability like Delta Scu and Gamma Dor pulsations, or transits.
# - The typical RMS of TFA lightcurves is 0.5% to 1.5% for stars with 8 < V < 10.
#
def _read_kelt_raw_n_tfa_tbls(
    kelt_id: str, data_dir="data/cache/kelt", flux_column=None
) -> lk.LightCurve:
    path_raw = f"{data_dir}/{kelt_id}_raw_lc.tbl"
    path_tfa = f"{data_dir}/{kelt_id}_tfa_lc.tbl"

    tab_r = Table.read(path_raw, format="ascii.ipac")
    tab_r.rename_column("TIME", "time")
    tab_r.rename_column("MAG", "raw_mag")
    tab_r.rename_column("MAG_ERR", "raw_mag_err")

    if os.path.exists(path_tfa):
        tab_t = Table.read(path_tfa, format="ascii.ipac")
        tab_t.rename_column("TIME", "time")
        tab_t.rename_column("MAG", "tfa_mag")
        tab_t.rename_column("MAG_ERR", "tfa_mag_err")

        tab = join(
            tab_t, tab_r, keys="time", join_type="outer", metadata_conflicts="silent"
        )

        # convert missing values from masked to nan for ease of subsequent processing
        for c in tab.colnames:
            if isinstance(tab[c], MaskedColumn):
                tab[c] = tab[c].filled(np.nan)
    else:
        tab = tab_r

    tab["flux"] = tab["raw_mag"]
    tab["flux_err"] = tab["raw_mag_err"]

    orientation = re.search("_([^_]+)$", kelt_id)[1]
    tab["orientation"] = orientation

    tab["time"] = Time(tab["time"], format="jd", scale="utc")

    lc = lk.LightCurve(data=tab)
    lc.meta.update(
        {
            "OBJNAME": kelt_id,
            "LABEL": kelt_id,
            "FLUX_ORIGIN": "raw_mag",
            "FILEURL_RAW": path_raw,
            "FILEURL_TFA": path_tfa,
        }
    )

    if flux_column is not None:
        return lc.select_flux(flux_column)

    return lc


def read_kelt_tbls(
    kelt_id: str,
    data_dir="data/cache/kelt",
    flux_column=None,
    normalize=True,
    normalize_unit="ppt",
) -> lk.LightCurve:
    kelt_id_prefix = kelt_id.replace("_east", "")
    if kelt_id_prefix == kelt_id:
        raise ValueError(f"Expected a kelt_id ended  with '_east'. Actual: {kelt_id}")

    # I have renamed the west data files with the prefix of the east counterpart with `kelt_map_west_id_to_easy.py`
    # so that I can easily read them together
    kelt_id_east = f"{kelt_id_prefix}_east"
    kelt_id_west = f"{kelt_id_prefix}_west"

    lc_east = _read_kelt_raw_n_tfa_tbls(
        kelt_id_east, data_dir=data_dir, flux_column=flux_column
    )
    lc_west = _read_kelt_raw_n_tfa_tbls(
        kelt_id_west, data_dir=data_dir, flux_column=flux_column
    )

    if normalize:
        # we normalize east / west data separately, in case they have some zero-point shift
        lc_east = lke.to_normalized_flux_from_mag(lc_east).normalize(
            unit=normalize_unit
        )
        lc_west = lke.to_normalized_flux_from_mag(lc_west).normalize(
            unit=normalize_unit
        )

    # stitch them together
    tab = vstack([lc_east, lc_west], metadata_conflicts="silent")
    tab.sort("time")

    lc = lk.LightCurve(data=tab)
    lc.meta = lc_east.meta

    return lc
