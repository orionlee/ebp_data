import sys

if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")

import os
import re
from types import SimpleNamespace

from astropy.table import Table, MaskedColumn, join, vstack
from astropy.time import Time

import numpy as np

import matplotlib.pyplot as plt

import lightkurve as lk

from common_plot import create_phase_plot

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


#
# Plot codes
#


def _plot_zoomed_lc(lc, lc_b, ax):
    lc_e = lc[lc["orientation"] == "east"]
    lc_w = lc[lc["orientation"] == "west"]
    lc_e.scatter(s=4, c="darkgreen", alpha=0.4, ax=ax, label=None)
    lc_w.scatter(s=4, c="orange", alpha=0.4, ax=ax, label=None)

    if lc_b is not None:
        lc_b.scatter(s=25, alpha=0.9, ax=ax, label=None)
    return ax


def _plot_zoomed_lc_w_label(lc, lc_b, ax):
    lc_e = lc[lc["orientation"] == "east"]
    lc_w = lc[lc["orientation"] == "west"]
    lc_e.scatter(s=4, c="darkgreen", alpha=0.4, ax=ax, label="east")
    lc_w.scatter(s=4, c="orange", alpha=0.4, ax=ax, label="west")

    if lc_b is not None:
        lc_b.scatter(s=25, alpha=0.9, ax=ax, label=None)
    ax.legend(loc="upper left")
    return ax


def create_kelt_phase_plot(
    row,
    flux_column=None,
    lc_preprocess_func=None,
    truncate_sigma_upper=3,
    truncate_sigma_lower=9,
    wrap_phase=0.7,
    display_plot=False,
    save_plot=False,
    plot_dir="plots/kelt",
    skip_if_created=False,
):
    r = row  # shorthand to be used below

    sourceid, tic = r.kelt_sourceid, r.TIC

    plot_filename = f"{sourceid.replace(' ', '_')}.png"
    plot_path = f"{plot_dir}/{plot_filename}"

    if skip_if_created and os.path.isfile(plot_path):
        return SimpleNamespace(
            sourceid=sourceid, tic=tic, plot_path=plot_path, skipped=True
        )

    lc_orig = read_kelt_tbls(sourceid, flux_column=flux_column)

    if lc_preprocess_func is not None:
        lc_orig = lc_preprocess_func(lc_orig)

    # returned LC has outliers truncated
    fig, lc, lc_f, stats = create_phase_plot(
        lc_orig,
        f"{lc_orig.label} / TIC {tic}",
        r,
        "btjd",
        truncate_sigma_upper=truncate_sigma_upper,
        truncate_sigma_lower=truncate_sigma_lower,
        wrap_phase=wrap_phase,
        plot_zoomed_lc_func_left=_plot_zoomed_lc_w_label,
        plot_zoomed_lc_func_right=_plot_zoomed_lc,
    )

    if save_plot:
        fig.savefig(plot_path, dpi=72, bbox_inches="tight")

    if display_plot:
        plt.show()

    return SimpleNamespace(
        sourceid=sourceid,
        tic=tic,
        lc_orig=lc_orig,
        lc=lc,
        lc_f=lc_f,
        fig=fig,
        stats=stats,
        plot_path=plot_path,
        skipped=False,
    )
