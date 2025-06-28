import sys

if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")

import os
import re
from types import SimpleNamespace

from astropy.table import Table, MaskedColumn, join, vstack
from astropy.time import Time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import lightkurve as lk

from common_plot import create_phase_plot

# my helpers
import lightkurve_ext as lke


# needed for create_all_plots
if "../" not in sys.path:  # to get some helpers
    sys.path.append("../")

import time
import requests
from multi_stars.common import to_csv  # from ../multi_stars


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


# not used in the module, but is kept as a convenience for module callers
# to replace te standard eclipse zoom plots with ones with errorbar
def _errorbar_zoomed_lc_w_label(lc, lc_b, ax):
    lc_e = lc[lc["orientation"] == "east"]
    lc_w = lc[lc["orientation"] == "west"]
    lc_e.errorbar(marker="o", markersize=2, c="darkgreen", alpha=0.4, ax=ax, label="east")
    lc_w.errorbar(marker="o", markersize=2, c="orange", alpha=0.4, ax=ax, label="west")

    if lc_b is not None:
        lc_b.errorbar(marker="o", markersize=4, c="black", alpha=0.9, ax=ax, label=None)
    ax.legend(loc="upper left")
    return ax


def create_kelt_phase_plot(
    row,
    lc_preprocess_func=None,
    truncate_sigma_upper=3,
    truncate_sigma_lower=9,
    wrap_phase=0.7,
    display_plot=False,
    save_plot=False,
    plot_dir="plots/kelt",
    skip_if_created=False,
    plot_zoomed_lc_func_left=_plot_zoomed_lc_w_label,
    plot_zoomed_lc_func_right=_plot_zoomed_lc,
):
    r = row  # shorthand to be used below

    sourceid, tic = r.kelt_sourceid, r.TIC

    plot_filename = f"{sourceid.replace(' ', '_')}.png"
    plot_path = f"{plot_dir}/{plot_filename}"

    if skip_if_created and os.path.isfile(plot_path):
        return SimpleNamespace(
            sourceid=sourceid, tic=tic, plot_path=plot_path, skipped=True
        )

    fig, axs = plt.subplot_mosaic(
        [
            ["r_pri", "r_pri",  "t_pri", "t_pri"],
            ["r_priz left", "r_priz right",  "t_priz left", "t_priz right"],  # zoomed to eclipses
        ],
        figsize=(8.5 * 2, 4 * 2),
    )
    fig.tight_layout(w_pad=4)  # to add horizontal space between the RAW plot (left) and TAF plot (right)

    flux_column ="raw_mag"
    lc_orig = read_kelt_tbls(sourceid, flux_column=flux_column)
    if lc_preprocess_func is not None:
        lc_orig = lc_preprocess_func(lc_orig)

    # returned LC has outliers truncated
    r_axs = {"pri": axs["r_pri"], "priz left": axs["r_priz left"], "priz right": axs["r_priz right"]}
    _, lc, lc_f, stats = create_phase_plot(
        lc_orig,
        f"{lc_orig.label} / TIC {tic}",
        r,
        "btjd",
        axs=r_axs,
        truncate_sigma_upper=truncate_sigma_upper,
        truncate_sigma_lower=truncate_sigma_lower,
        wrap_phase=wrap_phase,
        plot_zoomed_lc_func_left=plot_zoomed_lc_func_left,
        plot_zoomed_lc_func_right=plot_zoomed_lc_func_right,
    )
    r_axs["pri"].set_ylabel(f"Normalized {flux_column.upper()}")

    # repeat it for tfa_mag
    lc_tfa, lc_tfa_f, stats_tfa = None, None, None,
    try:
        flux_column = "tfa_mag"
        lc_orig = read_kelt_tbls(sourceid, flux_column=flux_column)
        if lc_preprocess_func is not None:
            lc_orig = lc_preprocess_func(lc_orig)

        # returned LC has outliers truncated
        t_axs = {"pri": axs["t_pri"], "priz left": axs["t_priz left"], "priz right": axs["t_priz right"]}
        _, lc_tfa, lc_tfa_f, stats_tfa = create_phase_plot(
            lc_orig,
            f"{lc_orig.label} / TIC {tic}",
            r,
            "btjd",
            axs=t_axs,
            truncate_sigma_upper=truncate_sigma_upper,
            truncate_sigma_lower=truncate_sigma_lower,
            wrap_phase=wrap_phase,
            plot_zoomed_lc_func_left=plot_zoomed_lc_func_left,
            plot_zoomed_lc_func_right=plot_zoomed_lc_func_right,
        )
        t_axs["pri"].set_ylabel(f"Normalized {flux_column.upper()}")
    except ValueError as ve:
        if "not a column" in str(ve):
            # case the source has no tfa data. Just no-op
            axs["t_pri"].set_title("[No TFA data]")
        else:
            raise ve

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
        lc_tfa=lc_tfa,
        lc_tfa_f=lc_tfa_f,
        stats_tfa=stats_tfa,
        fig=fig,
        stats=stats,
        plot_path=plot_path,
        skipped=False,
    )


def create_all_plots(sleep_time=0, first_n=None, stats_out_path="tmp/kelt_match_w_tic_stats.csv", plot_dir="plots/kelt", skip_if_created=True):
    ss_df = pd.read_csv("tmp/kelt_match_w_tic_ebp.csv")
    if first_n is not None:
        # process first_n entries, typically for trial, debug
        ss_df = ss_df.iloc[:first_n]
    print(f"Creating plots for {len(ss_df)} entries...")

    for i in range(len(ss_df)):
        row = ss_df.iloc[i]
        msg = f"{i: >4}: {row.kelt_sourceid}"
        res = SimpleNamespace(skipped=False)
        try:
            res = create_kelt_phase_plot(
                row, display_plot=False, save_plot=True, plot_dir=plot_dir, skip_if_created=skip_if_created
                )
            if res.skipped:
                msg += " [skipped]"
            plt.close()
            print(msg)
            if stats_out_path is not None and not res.skipped:
                # for now we only store the stats of raw Lc
                est_dict = dict(
                    sourceid=row.kelt_sourceid,
                    flux_err_median=res.stats.flux_err_median.value,
                    num_points_in_pri=res.stats.num_points_in_pri,
                    num_points_in_sec=res.stats.num_points_in_sec,
                    est_min_i_phase=res.stats.min_i_est.time,
                    est_min_i_flux=res.stats.min_i_est.flux.value,
                    est_min_i_depth=res.stats.min_i_est.depth.value,
                    )
                to_csv(est_dict, stats_out_path, mode="a")
        except requests.HTTPError as he:
            if 400 <= he.response.status_code < 500:
                # case the requested object is not found. Inform the users and continue
                msg += f" [not found] {he}"
                print(msg)
            else:
                # raise  other unexpected error
                raise he
        if sleep_time is not None and sleep_time > 0 and not res.skipped:
            # to avoid bombarding the server, not needed if all lightcurves have been downloaded locally
            time.sleep(sleep_time)


# From command line
# - might need to first remove existing tmp/kelt_match_w_tic_stats.csv if rerunning
#   (the script appends to csv)
if __name__ == "__main__":
    create_all_plots(
        # first_n=10,
        sleep_time=0,
        plot_dir="plots/tmp",
        skip_if_created=False,
    )
