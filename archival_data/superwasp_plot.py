import sys
if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")

if "../" not in sys.path:  # to get some helpers
    sys.path.append("../")

from astropy.table import Table
from astropy.time import Time
import astropy.units as u

import lightkurve as lk


import pandas as pd
import requests

import matplotlib.pyplot as plt

import os
import time
from types import SimpleNamespace

# my helpers
from common_plot import create_phase_plot
from multi_stars.common import to_csv  # from ../multi_stars
import lightkurve_ext as lke
import download_utils
# import tic_plot as tplt


def read_superwasp_dr1_csv(superwasp_id: str, cache_dir="data/cache/superwasp") -> lk.LightCurve:
    """Read SuperWASP DR1 data, in CSV format available at
    https://wasp.cerit-sc.cz/
    """
    from urllib.parse import quote_plus

    # camera data is not in FITs but can be found in CSV (CSV has no quality and some other columns though )
    csv_url = f"https://wasp.cerit-sc.cz/csv?object={quote_plus(superwasp_id)}"

    local_filename = f"{superwasp_id.replace(' ', '_')}.csv"
    local_filepath = download_utils.download_file(
        csv_url, filename=local_filename, download_dir=cache_dir, cache_policy_func=download_utils.CachePolicy.ALWAYS_USE
    )

    tab = Table.read(local_filepath, format="ascii")
    tab.rename_column("HJD", "time")  # follow lightkurve convention
    tab["time"] = Time(tab["time"], format="jd", scale="utc")
    tab.rename_column("magnitude error", "magnitude_err")  # follow lightkurve convention
    tab["magnitude"] *= u.mag
    tab["magnitude_err"] *= u.mag
    tab["flux"] = tab["magnitude"]
    tab["flux_err"] = tab["magnitude_err"]

    lc = lk.LightCurve(data=tab)
    lc.meta.update({
        "OBJNAME": superwasp_id,
        "LABEL": superwasp_id,
        "FLUX_ORIGIN": "magnitude",
        "FILEURL": csv_url,
    })

    return lc


def create_superwasp_phase_plot(
    row,
    lc_preprocess_func=None,
    truncate_sigma_upper=3, truncate_sigma_lower=9,
    wrap_phase=0.7,
    display_plot=False, save_plot=False, plot_dir="plots/superwasp", skip_if_created=False,
):
    r = row  # shorthand to be used below

    sourceid, tic = r.sourceid, r.TIC

    plot_filename = f"{sourceid.replace(' ', '_')}.png"
    plot_path = f"{plot_dir}/{plot_filename}"

    if skip_if_created and os.path.isfile(plot_path):
        return SimpleNamespace(sourceid=sourceid, tic=tic, plot_path=plot_path, skipped=True)

    lc_orig = read_superwasp_dr1_csv(sourceid)
    lc_orig = lke.to_normalized_flux_from_mag(lc_orig).normalize(unit="ppt")

    if lc_preprocess_func is not None:
        lc_orig = lc_preprocess_func(lc_orig)

    # returned LC has outliers truncated
    fig, lc, lc_f, stats = create_phase_plot(
        lc_orig, f"{lc_orig.label} / TIC {tic}", r, "btjd",
        truncate_sigma_upper=truncate_sigma_upper, truncate_sigma_lower=truncate_sigma_lower,
        wrap_phase=wrap_phase,
    )

    if save_plot:
        fig.savefig(plot_path, dpi=72, bbox_inches="tight")

    if display_plot:
        plt.show()

    return SimpleNamespace(
        sourceid=sourceid, tic=tic, lc_orig=lc_orig, lc=lc, lc_f=lc_f, fig=fig, stats=stats,
        plot_path=plot_path, skipped=False,
    )


def create_all_plots(sleep_time=1, first_n=None, stats_out_path="tmp/superwasptimeseries_match_w_tic_stats.csv", plot_dir="plots/superwasp", skip_if_created=True):
    ss_df = pd.read_csv("tmp/superwasptimeseries_match_w_tic_ebp.csv")
    if first_n is not None:
        # process first_n entries, typically for trial, debug
        ss_df = ss_df.iloc[:first_n]
    print(f"Creating plots for {len(ss_df)} entries...")

    for i in range(len(ss_df)):
        row = ss_df.iloc[i]
        msg = f"{i: >4}: {row.sourceid}"
        res = SimpleNamespace(skipped=False)
        try:
            res = create_superwasp_phase_plot(
                row, display_plot=False, save_plot=True, plot_dir=plot_dir, skip_if_created=skip_if_created
                )
            if res.skipped:
                msg += " [skipped]"
            print(msg)
            plt.close()
            if stats_out_path is not None and not res.skipped:
                est_dict = dict(
                    sourceid=row.sourceid,
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
# - might need to first remove existing tmp/superwasptimeseries_match_w_tic_stats.csv if rerunning
#   (the script appends to csv)
if __name__ == "__main__":
    create_all_plots(
        # first_n=10,
        sleep_time=0,
        plot_dir="plots/tmp",
        # skip_if_created=False,
    )
