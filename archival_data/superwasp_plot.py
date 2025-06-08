import sys
if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")

from astropy.table import Table
from astropy.time import Time
import astropy.units as u

import lightkurve as lk


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time
from types import SimpleNamespace

# my helpers
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


def _to_yyyy_mm(time_val: Time) -> str:
    dt = time_val.to_value("datetime")
    return f"{dt.year}-{dt.month:02}"


def create_superwasp_phase_plot(row, display_plot=False, save_plot=False, plot_dir="plots/superwasp"):
    r = row  # shorthand to be used below

    sourceid, tic = r.sourceid, r.TIC

    lc = read_superwasp_dr1_csv(sourceid)
    lc = lke.to_normalized_flux_from_mag(lc).normalize(unit="ppt")

    lc_orig = lc

    orig_min, orig_max = lc_orig.flux.min(), lc_orig.flux.max()

    lc = lc.remove_outliers(sigma_upper=3, sigma_lower=9)

    trunc_min, trunc_max = lc.flux.min(), lc.flux.max()

    lc_f = lc.fold(period=r.Period, epoch_time=Time(r.T0, format="btjd"), normalize_phase=True, wrap_phase=0.7)

    with plt.style.context(lk.MPLSTYLE):
        fig, axs = plt.subplots(2, 1, figsize=(8, 4 * 2))
        plt.tight_layout()

        fig.suptitle(f"{lc.label} / TIC {tic}, P: {r.Period} d", fontsize=12, y=1.05)

        # ax = tplt.scatter_partition_by(lc_f, "camera", ax=axs[0], s=0.3, alpha=0.2)
        ax = lc_f.scatter(ax=axs[0], s=0.7, label=None, c=lc_f.time_original.value, show_colorbar=False)
        ax.set_title(
            f"""\
# cameras: {len(np.unique(lc.camera))}  ;  truncated [{orig_min.unit}]: [{orig_min.value:.0f}, {trunc_min.value:.0f}), ({trunc_max.value:.0f}, {orig_max.value:.0f}]
baseline: {_to_yyyy_mm(lc.time.min())} - {_to_yyyy_mm(lc.time.max())} ({(lc.time.max() - lc.time.min()).value:.0f} d)""",
            fontsize=10,
            )
        ax.set_xlabel(None)

        # second plot: annotate the plot with EBP eclipse params,
        ax = lc_f.scatter(s=0.7, alpha=0.3, ax=axs[1], label=None)

        p_phase, p_depth, p_dur = 0, r["Depth [ppt]"], r["Duration [hr]"] / 24 / r.Period
        ax.axvspan(p_phase - p_dur / 2, p_phase + p_dur / 2, color="red", alpha=0.2)
        ax.vlines(p_phase, ymin=1000 - p_depth, ymax=1000, color="blue", linestyle="--", linewidth=3)

        s_phase, s_depth, s_dur = r["Phase_sec"], r["Depth_sec"], r["Duration_sec"] / 24 / r.Period
        ax.axvspan(s_phase - s_dur / 2, s_phase + s_dur / 2, color="red", alpha=0.1)
        ax.vlines(s_phase, ymin=1000 - s_depth, ymax=1000, color="blue", linestyle="--", linewidth=3)

    plot_path = None
    if save_plot:
        plot_filename = f"{sourceid.replace(' ', '_')}.png"
        plot_path = f"{plot_dir}/{plot_filename}"
        fig.savefig(plot_path, dpi=72, bbox_inches="tight")

    if display_plot:
        plt.show()

    return SimpleNamespace(sourceid=sourceid, tic=tic, lc_orig=lc_orig, lc=lc, lc_f=lc_f, ax=ax, plot_path=plot_path)


def create_all_plots(sleep_time=1, first_n=None):
    ss_df = pd.read_csv("tmp/superwasptimeseries_match_w_tic_ebp.csv")
    if first_n is not None:
        # process first_n entries, typically for trial, debug
        ss_df = ss_df.iloc[:first_n]
    print(f"Creating plots for {len(ss_df)} entries...")

    for i in range(len(ss_df)):
        res = create_superwasp_phase_plot(ss_df.iloc[i], display_plot=False, save_plot=True)
        print(f"{i: >4}: {res.sourceid}")
        if sleep_time is not None and sleep_time > 0:
            # to avoid bombarding the server, not needed if all lightcurves have been downloaded locally
            time.sleep(sleep_time)


# From command line
if __name__ == "__main__":
    create_all_plots(
        # first_n=10,
        # sleep_time=0,
    )
