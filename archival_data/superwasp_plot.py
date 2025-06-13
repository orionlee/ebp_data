import sys
if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")

if "../" not in sys.path:  # to get some helpers
    sys.path.append("../")

from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity

import lightkurve as lk


import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt

import os
import time
from types import SimpleNamespace

# my helpers
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


def _to_yyyy_mm(time_val: Time) -> str:
    dt = time_val.to_value("datetime")
    return f"{dt.year}-{dt.month:02}"


def bin_flux_and_plot(lc, ax, bins=200, aggregate_func=np.nanmedian):
    # I need to first copy the object because of issues in underlying astropy FoldedTimeSeries
    #  (e.g., normalized phase time column is in Quantity type rather than Time)
    lc_min = lc.copy()
    cols_to_remove = [c for c in lc_min.colnames if c not in ["time", "flux", "flux_err"]]
    lc_min.remove_columns(names=cols_to_remove)
    # by default use median (instead of mean) to reduce the effect of outliers
    lc_b = lc_min.bin(bins=bins, aggregate_func=aggregate_func)

    ax = lc_b.scatter(ax=ax, c="black", s=16, label="binned")

    return lc_b


def estimate_min_and_plot(lc_b, ax):
    min_idx = np.nanargmin(lc_b.flux)
    min_time = lc_b.time[min_idx]
    min_flux = lc_b.flux[min_idx]
    depth = (1 * u.dimensionless_unscaled - lc_b.flux[min_idx]).to(lc_b.flux.unit)

    ax.scatter(min_time,  min_flux.value,  edgecolor="red", facecolor=(0, 0, 0, 0), marker="o", s=64, label="Min estimate")
    return SimpleNamespace(idx=min_idx, time=min_time, flux=min_flux, depth=depth)


def create_phase_plot(
    lc, target_name, r, t0_time_format,
    truncate_sigma_upper=3, truncate_sigma_lower=9,
    wrap_phase=0.7,
    show_secondary_phase_plot=False,
):
    def safe_get(dict_like, key, default_value=np.nan):
        try:
            return dict_like[key]
        except KeyError:
            return default_value

    def get_t0_secondary():
        if np.isfinite(safe_get(r, "T0_sec")):  # has secondary eclipses
            return Time(r["T0_sec"], format=t0_time_format)
        elif np.isfinite(safe_get(r, ["Phase_sec"])):  # has secondary eclipses (in phase)
            return Time(r["T0"] + r["Period"] * r["Phase_sec"], format=t0_time_format)
        else:
            return None

    if target_name is None:
        target_name = lc.meta.get("LABEL", "<No target name>")

    lc = lc.remove_nans()  # nans will screw up truncation info display
    lc_orig = lc

    orig_min, orig_max = lc_orig.flux.min(), lc_orig.flux.max()

    lc = lc.remove_outliers(sigma_upper=truncate_sigma_upper, sigma_lower=truncate_sigma_lower)

    trunc_min, trunc_max = lc.flux.min(), lc.flux.max()

    t0 = Time(r["T0"], format=t0_time_format)
    lc_f = lc.fold(period=r['Period'], epoch_time=t0, normalize_phase=True, wrap_phase=wrap_phase)

    lc_f_sec = None
    if show_secondary_phase_plot:
        t0_sec = get_t0_secondary()
        period_sec = safe_get(r, "Period_sec")
        if not np.isfinite(period_sec):
            period_sec = r["Period"]
        lc_f_sec = lc.fold(period=period_sec, epoch_time=t0_sec, normalize_phase=True, wrap_phase=wrap_phase)

    with plt.style.context(lk.MPLSTYLE):
        if not show_secondary_phase_plot:
            fig, axs = plt.subplot_mosaic(
                [
                    ["pri", "pri"],
                    ["priz left", "priz right"],  # zoomed to eclipses
                ],
                figsize=(8, 4 * 2),
            )
        else:
            fig, axs = plt.subplot_mosaic(
                [
                    ["pri", "pri"],
                    ["priz left", "priz right"],  # zoomed to eclipses
                    ["sec", "sec"],
                ],
                figsize=(8, 4 * 3),
            )

        fig.tight_layout()

        fig.suptitle(f"{target_name}, P: {r['Period']} d", fontsize=12, y=1.05)

        # ax = tplt.scatter_partition_by(lc_f, "camera", ax=axs["pri"], s=1)
        ax = lc_f.scatter(ax=axs["pri"], s=1, label="t0: primary", c=lc_f.time_original.value, show_colorbar=False)
        ax.set_xlim(wrap_phase - 1, wrap_phase)  # ensure constant x scale independent of the data

        min_i_est = SimpleNamespace(idx=-1, time=np.nan, flux=np.nan * lc_f.flux.unit, depth=np.nan * lc_f.flux.unit)
        if len(lc_f) > 40:
            # attempt binning only if there is some reasonable number of data points
            try:
                lc_f_b = bin_flux_and_plot(lc_f, ax)
                min_i_est = estimate_min_and_plot(lc_f_b, ax)
            except Exception as e:
                print(f"Unexpected error in binning {target_name}. Binning is skipped. {e}")

        ax.legend(loc="lower right")
        camera_like_info = ""
        if "camera" in lc.colnames:  # SuperWASP-specific
            camera_like_info = f"  ;  # cameras: {len(np.unique(lc.camera))}"
        ax.set_title(
            f"""\
median err: {np.nanmedian(lc.flux_err):.0f} ; truncated: [{orig_min.value:.0f} - {trunc_min.value:.0f}), ({trunc_max.value:.0f} - {orig_max.value:.0f}] [{orig_min.unit}]
baseline: {_to_yyyy_mm(lc.time.min())} - {_to_yyyy_mm(lc.time.max())} ({(lc.time.max() - lc.time.min()).value:.0f} d){camera_like_info}""",
            fontsize=10,
            )
        ax.set_xlabel(None)

        # second / third plots: zoom to eclipses and annotate the plot with EBP eclipse params,
        p_phase, p_depth, p_dur = 0, r["Depth [ppt]"], r["Duration [hr]"] / 24 / r['Period']
        p_zoom_width = p_dur * 9  # zoom window proportional to eclipse duration
        p_zoom_width = min(max(p_zoom_width, 0.1), 0.5)  # but with a min / max of 0.1 / 0.5
        xlim = (p_phase - p_zoom_width / 2, p_phase + p_zoom_width / 2)
        p_lc_f = lc_f.truncate(*xlim)
        ax = p_lc_f.scatter(s=9, alpha=0.3, ax=axs["priz left"], label=None)
        ax.set_xlim(*xlim)  # ensure expected eclipses are centered
        ax.axvspan(p_phase - p_dur / 2, p_phase + p_dur / 2, color="red", alpha=0.2)
        f_median = np.nanmedian(lc.flux)
        ymin = (f_median - Quantity(p_depth, unit="ppt")).to(f_median.unit).value
        ymax = f_median.value
        ax.vlines(p_phase, ymin=ymin, ymax=ymax, color="blue", linestyle="-", linewidth=3)
        ax.set_xlabel(None)

        if np.isfinite(safe_get(r, "Phase_sec")):
            # Zoom in to secondary
            s_phase, s_depth, s_dur = r["Phase_sec"], r["Depth_sec"], r["Duration_sec"] / 24 / r['Period']
            s_zoom_width = s_dur * 9  # zoom window proportional to eclipse duration
            s_zoom_width = min(max(s_zoom_width, 0.1), 0.5)  # but with a min / max of 0.1 / 0.5
            xlim = (s_phase - s_zoom_width / 2, s_phase + s_zoom_width / 2)
            s_lc_f = lc_f.truncate(*xlim)
            ax = s_lc_f.scatter(s=9, alpha=0.3, ax=axs["priz right"], label=None)
            ax.set_xlim(*xlim)  # ensure expected eclipses are centered
            ax.axvspan(s_phase - s_dur / 2, s_phase + s_dur / 2, color="red", alpha=0.1)
            f_median = np.nanmedian(lc.flux)
            ymin = (f_median - Quantity(s_depth, unit="ppt")).to(f_median.unit).value
            ymax = f_median.value
            ax.vlines(s_phase, ymin=ymin, ymax=ymax, color="blue", linestyle="-", linewidth=3)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_ylim(*axs["priz left"].get_ylim())  # same y scale as the primary

        if lc_f_sec is not None:
            ax = lc_f_sec.scatter(ax=axs["sec"], s=1, label=f"t0: secondary,\nP_sec: {lc_f_sec.period}", c=lc_f_sec.time_original.value, show_colorbar=False)
            ax.set_xlim(wrap_phase - 1, wrap_phase)  # ensure constant x scale independent of the data
            ax.legend(loc="lower right")
            # Note: avoid using ax.set_title(), as it will bleed into the zoom plot above
            # ax.text(0.98, 0.98, f"P_sec: {lc_f_sec.period}", transform=ax.transAxes,  ha="right", va="top")

    return fig, lc, lc_f, min_i_est


def create_superwasp_phase_plot(
    row,
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

    # returned LC has outliers truncated
    fig, lc, lc_f, min_i_est = create_phase_plot(
        lc_orig, f"{lc_orig.label} / TIC {tic}", r, "btjd",
        truncate_sigma_upper=truncate_sigma_upper, truncate_sigma_lower=truncate_sigma_lower,
        wrap_phase=wrap_phase,
    )

    if save_plot:
        fig.savefig(plot_path, dpi=72, bbox_inches="tight")

    if display_plot:
        plt.show()

    return SimpleNamespace(
        sourceid=sourceid, tic=tic, lc_orig=lc_orig, lc=lc, lc_f=lc_f, fig=fig, min_i_est=min_i_est,
        plot_path=plot_path, skipped=False,
    )


def create_all_plots(sleep_time=1, first_n=None, min_i_est_out_path="tmp/superwasptimeseries_match_w_tic_min_i_est.csv", plot_dir="plots/superwasp", skip_if_created=True):
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
            if min_i_est_out_path is not None and not res.skipped:
                min_i_est_dict = dict(
                    sourceid=row.sourceid,
                    est_min_i_phase=res.min_i_est.time,
                    est_min_i_flux=res.min_i_est.flux.value,
                    est_min_i_depth=res.min_i_est.depth.value,
                    )
                to_csv(min_i_est_dict, min_i_est_out_path, mode="a")
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
# - might need to first remove existing tmp/superwasptimeseries_match_w_tic_min_i_est.csv if rerunning
#   (the script appends to csv)
if __name__ == "__main__":
    create_all_plots(
        # first_n=10,
        sleep_time=0,
        plot_dir="plots/tmp",
        # skip_if_created=False,
    )
