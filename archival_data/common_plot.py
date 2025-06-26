from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity

import lightkurve as lk


import numpy as np

import matplotlib.pyplot as plt

from types import SimpleNamespace

# my helpers
# import lightkurve_ext as lke
# import tic_plot as tplt


def _to_yyyy_mm(time_val: Time) -> str:
    dt = time_val.to_value("datetime")
    return f"{dt.year}-{dt.month:02}"


def bin_flux_and_plot(lc, ax, bins=200, aggregate_func=np.nanmedian, binned_label=None):
    # I need to first copy the object because of issues in underlying astropy FoldedTimeSeries
    #  (e.g., normalized phase time column is in Quantity type rather than Time)
    lc_min = lc.copy()
    cols_to_remove = [c for c in lc_min.colnames if c not in ["time", "flux", "flux_err"]]
    lc_min.remove_columns(names=cols_to_remove)
    # by default use median (instead of mean) to reduce the effect of outliers
    lc_b = lc_min.bin(bins=bins, aggregate_func=aggregate_func)

    ax = lc_b.scatter(ax=ax, c="black", s=16, label=binned_label)

    return lc_b


def estimate_min_and_plot(lc_b, ax):
    min_idx = np.nanargmin(lc_b.flux)
    min_time = lc_b.time[min_idx]
    min_flux = lc_b.flux[min_idx]
    depth = (1 * u.dimensionless_unscaled - lc_b.flux[min_idx]).to(lc_b.flux.unit)

    ax.scatter(min_time,  min_flux.value,  edgecolor="red", facecolor=(0, 0, 0, 0), marker="o", s=100, label="Min estimate")
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

    def get_t0_secondary():  # TODO: is the logic still sound given r["Phase_sec"] is rounded?
        if np.isfinite(safe_get(r, "T0_sec")):  # has secondary eclipses
            return Time(r["T0_sec"], format=t0_time_format)
        elif np.isfinite(safe_get(r, ["Phase_sec"])):  # has secondary eclipses (in phase)
            return Time(r["T0"] + r["Period"] * r["Phase_sec"], format=t0_time_format)
        else:
            return None

    def get_t0_secondary_phase():
        # Note: r["Phase_sec"] value is rounded, and is unsuitable for plots
        t0_sec = safe_get(r, "T0_sec")

        if not np.isfinite(t0_sec):
            return np.nan

        t0 = r["T0"]
        period = r["Period"]

        s_phase = ((t0_sec - t0) / period) % 1
        # shift s_phase to the plot range, defined by wrap_phase:
        if s_phase < wrap_phase - 1:  # too small, e.g. -0.4
            s_phase += 1
        elif s_phase > wrap_phase: # too large, e.g., 0.9
            s_phase -= 1
        return s_phase

    if target_name is None:
        target_name = lc.meta.get("LABEL", "<No target name>")

    lc = lc.remove_nans()  # nans will screw up truncation info display
    lc_orig = lc

    orig_min, orig_max = lc_orig.flux.min(), lc_orig.flux.max()

    lc = lc.remove_outliers(sigma_upper=truncate_sigma_upper, sigma_lower=truncate_sigma_lower)

    trunc_min, trunc_max = lc.flux.min(), lc.flux.max()

    # Need to first process s_phase before doing any plot
    # as it affects various aspects
    s_phase = get_t0_secondary_phase()
    if wrap_phase - s_phase < 0.05:  # too close to the right edge, shift wrap_phase to give it more margin
        wrap_phase = min(wrap_phase + 0.1, 1)

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
        ax = lc_f.scatter(ax=axs["pri"], s=1, label=None, c=lc_f.time_original.value, show_colorbar=False)
        ax.set_xlim(wrap_phase - 1, wrap_phase)  # ensure constant x scale independent of the data

        # gather some stats (to be returned and/or used in the plot)
        # - for num data points in pri, sec
        # - median error
        flux_err_median = np.nanmedian(lc.flux_err)
        num_points_in_pri = np.nan  # placeholder
        num_points_in_sec = np.nan  # placeholder
        # placeholder value in case Min I is not estimated
        min_i_est = SimpleNamespace(idx=-1, time=np.nan, flux=np.nan * lc_f.flux.unit, depth=np.nan * lc_f.flux.unit)
        lc_f_b = None
        if len(lc_f) > 40:
            # attempt binning only if there is some reasonable number of data points
            try:
                lc_f_b = bin_flux_and_plot(lc_f, ax)
                min_i_est = estimate_min_and_plot(lc_f_b, ax)
            except Exception as e:
                print(f"Unexpected error in binning {target_name}. Binning is skipped. {e}")

        ax.axvline(0, 0, 0.15, c="red", linewidth=2, linestyle="--", label="t0 (primary)")
        if np.isfinite(s_phase):
            ax.axvline(s_phase, 0, 0.08, c="red", linewidth=2, linestyle="dotted", label="t0_sec")

        ax.legend(loc="lower right" if s_phase < 0 else "lower left")

        camera_like_info = ""
        if "camera" in lc.colnames:  # SuperWASP-specific
            camera_like_info = f"  ;  # cameras: {len(np.unique(lc.camera))}"
        ax.set_title(
            f"""\
median err: {flux_err_median:.0f} ; truncated: [{orig_min.value:.0f} - {trunc_min.value:.0f}), ({trunc_max.value:.0f} - {orig_max.value:.0f}] [{orig_min.unit}]
baseline: {_to_yyyy_mm(lc.time.min())} - {_to_yyyy_mm(lc.time.max())} ({(lc.time.max() - lc.time.min()).value:.0f} d){camera_like_info}""",
            fontsize=10,
            )
        ax.set_xlabel(None)

        # second / third plots: zoom to eclipses and annotate the plot with EBP eclipse params,
        p_phase, p_depth, p_dur = 0, r["Depth [ppt]"], r["Duration [hr]"] / 24 / r['Period']
        num_points_in_pri = len(lc_f.truncate(p_phase - p_dur / 2, p_phase + p_dur / 2))
        p_zoom_width = p_dur * 9  # zoom window proportional to eclipse duration
        p_zoom_width = min(max(p_zoom_width, 0.1), 0.5)  # but with a min / max of 0.1 / 0.5
        xlim = (p_phase - p_zoom_width / 2, p_phase + p_zoom_width / 2)
        p_lc_f = lc_f.truncate(*xlim)
        ax = p_lc_f.scatter(s=4, c="gray", alpha=0.4, ax=axs["priz left"], label=None)
        if lc_f_b is not None:
            ax = lc_f_b.truncate(*xlim).scatter(s=25, alpha=0.9, ax=ax, label=None)
        ax.set_xlim(*xlim)  # ensure expected eclipses are centered and x scale is constant
        ax.axvspan(p_phase - p_dur / 2, p_phase + p_dur / 2, color="red", alpha=0.2)
        f_median = np.nanmedian(lc.flux)
        ymin = (f_median - Quantity(p_depth, unit="ppt")).to(f_median.unit).value
        ymax = f_median.value
        ax.vlines(p_phase, ymin=ymin, ymax=ymax, color="blue", linestyle="-", linewidth=3)
        p_depth_to_flux_err = Quantity(p_depth, "ppt") / flux_err_median
        ax.text(
            0.04, 0.02, f"Depth_pri\n / median err\n = {p_depth_to_flux_err:.0f}",
            transform=ax.transAxes, color="blue", ha="left", va="bottom",
        )
        ax.set_xlabel(None)

        if np.isfinite(s_phase):
            # Zoom in to secondary
            s_depth, s_dur = r["Depth_sec"], r["Duration_sec"] / 24 / r['Period']  # s_phase obtained at the beginning
            num_points_in_sec = len(lc_f.truncate(s_phase - s_dur / 2, s_phase + s_dur / 2))
            s_zoom_width = s_dur * 9  # zoom window proportional to eclipse duration
            s_zoom_width = min(max(s_zoom_width, 0.1), 0.5)  # but with a min / max of 0.1 / 0.5
            xlim = (s_phase - s_zoom_width / 2, s_phase + s_zoom_width / 2)
            s_lc_f = lc_f.truncate(*xlim)
            ax = s_lc_f.scatter(s=4, c="gray", alpha=0.4, ax=axs["priz right"], label=None)
            if lc_f_b is not None:
                ax = lc_f_b.truncate(*xlim).scatter(s=25, alpha=0.9, ax=ax, label=None)
            ax.set_xlim(*xlim)  # ensure expected eclipses are centered and x scale is constant
            ax.axvspan(s_phase - s_dur / 2, s_phase + s_dur / 2, color="red", alpha=0.1)
            f_median = np.nanmedian(lc.flux)
            ymin = (f_median - Quantity(s_depth, unit="ppt")).to(f_median.unit).value
            ymax = f_median.value
            ax.vlines(s_phase, ymin=ymin, ymax=ymax, color="blue", linestyle="-", linewidth=3)
            s_depth_to_flux_err = Quantity(s_depth, "ppt") / flux_err_median
            ax.text(
                0.04, 0.02, f"Depth_sec\n/ median err\n = {s_depth_to_flux_err:.0f}",
                transform=ax.transAxes, color="blue", ha="left", va="bottom",
            )
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_ylim(*axs["priz left"].get_ylim())  # same y scale as the primary

        if lc_f_sec is not None:
            ax = lc_f_sec.scatter(ax=axs["sec"], s=1, label=f"t0: secondary,\nP_sec: {lc_f_sec.period}", c=lc_f_sec.time_original.value, show_colorbar=False)
            ax.set_xlim(wrap_phase - 1, wrap_phase)  # ensure constant x scale independent of the data
            ax.legend(loc="lower right")
            # Note: avoid using ax.set_title(), as it will bleed into the zoom plot above
            # ax.text(0.98, 0.98, f"P_sec: {lc_f_sec.period}", transform=ax.transAxes,  ha="right", va="top")

    stats = SimpleNamespace(
        flux_err_median=flux_err_median,
        num_points_in_pri=num_points_in_pri,
        num_points_in_sec=num_points_in_sec,
        min_i_est=min_i_est,
    )
    return fig, lc, lc_f, stats
