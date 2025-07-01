import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.table import Table
from astropy.time import Time

import lightkurve as lk

import matplotlib.pyplot as plt

from common_plot import create_phase_plot

if "../../PH_TESS_LightCurveViewer/" not in sys.path:  # to get some helpers
    sys.path.append("../../PH_TESS_LightCurveViewer/")
import lightkurve_ext as lke


def move(df: pd.DataFrame, colname: str, before_colname: str):
    """Move the column `colname` before the column `before_colname`."""
    col_to_move = df.pop(colname)
    loc = df.columns.get_loc(before_colname)
    return df.insert(loc, colname, col_to_move)


# Catalina result can be identified by master ID (or MasterFrame column in bulk match),
# but its UI (and common usage) uses a coordinate-based naming scheme, e,g.,
# CSS_J143601.3-273409, SSS_J143601.3-273410
def format_catalina_iau_name(tab, id_colname="MasterFrame", ra_colname="RA", dec_colname="Decl"):
    def sign(v):
        return "-" if v < 0 else "+"

    def format_one_coord(id, ra, dec):
        coord = SkyCoord(ra, dec, unit=u.deg)

        # anecdotally, CSS / SSS can be deduced by MasterFrame
        prefix = "CSS" if id < 3000000000000 else "SSS"

        # example coord format: J065232.3+142630
        _r, _d = coord.ra.hms, coord.dec.dms
        return f"{prefix}_J{int(_r.h):02}{int(_r.m):02}{int(_r.s):02}.{round(_r.s % 1 * 10):01}{sign(_d.d)}{int(abs(_d.d)):02}{int(abs(_d.m)):02}{int(abs(_d.s)):02}"

    return [format_one_coord(id, r, d) for id, r, d in zip(tab[id_colname], tab[ra_colname], tab[dec_colname])]

#
# Catalina XMatch preparation / processing
#


def read_ebp_csv():
    eb_df = pd.read_csv("tmp/EBP_EBs_Catalog_Tmag_lt_13.csv")
    eb_df["matchid"] = eb_df.index + 1  # for join with NASA result later
    eb_df = eb_df.loc[:, ~eb_df.columns.str.contains("^Unnamed")]  # drop an empty unnamed column (between Notes and Gaia_ID)
    return eb_df


def prepare_coord_fils_for_xmatch(batch_size=100):
    eb_df = read_ebp_csv()
    # for bulk-matching CRTS
    # http://nesssi.cacr.caltech.edu/DataRelease/usage_multicone.html
    for i in range(0, int(np.ceil(len(eb_df) / batch_size))):
        # I could use TIC instead of matchid , it'd be more meaningful
        # but I use matchid as it's used by NASA Exoplanets Archive so subsequent matching logic can be adapted more easily
        eb_df[["matchid", "RA", "Dec"]].iloc[i*batch_size:(i+1)*batch_size].to_csv(f"tmp/EBP_EBs_Catalog_Tmag_lt_13_coords_cs_{i}.csv", sep=" ", index=False, header=False)


#
# XMatch is done by sending the coordinate csv files one by one to Catalina using browser
# using default search radius of 0.002 deg (7.2 arcsec), long output format
# http://nunuku.caltech.edu/cgi-bin/getmulticonedb_release2.cgi
#

# Ensure none of the output exceeds the 40000 line limit
# wc -l cs_match_raw_*.csv

# Combine all raw xmatch result in 1 file
# cp cs_match_raw_0.csv cs_match_raw_all.csv
# awk 'FNR>1' cs_match_raw_1.csv cs_match_raw_2.csv cs_match_raw_3.csv cs_match_raw_4.csv cs_match_raw_5.csv cs_match_raw_6.csv cs_match_raw_7.csv cs_match_raw_8.csv cs_match_raw_9.csv >> cs_match_raw_all.csv
#

def get_all_raw_xmatch_csv():
    df = pd.read_csv("tmp/cs_match_raw_all.csv")
    return df


# Create per-input (search coordinate) summary
def _to_per_input_summary(df_raw=None):
    if df_raw is None:
        df_raw = get_all_raw_xmatch_csv()

    # Each master identifies a coordinate and a sub-survey (CSS /SSS)
    by_master = df_raw.groupby("MasterFrame").agg(
        InputID=("InputID", "first"),  # first is enough, as InputID is the same for all rows in the group
        Mag=("Mag", "mean"),
        RA=("RA", "mean"),
        Decl=("Decl", "mean"),
        Epochs=("InputID", "count"),
        Blend=("Blend", "max"),
    ).sort_values(by=["InputID", "MasterFrame"], ascending=True).reset_index()

    def concat_masters(id1, id2):
        if id1 == id2:
            return id1
        return f"{id1}_{id2}"

    by_input = by_master.groupby("InputID").agg(
        Num_IDs=("MasterFrame", "count"),
        CS_Mag_Mean=("Mag", "mean"),
        MasterFrame1=("MasterFrame", "first"),
        MasterFrame2=("MasterFrame", "last"),
        RA1=("RA", "first"),
        Decl1=("Decl", "first"),
        RA2=("RA", "last"),
        Decl2=("Decl", "last"),
    ).reset_index()

    by_input["IAU_Name1"] = format_catalina_iau_name(by_input, id_colname="MasterFrame1", ra_colname="RA1", dec_colname="Decl1")
    by_input["IAU_Name2"] = format_catalina_iau_name(by_input, id_colname="MasterFrame2", ra_colname="RA2", dec_colname="Decl2")
    by_input["Masters"] = [concat_masters(id1, id2) for id1, id2 in zip(by_input["MasterFrame1"], by_input["MasterFrame2"])]  # serves as the logical ID

    move(by_input, "Masters", "Num_IDs")

    return by_input


def join_ebp_with_catalina():

    eb_df = read_ebp_csv()

    # Catalina XMatch result (summary on per object)
    df = _to_per_input_summary()

    # join df with df_eb by matchid / InputID
    df = pd.merge(df, eb_df, left_on="InputID", right_on="matchid", how="left")
    df.drop(columns="matchid", inplace=True)

    df["diff_mag"] = np.abs(df["CS_Mag_Mean"] - df["Tmag"])

    move(df, "TIC", "InputID")

    df.to_csv("tmp/cs_match_w_tic_ebp.csv", index=False, mode="w")
    return df


def to_catalina_lightcurves():
    """Split the Catalina bulk search result into per-coordinate LC files"""
    df_raw = get_all_raw_xmatch_csv()

    filenames = []
    by_input = _to_per_input_summary(df_raw)
    for input_id, master_ids, in zip(by_input.InputID, by_input.Masters):
        df = df_raw[df_raw.InputID == input_id]
        # OPEN: drop InputID column, as it's only relevant to bulk search input
        filename = f"cs_{master_ids}.csv"
        df.to_csv(f"data/cache/catalina/{filename}", index=False, mode="w")
        filenames.append(filename)
    return filenames


#
# Read Catalina lightcurve data
#


def read_catalina_csv(url, to_time_format=None, label=None, normalize=True, normalize_unit="ppt"):
    """Read CSV lightcurve files from Catalina Surveys Data Release.
    (i.e, the ASCII format in the UI)
    http://nesssi.cacr.caltech.edu/DataRelease/
    """

    def to_iau_names_str(tab, master_colname="masterframe"):
        # Each master identifies a coordinate and a sub-survey (CSS /SSS)
        by_master = tab.to_pandas().groupby(master_colname).agg(
            ra=("ra", "mean"),
            decl=("decl", "mean"),
        ).sort_values(by=[master_colname], ascending=True).reset_index()

        iau_names = format_catalina_iau_name(by_master, id_colname=master_colname, ra_colname="ra", dec_colname="decl")
        return ", ".join(iau_names)

    tab = Table.read(
        url,
        format="ascii.csv",
        converters={
            "MasterID": np.uint64,
            "MasterFrame": np.uint64,  # it shows up in the csv from bulk match (in place of MaserID)
            "ID": str,  # only in long format,
            #     ^^^ the int values (20 digits) could be too big for np.int64,
            #         not sure even np.unit64 could fit all. So use str to be safe.
            "FrameID": np.uint64,  # only in long format
        },
    )

    # rename columns to fit lightcurve convention
    tab.rename_column("MJD", "time")
    for c in tab.colnames:
        tab.rename_column(c, c.lower())
    if "flux" in tab.colnames:
        # Optional column "Flux" is in the long format
        # rename it to avoid collision with lightkurve's special flux column
        tab.rename_column("flux", "raw_flux")

    # add units
    tab["mag"] *= u.mag
    tab["magerr"] *= u.mag
    tab["ra"] *= u.deg
    tab["decl"] *= u.deg
    tab["time"] = Time(tab["time"], format="mjd", scale="utc")
    if to_time_format is not None:
        # for user convenience, e.g., to convert the default mjd to jd
        tab["time"].format = to_time_format

    # copy mag as the primary column
    tab["flux"] = tab["mag"]
    tab["flux_err"] = tab["magerr"]

    lc = lk.LightCurve(data=tab)
    lc.meta["FLUX_ORIGIN"] = "mag"
    lc.meta["FILEURL"] = url
    if label is not None:
        lc.meta["LABEL"] = label
    else:
        lc.meta["LABEL"] = to_iau_names_str(tab)

    if normalize:
        # OPEN: we might want to normalize separately by masterframe, in case they have some zero-point shift
        lc = lke.to_normalized_flux_from_mag(lc).normalize(unit=normalize_unit)

    return lc


#
# Create Plot figures
#

def _plot_zoomed_lc_w_label(lc, lc_b, ax):
    master_ids = np.unique(lc["masterframe"])
    master_ids.sort()  # ensure CSS before SSS

    colors = ["darkgreen", "orange"]
    for i, id in enumerate(master_ids):
        _lc = lc[lc["masterframe"] == id]

        # see the heuristics in format_catalina_iau_name()
        label = "CSS" if id < 3000000000000 else "SSS"
        _lc.scatter(s=9, c=colors[i], alpha=0.4, ax=ax, label=label)
    if lc_b is not None:
        lc_b.scatter(s=49, alpha=0.9, ax=ax, label=None)
    ax.legend(loc="upper left")
    return ax


def _plot_zoomed_lc(lc, lc_b, ax):
    ax = _plot_zoomed_lc_w_label(lc, lc_b, ax)
    ax.get_legend().remove()
    return ax


def create_catalina_phase_plot(
    row,
    lc_preprocess_func=None,
    truncate_sigma_upper=3,
    truncate_sigma_lower=9,
    wrap_phase=0.7,
    display_plot=False,
    save_plot=False,
    plot_dir="plots/catalina",
    skip_if_created=False,
    plot_zoomed_lc_func_left=_plot_zoomed_lc_w_label,
    plot_zoomed_lc_func_right=_plot_zoomed_lc,
):
    r = row  # shorthand to be used below

    sourceid, tic = r.Masters, r.TIC

    plot_filename = f"cs_{sourceid}.png"
    plot_path = f"{plot_dir}/{plot_filename}"

    if skip_if_created and os.path.isfile(plot_path):
        return SimpleNamespace(
            sourceid=sourceid, tic=tic, plot_path=plot_path, skipped=True
        )

    lc_path = f"data/cache/catalina/cs_{sourceid}.csv"
    lc_orig = read_catalina_csv(lc_path, to_time_format="jd")
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
        plot_zoomed_lc_func_left=plot_zoomed_lc_func_left,
        plot_zoomed_lc_func_right=plot_zoomed_lc_func_right,
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

