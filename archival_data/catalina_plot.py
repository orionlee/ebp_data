import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u


def move(df: pd.DataFrame, colname: str, before_colname: str):
    """Move the column `colname` before the column `before_colname`."""
    col_to_move = df.pop(colname)
    loc = df.columns.get_loc(before_colname)
    return df.insert(loc, colname, col_to_move)


# Catalina result can be identified by master ID (or MasterFrame column in bulk match),
# but its UI (and common usage) uses a coordinate-based naming scheme, e,g.,
# CSS_J143601.3-273409, SSS_J143601.3-273410
def format_catalina_iau_name(tab, id_colname="MasterFrame", ra_colname="RA", dec_colname="Decl"):
    def format_one_coord(id, ra, dec):
        coord = SkyCoord(ra, dec, unit=u.deg)

        # anecdotally, CSS / SSS can be deduced by MasterFrame
        prefix = "CSS" if id < 3000000000000 else "SSS"

        # example coord format: J065232.3+142630
        _r, _d = coord.ra.hms, coord.dec.dms
        return f"{prefix}_J{int(_r.h):02}{int(_r.m):02}{int(_r.s):02}.{round(_r.s % 1 * 10):01}{int(_d.d):+02}{int(abs(_d.m)):02}{int(abs(_d.s)):02}"

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

