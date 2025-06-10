import contextlib
with contextlib.redirect_stdout(None):
    # Suppress the "Could not import regions" warning from XMatch.
    # - it is a `print()`` call, so I have to redirect stdout,
    # running the risk of missing some other warning
    from astroquery.xmatch import XMatch

from astropy.table import Table
import astropy.units as u

import numpy as np


def read_ztf_quad_table():
    # from Vaessen+ 2024:
    # https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/682/A164

    # ignore "deg" so as to ignore the unit line after the column header
    tab = Table.read("tmp/ZTF_quads_Vaessen_2024.tsv", format="csv", delimiter=";", comment="#|-|deg")
    return tab


def xmatch_with_tic(first_n=None, max_distance=30 * u.arcsec, out_filename=None):
    # Note: 30 arcsec is totally unnecessary. The empirical results show the max distance of the matched TIC is 0.28 arcsec
    catalog = "vizier:IV/39/tic82"

    src = read_ztf_quad_table()
    if first_n is not None:
        src = src[:first_n]
    for c in src.colnames:
        src.rename_column(c, f"ZTF_{c}")
    src["ZTF_matchid"] = np.arange(1, len(src) + 1)  # to preserve the order of the original catalog

    df = XMatch.query(cat1=src, cat2=catalog, colRA1="ZTF_RAJ2000", colDec1="ZTF_DEJ2000", max_distance=max_distance).to_pandas()

    df = df.sort_values(by=["ZTF_matchid", "angDist"])
    # Add a new column 'order_within_matchid' to represent the order within each group
    df['order_within_matchid'] = df.groupby('ZTF_matchid').cumcount() + 1
    df["match_GAIA"] = df["ZTF_GaiaDR3"] == df["GAIA"]  # if Gaia sourceid matches

    if out_filename is not None:
        df.to_csv(out_filename, index=False)

    return df


def process_xmatch_raw_result(df, out_filename=None):
    """Process the raw CDS XMatch result to produce the final version"""

    # reviewed the raw results,
    # for all but 3: the closest match also has matching Gaia source (DR3 in ZTF, DR2 in TIC)
    # for the 3: the closest match is  the the correct one.
    df = df[df["order_within_matchid"] == 1]

    # for our purpose, we don't need the entire TIC metadata, so select only a few columns

    # first keep all ZTF columns
    cols_out = [c for c in df.columns if c.startswith("ZTF_")]
    cols_out += ["TIC", "Tmag", "GAIA", "angDist"]
    df = df[cols_out]

    df.reset_index(drop=True, inplace=True)

    if out_filename is not None:
        df.to_csv(out_filename, index=False)

    return df
