from collections.abc import Mapping, Sequence
import csv
import json
import os

from astropy.table import Table
import numpy as np
import pandas as pd


def _df_to_csv(df, out_path, mode="a"):
    if (not mode.startswith("w")) and (os.path.exists(out_path)) and (os.path.getsize(out_path) > 0):
        header = False
    else:
        header = True
    return df.to_csv(out_path, index=False, mode=mode, header=header)


def to_csv(data, out_path, mode="a", fieldnames=None):
    if isinstance(data, Table):
        data = data.to_pandas()

    if isinstance(data, pd.DataFrame):
        return _df_to_csv(data, out_path, mode=mode)

    # parameters processing
    if fieldnames is None:
        if isinstance(data, Mapping):
            fieldnames = data.keys()
        elif isinstance(data, Sequence):
            fieldnames = data[0].keys()
        else:
            raise TypeError(f"Unsupported type for `data`: {type(dict)}")

    def write_header_if_needed():
        if (not mode.startswith("w")) and (os.path.exists(out_path)) and (os.path.getsize(out_path) > 0):
            return False  # the file has content. no need to write header
        header = ",".join(fieldnames)
        header = header + "\n"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(header)

    def to_csv_of_dict(a_dict):
        with open(out_path, mode, encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames, dialect="unix")
            csv_writer.writerow(a_dict)

    # Main logic
    write_header_if_needed()
    if isinstance(data, Mapping):
        to_csv_of_dict(data)
    elif isinstance(data, Sequence):
        [to_csv_of_dict(a_dict) for a_dict in data]
    else:
        raise TypeError(f"Unsupported type for `data`: {type(dict)}")


# from https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


## Used to serialize Comments
def json_np_dump(obj, fp, **kwargs):
    """JSON dump that supports numpy data types"""
    kwargs["cls"] = NpEncoder
    return json.dump(obj, fp, **kwargs)
