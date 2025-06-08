from pathlib import Path
import re

import numpy as np

from zn_common import get_subject_ids_of_tag, get_subject_ids_of_collection


# EBP Zooniverse Project ID used in API calls
EBP_ZN_ID = 22939


def save_subject_ids_of_page(out_path, subject_ids, call_i=None, call_kwargs=None):
    with open(out_path, "a") as f:
        np.savetxt(f, subject_ids, fmt="%s")

    return out_path


def _csv_path_of_tag(tag):
    return Path(f"data/cache/ebp_tag_subj_ids_{tag}.csv")


def save_all_subject_ids_of_tag(tag, skip_if_exists=True):
    out_path = _csv_path_of_tag(tag)

    if skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[DEBUG] tag {tag} skipped (with existing data)")
        return

    def do_save(subject_ids, call_i, call_kwargs):
        save_subject_ids_of_page(out_path, subject_ids, call_i, call_kwargs)

    get_subject_ids_of_tag(EBP_ZN_ID, tag, page_result_func=do_save)


MULTI_STAR_TAGS = [
    "multiple-system-candidate",
    "multiple_star",
    "multi-star",
    "multiple-system",
    "extra-eclipses",
]


def save_all_subject_ids_from_tags(tags=MULTI_STAR_TAGS, skip_if_exists=True):
    for a_tag in tags:
        save_all_subject_ids_of_tag(a_tag, skip_if_exists=skip_if_exists)


def _csv_path_of_collection(collection_name):
    collection_name_encoded = re.sub("[/\:]", "_", collection_name)
    return Path(f"data/cache/ebp_coll_subj_ids_{collection_name_encoded}.csv")


def save_all_subject_ids_of_collection(collection_name, skip_if_exists=True):
    out_path = _csv_path_of_collection(collection_name)

    if skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[DEBUG] collection {collection_name} skipped (with existing data)")
        return

    def do_save(subject_ids):
        save_subject_ids_of_page(out_path, subject_ids, None, None)

    get_subject_ids_of_collection(collection_name, result_func=do_save)


MULTI_STAR_COLLECTIONS = [
    "orionlee/ebp-multi-star-system-candidates-new",
    "mhuten/multiple-star-system-eb-patrol",
    # d0ct0r's collection is not public,
    # manually extract JSON result from an authenticated browser session:
    # https://www.zooniverse.org/api/collections?http_cache=true&slug=d0ct0r%2Fmultiple-system-candidates-eclipsing-binary-patrol
    # "d0ct0r/multiple-system-candidates-eclipsing-binary-patrol",
]


def save_all_subject_ids_from_collections(collections=MULTI_STAR_COLLECTIONS, skip_if_exists=True):
    for a_coll in collections:
        save_all_subject_ids_of_collection(a_coll, skip_if_exists=skip_if_exists)


def _combine_n_save_all_subject_ids():
    out_path = Path("data/ebp_subj_ids.csv")

    paths_tags = [_csv_path_of_tag(t) for t in MULTI_STAR_TAGS]
    paths_colls = [_csv_path_of_collection(c) for c in MULTI_STAR_COLLECTIONS]
    paths_all = paths_tags + paths_colls

    all_ids = np.array([], dtype=int)
    for a_path in paths_all:
        a_file_ids = np.genfromtxt(a_path, dtype=int)
        all_ids = np.concatenate((all_ids, a_file_ids), 0)
    all_ids = np.unique(all_ids)

    with open(out_path, "w") as f:  # always overwrite
        np.savetxt(f, all_ids, fmt="%s")


def save_all_subject_ids_from_all(skip_if_exists=True):
    save_all_subject_ids_from_tags(skip_if_exists=skip_if_exists)
    save_all_subject_ids_from_collections(skip_if_exists=skip_if_exists)
    _combine_n_save_all_subject_ids()


def load_subject_ids_from_file():
    csv_path = Path("data/ebp_subj_ids.csv")
    return np.genfromtxt(csv_path, dtype=int)


#
# Top level driver
#
if __name__ == "__main__":
    save_all_subject_ids_from_all(skip_if_exists=True)
