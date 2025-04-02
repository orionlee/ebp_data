from pathlib import Path
import re

import numpy as np

from zn_common import get_subject_ids_of_tag, get_subject_ids_of_collection


# EBP Zooniverse Project ID used in API calls
EBP_ZN_ID = 22939


def save_subject_ids_of_page(out_path, subject_ids, call_i, call_kwargs):
    with open(out_path, "a") as f:
        np.savetxt(f, subject_ids, fmt="%s")

    return out_path


def save_all_subject_ids_of_tag(tag, skip_if_exists=True):
    out_path = Path(f"data/cache/ebp_tag_subj_ids_{tag}.csv")

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


def save_all_subject_ids_of_collection(collection_name, skip_if_exists=True):
    collection_name_encoded = re.sub("[/\:]", "_", collection_name)
    out_path = Path(f"data/cache/ebp_coll_subj_ids_{collection_name_encoded}.csv")

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


# #
# # Top level driver
# #
# if __name__ == "__main__":
#     # params for sectors #eclipsingbinary in sectors 1 to 39
#     # Subject 68601250 is the first sector 40 on page 3230
#     # get_subject_ids_of_tag(
#     #     "eclipsingbinary", 1, 3230, end_subject_id_exclusive=68601250, page_result_func=save_subject_ids_of_page
#     # )

#     # I decided to crawl all pages instead of trying to limit to sectors 1 to 39
#     # because the subject ids are not strictly increasing.
#     # Filtering will be done afterwards
#     page_start, page_end_inclusive, end_subject_id_exclusive = 1, 3602, None
#     # page_start, page_end_inclusive, end_subject_id_exclusive = 1001, 3602, None

#     print(f"EB subject ids of page [{page_start}, {page_end_inclusive}] ; end_subject_id_exclusive={end_subject_id_exclusive}")
#     get_subject_ids_of_tag(
#         "eclipsingbinary",
#         page_start,
#         page_end_inclusive,
#         end_subject_id_exclusive=end_subject_id_exclusive,
#         page_result_func=save_subject_ids_of_page,
#     )
