import json
import os
from pathlib import Path

import numpy as np

from ratelimit import limits, sleep_and_retry

from common import json_np_dump, to_csv
from zn_common import fetch_json, bulk_process
from ebp_subj_ids import load_subject_ids_from_file

# throttle HTTP calls to Zooniverse
NUM_CALLS = 5
TEN_SECONDS = 10


@sleep_and_retry
@limits(calls=NUM_CALLS, period=TEN_SECONDS)
def _get_subject_comment_of_id_n_page(id, page):
    url = (
        f"https://talk.zooniverse.org/comments?http_cache=true&section=project-22939&focus_type=Subject"
        f"&sort=-created_at&focus_id={id}&page={page}"
    )
    return fetch_json(url)


def _get_subject_comments_of_id_from_remote(id):
    # fetch all pages and combine them to 1 JSON object

    res = _get_subject_comment_of_id_n_page(id, 1)
    res["meta"]["subject_id"] = id  # add it to the result for ease of identification
    num_pages = res["meta"]["comments"]["page_count"]
    for page in range(2, num_pages + 1):
        page_res = _get_subject_comment_of_id_n_page(id, page)
        res["comments"] = res["comments"] + page_res["comments"]

    return res


def _get_subject_comments_of_id(id):
    try:
        return load_subject_comments_of_id_from_file(id)
    except FileNotFoundError:
        return _get_subject_comments_of_id_from_remote(id)


def get_subject_comments_of_ids(ids, subject_result_func=None):
    kwargs_list = [dict(id=id) for id in ids]
    return bulk_process(_get_subject_comments_of_id, kwargs_list, process_result_func=subject_result_func)


def save_comments_of_subject(subject_comments, call_i, call_kwargs):
    id = subject_comments["meta"]["subject_id"]
    out_path = Path(f"data/cache/comments/c{id}.json")  # the c prefix hints it is a comment
    with open(out_path, "w") as f:
        json_np_dump(subject_comments, f)


def save_comment_authors_of_subject(subject_comments, call_i, call_kwargs):
    out_path = "tmp/ebp_quad_vetted_VK_subj_comments_meta.csv"
    id = subject_comments["meta"]["subject_id"]
    for c in subject_comments["comments"]:
        # the unique URL suffix for the comment in ZN UI
        url_suffix = f'/{c["board_id"]}/{c["discussion_id"]}?comment={c["id"]}'
        to_csv(
            dict(
                subject=id,
                user_login=c["user_login"],
                comment_href=url_suffix,
                tagging="|".join(c["tagging"].keys()),
            ),
            out_path
        )


def process_comments_of_subject(subject_comments, call_i, call_kwargs):
    save_comments_of_subject(subject_comments, call_i, call_kwargs)
    save_comment_authors_of_subject(subject_comments, call_i, call_kwargs)


def load_subject_comments_of_id_from_file(subject_id):
    with open(f"data/cache/comments/c{subject_id}.json", "r") as f:
        return json.load(f)


def load_vk_subject_ids():
    csv_path = Path("tmp/ebp_quad_vetted_VK_subj_ids.txt")
    return np.genfromtxt(csv_path, dtype=int)


#
# Top level driver
#
# - directory data/cache/comments need to be first created
if __name__ == "__main__":
    # ids = load_subject_ids_from_file()
    ids = load_vk_subject_ids()
    # ids = ids[0:2]
    print(f"Comments for {len(ids)} subjects: {ids[0]} ... {ids[-1]}")

    try:  # to create a new comment summary csv each time we run
        os.remove("tmp/ebp_quad_vetted_VK_subj_comments_meta.csv")
    except OSError:
        pass
    get_subject_comments_of_ids(ids, subject_result_func=process_comments_of_subject)
