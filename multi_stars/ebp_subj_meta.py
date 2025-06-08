from pathlib import Path
import re

from common import to_csv
from zn_common import get_subject_meta_of_id, bulk_process
from ebp_subj_ids import load_subject_ids_from_file


def _get_ebp_subject_meta_of_id(id, json=False):
    res = get_subject_meta_of_id(id)
    if json:
        return res

    subject = res["subjects"][0]

    tic_id = int(subject["metadata"].get("TIC_ID", -1))

    # extract subject image's uuid
    img_url = subject["locations"][0].get("image/png", "")
    img_id = img_url
    match_res = re.match(r"https://panoptes-uploads.zooniverse.org/subject_location/(.+)[.]png", img_url)
    if match_res is not None:
        img_id = match_res[1]

    res = dict(
        subject_id=int(subject["id"]),
        tic_id=tic_id,
        img_id=img_id,
    )
    return res


def get_subject_meta_of_ids(ids, subject_result_func=None):
    kwargs_list = [dict(id=id) for id in ids]
    return bulk_process(_get_ebp_subject_meta_of_id, kwargs_list, process_result_func=subject_result_func)


def save_meta_of_subject(subject_meta, call_i, call_kwargs):
    out_path = Path("data/ebp_subj_meta.csv")
    fieldnames = ["subject_id", "tic_id", "img_id"]
    to_csv(subject_meta, out_path, mode="a", fieldnames=fieldnames)


#
# Top level driver
#
if __name__ == "__main__":
    ids = load_subject_ids_from_file()
    # ids = ids[:2]
    print(f"Meta for {len(ids)} subjects: {ids[0]} ... {ids[-1]}")
    get_subject_meta_of_ids(ids, subject_result_func=save_meta_of_subject)

