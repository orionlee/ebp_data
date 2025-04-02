import time

import requests
from tqdm import tqdm

from ratelimit import limits, sleep_and_retry
from urllib.parse import quote


# throttle HTTP calls to Zooniverse
NUM_CALLS = 10
TEN_SECONDS = 10


def fetch_json(url):
    # the header is needed for Zooniverse subject metadata API call
    headers = {"accept": "application/vnd.api+json; version=1", "content-type": "application/json"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()


def bulk_process(
    process_fn,
    process_kwargs_list,
    return_result=False,
    process_result_func=None,
    tqdm_kwargs=None,
    num_retries=2,
    retry_wait_time_seconds=30,
):
    if tqdm_kwargs is None:
        tqdm_kwargs = dict()

    num_to_process = len(process_kwargs_list)
    res = []
    for i in tqdm(range(num_to_process), **tqdm_kwargs):
        for try_num in range(1, num_retries + 1):
            try:
                i_res = process_fn(**process_kwargs_list[i])
                if return_result:
                    res.append(i_res)
                else:
                    res.append(True)  # indicate the process is a success, in case we support skipping those that lead to error
                if process_result_func is not None:
                    process_result_func(i_res, i, process_kwargs_list[i])
                break
            except BaseException as err:
                print(f"Error in processing {i}th call. Arguments: {process_kwargs_list[i]}. Error: {type(err)} {err}")
                if try_num < num_retries:
                    print(f"Retry after {retry_wait_time_seconds} seconds for try #{try_num + 1}")
                    time.sleep(retry_wait_time_seconds)
                else:
                    raise err

    return res


@sleep_and_retry
@limits(calls=NUM_CALLS, period=TEN_SECONDS)
def _get_subject_ids_of_tag_page(project_id, tag, page, end_subject_id_exclusive=None, json=False, also_return_meta=False):
    url = (
        f"https://talk.zooniverse.org/tags/popular?http_cache=true&taggable_type=Subject"
        f"&section=project-{project_id}&name={tag}&page={page}"
    )
    res = fetch_json(url)
    if json:
        return res

    res_json = res
    res = [e["taggable_id"] for e in res["popular"]]

    # use case: filter out subjects beyond the intended sector range in the last page
    if end_subject_id_exclusive is not None:
        res_all = res
        res = []
        for id in res_all:
            if id < end_subject_id_exclusive:
                res.append(id)
            else:
                print(f"[DEBUG] subject {id} at page {page} is excluded")
    if also_return_meta:
        return res, res_json["meta"]["popular"]
    else:
        return res


def get_subject_ids_of_tag(project_id, tag, page_start=None, page_end_inclusive=None, end_subject_id_exclusive=None, page_result_func=None):

    # first determine the start, end page, defaulted to all by using the result metadata
    if page_start is None:
        page_start = 1

    if page_end_inclusive is None:
        _, meta = _get_subject_ids_of_tag_page(project_id=project_id, tag=tag, page=page_start, also_return_meta=True)
        # print(_)  # they are subject_ids
        # print(meta)
        page_end_inclusive = meta["page_count"]
        print(f"[DEBUG] tag: {tag}, page_start: {page_start}, page_end_inclusive: {page_end_inclusive}")

    kwargs_list = [
        dict(project_id=project_id, tag=tag, page=i, end_subject_id_exclusive=end_subject_id_exclusive)
        for i in range(page_start, page_end_inclusive + 1)
    ]
    return bulk_process(_get_subject_ids_of_tag_page, kwargs_list, process_result_func=page_result_func)





@sleep_and_retry
@limits(calls=NUM_CALLS, period=TEN_SECONDS)
def _get_subject_ids_of_collection(collection_name, json=False):
    # collection_name is in the form of username/collection_name
    collection_name_encoded = quote(collection_name, safe="")  # safe="" to ensure "/" is encoded
    url = (
        f"https://www.zooniverse.org/api/collections?http_cache=true"
        f"&slug={collection_name_encoded}"
    )
    res = fetch_json(url)
    if json:
        return res

    # res_json = res
    res = res["collections"][0]["links"]["subjects"]
    return res


def get_subject_ids_of_collection(collection_name, result_func=None):
    print(f"[DEBUG] collection: {collection_name}")
    subject_ids = _get_subject_ids_of_collection(collection_name)
    result_func(subject_ids)
