import re


# For KELT data in NASA Exoplanet Archive
#  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblSearch/nph-tblSearchInit?app=ExoTbls&config=kelttimeseries
# Each coordinate / source typically is split into two, one for east orientation and one for west orientation
# In the database, the kelt_sourceid only uses east sourceid. i.e., a row for the west orientation still has the east sourceid
#
# To combine the east and west data as a single source, we rename the lightcurve data of west orientation using  the id from the east
#
def map_west_to_east(bulk_download_file, first_n=None):
    with open(bulk_download_file) as f:
        cur_east_id = None
        for i, line in enumerate(f):
            if not line.startswith("wget "):
                continue
            if first_n is not None and i >= first_n:
                break

            # wget -O 'KELT_N04_lc_092937_V01_east_raw_lc.tbl' 'http://exoplanetarchive.ipac.caltech.edu:80/data/ETSS//KELT2/005/081/31/KELT_N04_lc_092937_V01_east_raw_lc.tbl' -a search_30522120.log
            cur_file_match = re.search("'([^']+)'", line)
            if cur_file_match is None:
                print("Unexpected line: skipped. ", line)
                continue

            cur_file = cur_file_match[1]
            id_match = re.match("(.+)(_(raw|tfa)_lc.tbl)", cur_file)
            cur_id, cur_suffix = id_match[1], id_match[2]
            # print(cur_id, cur_suffix, "-", line)
            if cur_id.endswith("_east"):
                cur_east_id = cur_id
                continue
            # else it's west, map it to east
            new_west_id = cur_east_id.replace("_east", "_west")
            print(f"mv {cur_file} {new_west_id}{cur_suffix}")


#
# Main entry point
# Usage:
# python kelt_map_west_id_to_east.py  > data/cache/kelt/kelt_rename_west_to_east_id.sh
#
if __name__ == "__main__":
    map_west_to_east(
        "data/cache/kelt/kelt_download_exoarch_8507.sh",
        # first_n=20,
    )
