import polars as pl
import os
import json  # if proves to be too slow or memory consuming look into https://github.com/ICRAR/ijson + https://github.com/lloyd/yajl
from collections import Counter
import argparse
from functools import reduce, partial, lru_cache
from collections.abc import Iterable
from operator import itemgetter
import datetime
from dateutil.parser import parse

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--pt_record_csv",
    type=str,
    help="CSV containing patient MRNs and earliest dates",
)


parser.add_argument(
    "--notes_dir",
    type=str,
    help="Directory containing nested directories of notes contained in JSON files",
)


# Keep it simple by avoiding time information for now
# we can fold it back in if we need that degree of granularity
@lru_cache
def parse_and_normalize_date(dt_str: str) -> datetime.date:
    parsed_dt = parse(dt_str, fuzzy=True)
    return parsed_dt.date()


@lru_cache
def is_before(pt_earliest: str, note_date: str) -> bool:
    return parse_and_normalize_date(pt_earliest) <= parse_and_normalize_date(note_date)


def fn_match_and_after_earlist(
    mrn_to_earliest_date: dict[int, str], note_json: dict[str, str | int]
) -> tuple[str, bool]:
    mrn = int(note_json["DFCI_MRN"])
    if mrn not in mrn_to_earliest_date.keys():
        return mrn, False
    pt_earliest = mrn_to_earliest_date.get(mrn)
    if pt_earliest is None:
        ValueError(
            f"{mrn} is not associated with a date despite nulls having been dropped earlier"
        )
        return mrn, False
    note_date = note_json.get("EVENT_DATE")
    if note_date is None:
        ValueError(f"{note_json} missing an event date")
        return mrn, False
    return mrn, is_before(pt_earliest, note_date)


def get_file_totals(
    mrn_to_earliest_date: dict[int, str], json_path: str
) -> Counter[str]:
    with open(json_path) as f:
        note_json_list = json.load(f)["response"]["docs"]
    local_relevance_check = partial(fn_match_and_after_earlist, mrn_to_earliest_date)
    return Counter(
        mrn
        for mrn, is_relevant in map(local_relevance_check, note_json_list)
        if is_relevant
    )


def get_dir_totals(
    mrn_to_earliest_date: dict[int, str], notes_dir: str
) -> Iterable[Counter[int]]:
    for root, dirs, files in os.walk(notes_dir):
        for fn in files:
            if fn.lower().endswith("json"):
                yield get_file_totals(mrn_to_earliest_date, os.path.join(root, fn))


def get_totals(
    mrn_to_earliest_date: dict[int, str], notes_dir: str
) -> Iterable[tuple[int, int]]:
    full_counter = reduce(
        Counter.__add__, get_dir_totals(mrn_to_earliest_date, notes_dir)
    )
    # return sorted by totals
    return sorted(full_counter.items(), key=itemgetter(1), reverse=True)


def write_totals(pt_record_csv: str, notes_dir: str, output_dir: str) -> None:
    pt_record_df = pl.read_csv(pt_record_csv)
    mrn_and_date_df = (
        pt_record_df.with_columns(pl.col("mrn").cast(pl.Int64).alias("mrn"))
        .select("mrn", "earliest_date")
        .drop_nulls()
    )
    assert all(mrn_and_date_df.is_unique()), (
        f"Not unique in {mrn_and_date_df} {mrn_and_date_df.is_unique()}"
    )
    mrn_to_earliest_date = {
        mrn: earliest_date
        for mrn, earliest_date in zip(
            mrn_and_date_df["mrn"], mrn_and_date_df["earliest_date"]
        )
    }
    final_sorted_df = pl.DataFrame(
        get_totals(mrn_to_earliest_date, notes_dir),
        schema=["MRN", "TOTAL_AFTER_EARLIEST"],
        orient="row",
    )
    final_sorted_df.write_csv(os.path.join(output_dir, "totals.csv"), separator=",")


def main():
    args = parser.parse_args()
    write_totals(args.pt_record_csv, args.notes_dir)


if __name__ == "__main__":
    main()
