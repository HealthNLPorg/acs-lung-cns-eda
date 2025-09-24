import polars as pl
import os
import json  # if proves to be too slow or memory consuming look into https://github.com/ICRAR/ijson + https://github.com/lloyd/yajl
from collections import Counter
import argparse
from itertools import chain, groupby
from functools import reduce, partial, lru_cache
from collections.abc import Iterable
from operator import itemgetter
from typing import Callable, Any
import datetime
import logging
import pathlib
from dateutil.parser import parse

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--pt_record_csv",
    type=str,
    help="CSV containing patient MRNs and earliest dates",
)

parser.add_argument(
    "--fields",
    type=str,
    nargs="+",
    default=["SUBJECT", "PROVIDER_TYPE", "SPECIALTY_NAME", "PROVIDER_DEPARTMENT"],
    help="Fields for which we want to get the totals",
)

parser.add_argument(
    "--notes_dir",
    type=str,
    help="Directory containing nested directories of notes contained in JSON files",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=".",
    help="Directory for outputting table",
)
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# Keep it simple by avoiding time information for now
# we can fold it back in if we need that degree of granularity
@lru_cache
def parse_and_normalize_date(dt_str: str) -> datetime.date:
    parsed_dt = parse(dt_str, fuzzy=True)
    return parsed_dt.date()


@lru_cache
def is_before(pt_earliest: str, note_date: str | None) -> bool:
    if note_date is None:
        return False
    return parse_and_normalize_date(pt_earliest) <= parse_and_normalize_date(note_date)


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)


def generic_sample(
    note_json_list: list[dict[str, int | str]],
    minimum_total_words: int = 500,
) -> list[dict[str, int | str]]:
    def has_minimum_total_words(
        note_json: dict[str, int | str], miminum_total_words: int = minimum_total_words
    ) -> bool:
        return note_json.get("RPT_TEXT", "").split() >= miminum_total_words


def lmr_sampling(
    note_json_list: list[dict[str, int | str]], sample_total: int = 250
) -> list[dict[str, int | str]]:
    pass


def inpatient_and_progress_sampling(
    note_json_list: list[dict[str, int | str]], sample_total: int = 250
) -> list[dict[str, int | str]]:
    pass


def has_valid_mrn_and_date(
    mrn_to_earliest_date: dict[int, str], note_json: dict[str, str | int]
) -> bool:
    mrn = int(note_json["DFCI_MRN"])
    pt_earliest = mrn_to_earliest_date.get(mrn)
    note_date = note_json.get("EVENT_DATE")
    return is_before(pt_earliest, note_date)


def get_valid_mrn_and_date_notes_from_csv(
    mrn_to_earliest_date: dict[int, str], csv_path: str
) -> list[dict[str, str | int]]:
    local_valid_mrn_and_date = partial(has_valid_mrn_and_date, mrn_to_earliest_date)
    note_json_list = pl.read_csv(csv_path).to_dicts()
    return [
        note_json for note_json in note_json_list if local_valid_mrn_and_date(note_json)
    ]


def get_valid_mrn_and_date_notes_from_json(
    mrn_to_earliest_date: dict[int, str],
    json_path: str,
    debug_source: str | None = None,
) -> list[dict[str, str | int]]:
    with open(json_path) as f:
        note_json_list = json.load(f)["response"]["docs"]
    local_valid_mrn_and_date = partial(has_valid_mrn_and_date, mrn_to_earliest_date)
    results = [
        note_json for note_json in note_json_list if local_valid_mrn_and_date(note_json)
    ]
    if debug_source is not None:
        for node_json in results:
            node_json["debug_source"] = debug_source
    return results


def get_dir_to_valid_mrn_and_date_notes(
    mrn_to_earliest_date: dict[int, str], notes_dir: str
) -> dict[str, list[dict[str, str | int]]]:
    def get_valid_mrn_and_date_notes(root: str, fn: str) -> list[dict[str, str | int]]:
        if fn.lower().endswith("json"):
            return get_valid_mrn_and_date_notes_from_json(
                mrn_to_earliest_date, os.path.join(root, fn)
            )
        elif fn.lower().endswith("csv"):
            return get_valid_mrn_and_date_notes_from_csv(
                mrn_to_earliest_date, os.path.join(root, fn)
            )
        else:
            ValueError(f"{os.path.join(root, fn)} has bad extension")
            return []

    return {
        os.path.basename(root): get_valid_mrn_and_date_notes(root, fn, root)
        for root, dirs, files in os.walk(notes_dir)
        for fn in files
    }


# NB: depending on the predicates used, there may be notes/folders left out,
# this is intentional for adjusting later
def merge_by_named_predicates(
    dir_to_valid_mrn_and_date_notes: dict[str, list[dict[str, str | int]]],
    name_to_predicate: dict[str, Callable[[str], bool]],
) -> dict[str, list[dict[str, str | int]]]:
    def grouped_by_predicate(
        predicate: Callable[[str], bool],
    ) -> Iterable[str, list[dict[str, str | int]]]:
        def key_fn(dir_and_notes: tuple[str, list[dict[str, str | int]]]) -> bool:
            dirname, _ = dir_and_notes
            return predicate(dirname)

        return groupby(
            sorted(dir_to_valid_mrn_and_date_notes.items(), key=key_fn), key=key_fn
        )

    def re_grouped_notes(
        predicate: Callable[[str], bool],
    ) -> list[dict[str, str | int]]:
        for is_group, note_clusters in grouped_by_predicate(predicate):
            if is_group:
                return list(chain.from_iterable(note_clusters))
        ValueError("Something wrong with clustering - check in more detail")
        return []

    return {
        predicate_name: re_grouped_notes(predicate)
        for predicate_name, predicate in name_to_predicate.items()
    }


def collect_notes_and_write_metrics(
    pt_record_csv: str, notes_dir: str, output_dir: str, fields: list[str]
) -> None:
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
    dir_to_valid_mrn_and_date_notes = get_dir_to_valid_mrn_and_date_notes(
        mrn_to_earliest_date, notes_dir
    )

    def is_one_of(core_names: Iterable[str]) -> Callable[[str], bool]:
        def __is_one_of(dirname: str) -> bool:
            normed = dirname.strip().lower()
            for core_name in core_names:
                if normed.startswith(core_name):
                    return True
            return False

        return __is_one_of

    # disjunctive to conjunctive terms is weird at first but makes sense in terms of human legibility
    name_to_predicate = {
        "lmr": is_one_of(("lmr",)),
        "inpatient_and_progress": is_one_of(("inpatient", "progress")),
    }
    synthetic_category_to_notes = merge_by_named_predicates(
        dir_to_valid_mrn_and_date_notes, name_to_predicate
    )


def main():
    args = parser.parse_args()
    collect_notes_and_write_metrics(
        args.pt_record_csv, args.notes_dir, args.output_dir, args.fields
    )


# RPT_TEXT
if __name__ == "__main__":
    main()
