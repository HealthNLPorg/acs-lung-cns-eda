import polars as pl
import os
import json  # if proves to be too slow or memory consuming look into https://github.com/ICRAR/ijson + https://github.com/lloyd/yajl
import argparse
import random
from functools import partial, lru_cache
from itertools import chain
from collections.abc import Iterable
from typing import Callable, cast
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
note_dict = dict[str, str | int]


def __normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


# Keep it simple by avoiding time information for now
# we can fold it back in if we need that degree of granularity
@lru_cache
def parse_and_normalize_date(dt_str: str) -> datetime.date:
    parsed_dt = parse(dt_str, fuzzy=True)
    return parsed_dt.date()


@lru_cache
def is_before(pt_earliest: str, note_date: str | None) -> bool:
    if note_date is None:
        # If we don't know then rule it out
        return False
    return parse_and_normalize_date(pt_earliest) <= parse_and_normalize_date(note_date)


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)


def save_jsonl(output_dir: str, fn: str, note_json_list: list[dict]) -> None:
    mkdir(output_dir)

    # Honestly can't believe Python doesn't implement this part
    def __to_line(d: dict) -> str:
        return f"{json.dumps(d)}\n"

    with open(os.path.join(output_dir, f"{fn}.jsonl"), mode="w") as f:
        f.writelines(map(__to_line, note_json_list))


def word_count_filter(
    note_json_list: list[dict[str, int | str]],
    minimum_total_words: int = 500,
) -> list[dict[str, int | str]]:
    def has_minimum_total_words(
        note_json: dict[str, int | str], mininum_total_words: int = minimum_total_words
    ) -> bool:
        return len(str(note_json.get("RPT_TEXT", "")).split()) >= mininum_total_words

    return [
        note_json for note_json in note_json_list if has_minimum_total_words(note_json)
    ]


def lmr_provider_type_and_specialty_filter(
    note_json_list: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    relevant_provider_types = {
        "attending",
        "physician",
        "nurse practitioner",
        "physician assistant",
        "resident",
        "fellow",
        "intern",
    }
    relevant_specialty_names = {
        "oncology",
        "radiation oncology",
        "internal medicine",
        "cardiology",
        "hematology/oncology",
        "surgery",
        "thoracic",
    }

    def __has_relevant_provider_type(
        note_json: dict[str, int | str],
        relevant_provider_types: set[str] = relevant_provider_types,
    ) -> bool:
        return (
            __normalize(cast(str, note_json.get("PROVIDER_TYPE", "")))
            in relevant_provider_types
        )

    def __has_relevant_specialty_name(
        note_json: dict[str, int | str],
        relevant_specialty_names: set[str] = relevant_specialty_names,
    ) -> bool:
        return (
            __normalize(cast(str, note_json.get("SPECIALTY_NAME", "")))
            in relevant_specialty_names
        )

    def __has_criteria(
        note_json: dict[str, int | str],
        relevant_provider_types: set[str] = relevant_provider_types,
        relevant_specialty_names: set[str] = relevant_specialty_names,
    ) -> bool:
        return __has_relevant_provider_type(
            note_json, relevant_provider_types
        ) and __has_relevant_specialty_name(note_json, relevant_specialty_names)

    result = [note_json for note_json in note_json_list if __has_criteria(note_json)]
    logger.info(
        f"Total LMR notes before provider type and specialty name filtration: {len(note_json_list)} - after: {len(result)}"
    )
    return result


def inpatient_and_progress_provider_filter(
    note_json_list: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    relevant_provider_types = {
        "physician",
        "nurse practitioner",
        "physician assistant",
        "resident",
        "fellow",
    }

    def __has_relevant_provider_type(
        note_json: dict[str, int | str],
        relevant_provider_types: set[str] = relevant_provider_types,
    ) -> bool:
        return (
            __normalize(cast(str, note_json.get("PROVIDER_TYPE", "")))
            in relevant_provider_types
        )

    result = [
        note_json
        for note_json in note_json_list
        if __has_relevant_provider_type(note_json)
    ]
    logger.info(
        f"Total Inpatient+Progress notes before provider type filtration: {len(note_json_list)} - after: {len(result)}"
    )
    return result


def has_valid_mrn_and_date(
    mrn_to_earliest_date: dict[int, str], note_json: note_dict
) -> bool:
    mrn = int(note_json["DFCI_MRN"])
    if mrn not in mrn_to_earliest_date:
        # invalid MRN
        return False
    pt_earliest = mrn_to_earliest_date.get(mrn)
    # Everything in the table has an earliest date
    # so don't need to worry about misses
    note_date = note_json.get("EVENT_DATE")
    # Absent dates handled here
    return is_before(pt_earliest, note_date)


def raw_csv_parse(csv_path: str) -> list[note_dict]:
    return pl.read_csv(csv_path).to_dicts()


def get_valid_mrn_and_date_notes_from_csv(
    mrn_to_earliest_date: dict[int, str],
    csv_path: str,
    debug_source: str | None = None,
) -> list[note_dict]:
    local_valid_mrn_and_date = partial(has_valid_mrn_and_date, mrn_to_earliest_date)
    note_json_list = pl.read_csv(csv_path).to_dicts()

    result = [
        note_json for note_json in note_json_list if local_valid_mrn_and_date(note_json)
    ]
    logger.info(
        f"Total {debug_source}  notes before MRN and date filtration: {len(note_json_list)} - after: {len(result)}"
    )
    if debug_source is not None:
        for node_json in result:
            node_json["debug_source"] = debug_source
    return result


def raw_json_parse(json_path: str) -> list[note_dict]:
    with open(json_path) as f:
        return json.load(f)["response"]["docs"]


def get_valid_mrn_and_date_notes_from_json(
    mrn_to_earliest_date: dict[int, str],
    json_path: str,
    debug_source: str | None = None,
) -> list[note_dict]:
    with open(json_path) as f:
        note_json_list = json.load(f)["response"]["docs"]
    local_valid_mrn_and_date = partial(has_valid_mrn_and_date, mrn_to_earliest_date)
    result = [
        note_json for note_json in note_json_list if local_valid_mrn_and_date(note_json)
    ]
    logger.info(
        f"Total {debug_source}  notes before MRN and date filtration: {len(note_json_list)} - after: {len(result)}"
    )
    if debug_source is not None:
        for node_json in result:
            node_json["debug_source"] = debug_source
    return result


def identify_keys_with_unique_values(
    unique_id_debug_dict: dict[str, list[note_dict]],
) -> None:
    @lru_cache
    def __norm_key(k: str) -> str:
        return k.strip().lower()

    def __is_index_key(k: str) -> bool:
        norm_k = __norm_key(k)
        return "mrn" in norm_k or "id" in norm_k

    def __is_report_key(k: str) -> bool:
        norm_k = __norm_key(k)
        return "rpt" in norm_k

    collected_note_dicts = [
        _note_dict for ls in unique_id_debug_dict.values() for _note_dict in ls
    ]
    all_keys = {
        k
        for k in set(
            chain.from_iterable(
                _note_dict.keys() for _note_dict in collected_note_dicts
            )
        )
    }
    indexing_keys = set(filter(__is_index_key, all_keys))
    for indexing_key in indexing_keys:

        def __local_get(_note_dict: note_dict) -> str | None:
            result = _note_dict.get(indexing_key)
            if result is not None:
                return str(result)
            return result

        collected_values = list(filter(None, map(__local_get, collected_note_dicts)))
        unique_values = set(collected_values)
        if len(collected_values) == len(unique_values):
            logger.info(
                f"{indexing_key} has unique values, total: {len(unique_values)}"
            )
        else:
            logger.info(
                f"{indexing_key} values not unique, total values {len(collected_values)} unique values {len(unique_values)}"
            )

    report_keys = sorted(filter(__is_report_key, all_keys))
    logger.info(f"Report keys:\n{report_keys}")


def get_dir_to_valid_mrn_and_date_notes(
    mrn_to_earliest_date: dict[int, str], notes_dir: str, relevant_dirs: set[str]
) -> dict[str, list[note_dict]]:
    def is_relevant(dirname) -> bool:
        for relevant_dir in relevant_dirs:
            if dirname.lower().startswith(relevant_dir):
                return True
        return False

    def get_valid_mrn_and_date_notes(
        root: str, files: list[str]
    ) -> Iterable[note_dict]:
        for fn in files:
            if fn.lower().endswith("json"):
                yield from raw_json_parse(os.path.join(root, fn))
            elif fn.lower().endswith("csv"):
                yield from raw_csv_parse(os.path.join(root, fn))
            else:
                raise ValueError(f"{os.path.join(root, fn)} has bad extension")

    unique_id_debug_dict = {
        os.path.basename(root): list(get_valid_mrn_and_date_notes(root, files))
        for root, dirs, files in os.walk(notes_dir)
        # if is_relevant(os.path.basename(root))
    }
    identify_keys_with_unique_values(unique_id_debug_dict)
    initial = {
        dirname: note_dicts
        for dirname, note_dicts in unique_id_debug_dict.items()
        if is_relevant(dirname)
    }

    local_valid_mrn_and_date = partial(has_valid_mrn_and_date, mrn_to_earliest_date)
    final = {}
    for dirname, all_notes in initial.items():
        filtered = [
            note_json for note_json in all_notes if local_valid_mrn_and_date(note_json)
        ]
        logger.info(
            f"Total {dirname}  notes before MRN and date filtration: {len(all_notes)} - after: {len(filtered)}"
        )
        final[dirname] = filtered
    return final


# NB: depending on the predicates used, there may be notes/folders left out,
# this is intentional for adjusting later
def merge_by_named_predicates(
    dir_to_valid_mrn_and_date_notes: dict[str, list[note_dict]],
    name_to_predicate: dict[str, Callable[[str], bool]],
) -> dict[str, list[note_dict]]:
    def re_grouped_notes(
        predicate: Callable[[str], bool],
    ) -> Iterable[note_dict]:
        for dirname, notes in dir_to_valid_mrn_and_date_notes.items():
            if predicate(dirname):
                yield from notes

    return {
        predicate_name: list(re_grouped_notes(predicate))
        for predicate_name, predicate in name_to_predicate.items()
    }


def collect_notes_and_write_metrics(
    pt_record_csv: str,
    notes_dir: str,
    output_dir: str,
    fields: list[str],
    subsample_total: int = 250,
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
    dir_to_valid_mrn_and_date_notes = get_dir_to_valid_mrn_and_date_notes(
        mrn_to_earliest_date, notes_dir, {"lmr", "inpatient", "progress"}
    )

    name_to_initial_filter = {
        "lmr": lmr_provider_type_and_specialty_filter,
        "inpatient_and_progress": inpatient_and_progress_provider_filter,
    }
    synthetic_category_to_notes = merge_by_named_predicates(
        dir_to_valid_mrn_and_date_notes, name_to_predicate
    )
    synthetic_category_to_title = {
        "lmr": "LMR",
        "inpatient_and_progress": "Inpatient+Progress",
    }
    mkdir(output_dir)
    for synthetic_category, notes in synthetic_category_to_notes.items():
        initial_filter = name_to_initial_filter.get(synthetic_category)
        if initial_filter is None:
            logger.info("Skipping %s", synthetic_category)
            continue
        initial_filtered = initial_filter(notes)
        save_jsonl(
            os.path.join(output_dir, "before_word_count_filter"),
            synthetic_category,
            initial_filtered,
        )
        word_count_filtered = word_count_filter(initial_filtered)
        logger.info(
            f"{synthetic_category_to_title[synthetic_category]} total after word count filtration - {len(word_count_filtered)}"
        )
        save_jsonl(
            os.path.join(output_dir, "after_word_count_filter"),
            synthetic_category,
            word_count_filtered,
        )
        save_jsonl(
            os.path.join(output_dir, f"abbrev_{subsample_total}"),
            synthetic_category,
            random.sample(word_count_filtered, subsample_total),
        )


def main():
    args = parser.parse_args()
    collect_notes_and_write_metrics(
        args.pt_record_csv, args.notes_dir, args.output_dir, args.fields
    )


# RPT_TEXT
if __name__ == "__main__":
    main()
