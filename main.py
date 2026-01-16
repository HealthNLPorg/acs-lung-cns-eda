import polars as pl
from collections import namedtuple
import os
import json
import argparse
import random
from enum import Enum
from functools import partial, lru_cache
from itertools import chain
from collections.abc import Iterable, Mapping, Sequence
from typing import cast
import datetime
import logging
import pathlib
from dateutil.parser import parse

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--casenum_ade_date_table",
    type=str,
    help="Excel spreadsheet (xlsx) containing case numbers, descriptions of toxicity events, and earliest dates",
)

parser.add_argument(
    "--inter_site_mrn_table",
    type=str,
    help="CSV containing patient names coordinated with corresponding MRNs (if any) from MGB, EMPI, and MGH",
)

parser.add_argument(
    "--casenum_mrn_table",
    type=str,
    help="Excel spreadsheet (xlsx) containing case numbers coordinated with names and MRNs from some site.  Which site?  Let's find out!",
)

parser.add_argument(
    "--fields",
    type=str,
    nargs="+",
    default=["SUBJECT", "PROVIDER_TYPE", "SPECIALTY_NAME", "PROVIDER_DEPARTMENT"],
    help="Fields for which we want to get the totals",
)

parser.add_argument(
    "--inpatient_json_path",
    type=str,
    help="In patient JSON",
)

parser.add_argument(
    "--outpatient_json_path",
    type=str,
    help="Out patient JSON",
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


class MRNSpace(Enum):
    DFCI = "DFCI"
    EMPI = "EMPI"
    MGH = "MGH"


InterSiteMRNTuple = namedtuple("InterSiteMRNTuple", [space.value for space in MRNSpace])


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
    mrn_to_earliest_date: dict[int, str], target_mrn_space: Enum, note_json: note_dict
) -> bool:
    match target_mrn_space:
        case MRNSpace.DFCI:
            mrn_key = "DFCI_MRN"
        case _:
            raise NotImplementedError(
                "Turns out it wasn't DFCI. Need to find the right MRN key"
            )
    mrn = int(note_json[mrn_key])
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
    mrn_to_earliest_date: Mapping[int, str],
    target_mrn_space: Enum,
    json_path: str,
) -> Sequence[note_dict]:
    all_notes = raw_json_parse(json_path)
    local_valid_mrn_and_date = partial(
        has_valid_mrn_and_date, mrn_to_earliest_date, target_mrn_space
    )
    filtered = [
        note_json for note_json in all_notes if local_valid_mrn_and_date(note_json)
    ]
    logger.info(
        f"Total {json_path} notes before MRN and date filtration: {len(all_notes)} - after: {len(filtered)}"
    )
    return filtered


def build_case_number_to_raw_mrn_map(
    casenum_mrn_table: str,
) -> Mapping[int, int]:
    casenum_mrn_frame = pl.read_excel(casenum_mrn_table).select("casenum", "MRN")
    return {
        casenum: mrn
        for casenum, mrn in zip(
            casenum_mrn_frame["casenum"].cast(pl.Int64),
            casenum_mrn_frame["MRN"].cast(pl.Int64),
        )
    }


def get_inter_site_mrn_tuples(inter_site_mrn_table: str) -> set[InterSiteMRNTuple]:
    def row_dict_to_named_tuple(
        row_dict: Mapping[str, int | None],
    ) -> InterSiteMRNTuple:
        return InterSiteMRNTuple(
            row_dict.get("DFCI_MRN"),
            row_dict.get("EMPI"),
            row_dict.get("MGH_MRN"),
        )

    inter_site_mrn_frame = (
        pl.read_csv(inter_site_mrn_table)
        .select("DFCI_MRN", "EMPI", "MGH_MRN")
        .filter(~pl.all_horizontal(pl.all().is_null()))
    )
    return {
        row_dict_to_named_tuple(row_dict)
        for row_dict in inter_site_mrn_frame.to_dicts()
    }


def clean_possible_dates(possible_dates: Iterable[str | None]) -> list[str]:
    def clean_date(possible_date: str) -> str:
        year, month, day = possible_date.split("/")
        if year.isnumeric() and month.isnumeric():
            if day.isnumeric():
                return possible_date
            elif day == "UNK":
                return f"{year}/{month}/1"
        else:
            raise ValueError(f"Incorrectly formatted date - {possible_date}")
        return "ERROR"

    return list(map(clean_date, filter(None, possible_dates)))


def build_case_number_to_event_date_map(
    casenum_ade_date_table: str,
    # not parsing to datetime.date yet, that's downstream
) -> Mapping[int, str]:
    case_number_to_event_dates = {}
    casenum_ade_date_frame = pl.read_excel(casenum_ade_date_table).select(
        "casenum", "TOXDESC", "DTS_DTTOXSTART1"
    )
    for (casenum, _), sub_frame in casenum_ade_date_frame.group_by(
        "casenum",
        "TOXDESC",
    ):
        possible_dates = clean_possible_dates(sub_frame["DTS_DTTOXSTART1"])
        if len(possible_dates) > 0:
            case_number_to_event_dates[int(casenum)] = random.choice(possible_dates)
    return case_number_to_event_dates


def build_mrn_to_raw_event_date_map(
    casenum_ade_date_table: str,
    inter_site_mrn_table: str,
    casenum_mrn_table: str,
) -> tuple[Mapping[int, str], Enum]:
    mrn_tuples = get_inter_site_mrn_tuples(inter_site_mrn_table)
    dfci_mrns = {mrn_tuple.DFCI for mrn_tuple in mrn_tuples}
    empi_mrns = {mrn_tuple.EMPI for mrn_tuple in mrn_tuples}
    mgh_mrns = {mrn_tuple.MGH for mrn_tuple in mrn_tuples}
    case_number_to_raw_mrn_map = build_case_number_to_raw_mrn_map(casenum_mrn_table)
    unique_mrns = set(case_number_to_raw_mrn_map.values())
    missing_in_dfci = len(unique_mrns - dfci_mrns)
    missing_in_empi = len(unique_mrns - empi_mrns)
    missing_in_mgh = len(unique_mrns - mgh_mrns)
    space_to_missing = {
        MRNSpace.DFCI.value: missing_in_dfci,
        MRNSpace.EMPI.value: missing_in_empi,
        MRNSpace.MGH.value: missing_in_mgh,
    }
    covered_spaces = {
        space: missing for space, missing in space_to_missing.items() if missing == 0
    }
    target_space = None
    match len(covered_spaces):
        case 1:
            target_space = MRNSpace(next(iter(covered_spaces.keys())))
            logger.info("Using %s for MRNs", target_space)
        case 0:
            raise ValueError(
                f"None of {', '.join(sorted(covered_spaces.keys()))} are covered"
            )
        case _:
            raise ValueError(
                f"More than one of {', '.join(sorted(covered_spaces.keys()))} are covered"
            )
    case_number_to_event_date_map = build_case_number_to_event_date_map(
        casenum_ade_date_table
    )
    mrn_to_event_dates_map = {}
    for case_number, event_date in case_number_to_event_date_map.items():
        # case number not included if not enough MRNs
        # so not worrying about those
        mrn = case_number_to_raw_mrn_map.get(case_number)
        if mrn is not None:
            mrn_to_event_dates_map[mrn] = event_date
    return mrn_to_event_dates_map, target_space


def collect_notes_and_write_metrics(
    # pt_record_csv: str,
    casenum_ade_date_table: str,
    inter_site_mrn_table: str,
    casenum_mrn_table: str,
    inpatient_json_path: str,
    outpatient_json_path: str,
    output_dir: str,
    fields: list[str],
    subsample_total: int = 250,
) -> None:
    mrn_to_selected_date, target_mrn_space = build_mrn_to_raw_event_date_map(
        casenum_ade_date_table,
        inter_site_mrn_table,
        casenum_mrn_table,
    )
    filtered_inpatient_notes = get_dir_to_valid_mrn_and_date_notes(
        mrn_to_earliest_date=mrn_to_selected_date,
        target_mrn_space=target_mrn_space,
        json_path=inpatient_json_path,
    )
    filtered_outpatient_notes = get_dir_to_valid_mrn_and_date_notes(
        mrn_to_earliest_date=mrn_to_selected_date,
        target_mrn_space=target_mrn_space,
        json_path=outpatient_json_path,
    )
    with open(os.path.join(output_dir, "filtered_inpatient.json"), mode="w") as f:
        json.dump(filtered_inpatient_notes, f)

    with open(os.path.join(output_dir, "filtered_outpatient.json"), mode="w") as f:
        json.dump(filtered_outpatient_notes, f)


def main():
    args = parser.parse_args()
    collect_notes_and_write_metrics(
        # args.pt_record_csv,
        args.casenum_ade_date_table,
        args.inter_site_mrn_table,
        args.casenum_mrn_table,
        args.inpatient_json_path,
        args.outpatient_json_path,
        args.output_dir,
        args.fields,
    )


# RPT_TEXT
if __name__ == "__main__":
    main()
