import polars as pl
from collections import namedtuple, defaultdict
import os
import json
import argparse
from enum import Enum
from functools import cache
from collections.abc import Mapping, Sequence, Collection, Iterable
from operator import itemgetter, is_not_none
from math import floor
import datetime
import logging
import pathlib
from dateutil.parser import parse

SAMPLE_SEED = 42

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--casenum_ade_date_table",
    type=str,
    help="Excel spreadsheet (xlsx) containing case numbers, descriptions of toxicity events, and earliest dates",
)

parser.add_argument(
    "--casenum_dfci_mrn_table",
    type=str,
    help="CSV with casenum and DFCI MRNS",
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
note_dict = Mapping[str, str | int]


SIX_WEEKS = datetime.timedelta(days=42)
SAME_DAY = datetime.timedelta(days=0)


class MRNSpace(Enum):
    DFCI = "DFCI"
    EMPI = "EMPI"
    MGH = "MGH"


InterSiteMRNTuple = namedtuple("InterSiteMRNTuple", [space.value for space in MRNSpace])


def __normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


# Keep it simple by avoiding time information for now
# we can fold it back in if we need that degree of granularity
@cache
def parse_and_normalize_date(dt_str: str) -> datetime.date:
    parsed_dt = parse(dt_str, fuzzy=True)
    return parsed_dt.date()


@cache
def dates_within_range(
    pt_earliest: datetime.date,
    note_date: str | None,
    upper_bound: datetime.timedelta = SIX_WEEKS,
    lower_bound: datetime.timedelta = SAME_DAY,
) -> bool:
    if note_date is None:
        # If we don't know then rule it out
        return False
    return (
        upper_bound
        >= (parse_and_normalize_date(note_date) - pt_earliest)
        >= lower_bound
    )


def word_count_filter(
    note_json_list: Collection[note_dict],
    minimum_total_words: int = 500,
) -> Sequence[note_dict]:
    def has_minimum_total_words(
        note_json: note_dict, mininum_total_words: int = minimum_total_words
    ) -> bool:
        return len(str(note_json.get("RPT_TEXT", "")).split()) >= mininum_total_words

    return [
        note_json for note_json in note_json_list if has_minimum_total_words(note_json)
    ]


def mkdir(dir_name: str) -> None:
    _dir_name = pathlib.Path(dir_name)
    _dir_name.mkdir(parents=True, exist_ok=True)


def save_jsonl(output_dir: str, fn: str, note_json_list: Iterable[dict]) -> None:
    mkdir(output_dir)

    # Honestly can't believe Python doesn't implement this part
    def __to_line(d: dict) -> str:
        return f"{json.dumps(d)}\n"

    with open(os.path.join(output_dir, f"{fn}.jsonl"), mode="w") as f:
        f.writelines(map(__to_line, note_json_list))


def has_valid_mrn_and_date(
    mrn_to_earliest_dates: Mapping[int, Collection[tuple[datetime.date, str]]],
    note_json: note_dict,
) -> bool:
    mrn_key = "DFCI_MRN"
    mrn = int(note_json[mrn_key])
    if mrn not in mrn_to_earliest_dates:
        # invalid MRN
        return False
    note_date = note_json.get("EVENT_DATE")
    # Everything in the table has an earliest date
    # so don't need to worry about misses
    # Absent dates handled here
    return any(
        dates_within_range(pt_earliest, note_date)
        for pt_earliest, _ in mrn_to_earliest_dates[mrn]
    )


def raw_json_parse(json_path: str) -> list[note_dict]:
    with open(json_path) as f:
        return json.load(f)["response"]["docs"]


def filter_provider_types(
    note_jsons: Collection[note_dict],
) -> Sequence[note_dict]:
    inpatient_provider_types = {
        "Physician",
        "Physician Assistant",
        "Nurse Practitioner",
        "Fellow",
        "Resident",
    }
    filtered = [
        note_json
        for note_json in note_jsons
        if note_json.get("PROVIDER_TYPE") in inpatient_provider_types
    ]
    logger.info(
        f"Total in patient notes before provider types filtration: {len(note_jsons):,} - after: {len(filtered):,}"
    )
    return filtered


def get_radiation_relation_label(
    note_json: note_dict,
    mrn_to_earliest_date: Mapping[int, tuple[datetime.date, str]],
) -> str:
    # since has.. is the filter predicate we don't
    # have to worry if the MRN isn't in the table
    mrn_key = "DFCI_MRN"
    mrn = int(note_json[mrn_key])
    _, radiation_relation = mrn_to_earliest_date[mrn]
    return radiation_relation


def filter_valid_mrn_and_date_notes(
    note_type: str,
    note_dicts: Collection[note_dict],
    mrn_to_earliest_dates: Mapping[int, Collection[tuple[datetime.date, str]]],
) -> Sequence[note_dict]:
    filtered = [
        note_json
        for note_json in note_dicts
        if has_valid_mrn_and_date(
            mrn_to_earliest_dates=mrn_to_earliest_dates, note_json=note_json
        )
    ]
    logger.info(
        f"Total {note_type} notes before MRN and date filtration: {len(note_dicts):,} - after: {len(filtered):,}"
    )
    return filtered


@cache
def convert_valid_date(possible_date: str) -> datetime.date | None:
    all_components = possible_date.split("/")
    if len(all_components) != 3:
        logger.warning("Invalid date: %s", str(possible_date))
        return None
    year, month, day = all_components
    match year, month, day:
        case (
            year,
            month,
            day,
        ) if all(map(str.isnumeric, all_components)):
            result = datetime.date(year=int(year), month=int(month), day=int(day))
        case (
            year,
            month,
            "UNK",
        ) if year.isnumeric() and month.isnumeric():
            result = datetime.date(year=int(year), month=int(month), day=1)
        case _:
            logger.warning("Invalid date: %s", str(possible_date))
            result = None
    return result


def convert_and_filter_valid_dates(
    frame: pl.DataFrame,
    date_column_name: str = "NORMALIZED_DATE",
) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("DTS_DTTOXSTART1")
        # unlike with inpatient everything is accounted for
        .map_elements(convert_valid_date, return_dtype=pl.Date)
        .alias(date_column_name)
    ).filter(pl.col(date_column_name).is_not_null())


def sample_valid_dates(
    casenum_toxdesc_with_valid_dates_subframe: pl.DataFrame,
    sample_size: int = 1,
    sample_seed: int | None = SAMPLE_SEED,
) -> pl.DataFrame | None:
    if len(casenum_toxdesc_with_valid_dates_subframe) < sample_size:
        return None
    return casenum_toxdesc_with_valid_dates_subframe.sample(
        n=sample_size, seed=sample_seed
    )


def relation_category_sampling(
    frame: pl.DataFrame,
    relation_category: str = "No Relation",
    target_ratio: float = 0.25,
    sample_seed: int | None = SAMPLE_SEED,
) -> pl.DataFrame:
    others = frame.filter(pl.col("D_ATTN") != relation_category)
    remainder = 1.0 - target_ratio
    target_total = floor((1.0 / remainder) * len(others))
    relation_target = target_total - len(others)
    return pl.concat(
        (
            frame.filter(pl.col("D_ATTN") == relation_category).sample(
                n=relation_target, seed=sample_seed
            ),
            others,
        )
    )


def build_casenum_filtered_frame(
    valid_casenums: Collection[int],
    casenum_ade_date_table: str,
) -> pl.DataFrame:
    # First try grouping by toxdesc, selecting by date, then doing fractional sampling
    # by d_attn
    casenum_ade_date_frame = pl.read_excel(casenum_ade_date_table).select(
        "casenum", "TOXDESC", "D_ATTN", "DTS_DTTOXSTART1"
    )
    print(f"Before casenum filtering - total instances {len(casenum_ade_date_frame)}")
    print(casenum_ade_date_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    casenum_ade_date_frame = casenum_ade_date_frame.filter(
        pl.col("casenum").is_in(valid_casenums)
    )
    print(
        f"After valid DFCI MRN filtering - total instances {len(casenum_ade_date_frame)}"
    )
    print(casenum_ade_date_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    return casenum_ade_date_frame


def build_date_filtered_frame(frame: pl.DataFrame) -> pl.DataFrame:
    # I'll do almost any ridiculous thing
    # to appease type checkers
    date_filtered_frame = pl.concat(
        filter(
            is_not_none,
            map(
                sample_valid_dates,
                map(
                    itemgetter(1),
                    frame.group_by(
                        "casenum",
                        "TOXDESC",
                    ),
                ),
            ),
        )
    )

    print(f"After TOXDESC etc filtering - total instances {len(date_filtered_frame)}")
    print(date_filtered_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    return date_filtered_frame


def build_mrn_to_event_dates_map(
    casenum_ade_date_table: str,
    casenum_dfci_mrn_table: str,
) -> Mapping[int, Collection[tuple[datetime.date, str]]]:
    result = defaultdict(set)
    casenum_dfci_mrn_df = pl.read_csv(casenum_dfci_mrn_table)
    casenum_to_dfci_mrn_map = {
        casenum: DFCI_MRN
        for casenum, DFCI_MRN in zip(
            casenum_dfci_mrn_df["casenum"], casenum_dfci_mrn_df["DFCI_MRN"]
        )
    }

    def get_mrn(case_number: int) -> int:
        mrn = casenum_to_dfci_mrn_map.get(case_number)
        if mrn is None:
            raise ValueError(f"No MRN for {case_number} even after filtering")
        return mrn

    casenum_filtered_frame = build_casenum_filtered_frame(
        casenum_to_dfci_mrn_map.keys(), casenum_ade_date_table
    )
    filtered_frame = convert_and_filter_valid_dates(casenum_filtered_frame)
    for case_number, normalized_date, radiation_relation in zip(
        filtered_frame["casenum"],
        filtered_frame["NORMALIZED_DATE"],
        filtered_frame["D_ATTN"],
    ):
        result[get_mrn(case_number)].add((normalized_date, radiation_relation))
    return result


def collect_notes_and_write_metrics(
    casenum_ade_date_table: str,
    casenum_dfci_mrn_table: str,
    inpatient_json_path: str,
    outpatient_json_path: str,
    output_dir: str,
) -> None:
    mrn_to_earliest_dates = build_mrn_to_event_dates_map(
        casenum_ade_date_table,
        casenum_dfci_mrn_table,
    )

    all_inpatient_notes = raw_json_parse(inpatient_json_path)
    all_outpatient_notes = raw_json_parse(outpatient_json_path)
    provider_type_filtered_inpatient_notes = filter_provider_types(all_inpatient_notes)
    provider_type_filtered_outpatient_notes = filter_provider_types(
        all_outpatient_notes
    )
    mrn_date_filtered_inpatient_notes = filter_valid_mrn_and_date_notes(
        note_type="inpatient",
        note_dicts=provider_type_filtered_inpatient_notes,
        mrn_to_earliest_dates=mrn_to_earliest_dates,
    )
    mrn_date_filtered_outpatient_notes = filter_valid_mrn_and_date_notes(
        note_type="outpatient",
        note_dicts=provider_type_filtered_outpatient_notes,
        mrn_to_earliest_dates=mrn_to_earliest_dates,
    )

    def to_jsonl(note_json: note_dict) -> str:
        return json.dumps(note_json) + "\n"

    with open(
        os.path.join(output_dir, "inpatient", "filtered_inpatient.jsonl"), mode="w"
    ) as f:
        f.writelines(map(to_jsonl, mrn_date_filtered_inpatient_notes))

    with open(
        os.path.join(output_dir, "outpatient", "filtered_outpatient.jsonl"), mode="w"
    ) as f:
        f.writelines(map(to_jsonl, mrn_date_filtered_outpatient_notes))


def main():
    args = parser.parse_args()
    collect_notes_and_write_metrics(
        args.casenum_ade_date_table,
        args.casenum_dfci_mrn_table,
        args.inpatient_json_path,
        args.outpatient_json_path,
        args.output_dir,
    )


# RPT_TEXT
if __name__ == "__main__":
    main()
