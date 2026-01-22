import polars as pl
from collections import namedtuple
import os
import json
import argparse
from enum import Enum
from functools import lru_cache
from collections.abc import Mapping, Sequence, Collection
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
note_dict = dict[str, str | int]


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
@lru_cache
def parse_and_normalize_date(dt_str: str) -> datetime.date:
    parsed_dt = parse(dt_str, fuzzy=True)
    return parsed_dt.date()


@lru_cache
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


def has_valid_mrn_and_date(
    mrn_to_earliest_date: Mapping[int, tuple[datetime.date, str]],
    note_json: note_dict,
) -> bool:
    mrn_key = "DFCI_MRN"
    mrn = int(note_json[mrn_key])
    if mrn not in mrn_to_earliest_date:
        # invalid MRN
        return False
    pt_earliest = mrn_to_earliest_date.get(mrn)
    # Everything in the table has an earliest date
    # so don't need to worry about misses
    note_date = note_json.get("EVENT_DATE")
    # Absent dates handled here
    return dates_within_range(pt_earliest, note_date)


def raw_json_parse(json_path: str) -> list[note_dict]:
    with open(json_path) as f:
        return json.load(f)["response"]["docs"]


# etg@DIMJ0JW5Y9T3G cns_eda_workspace % rg "PROVIDER_TYPE\"" Inpatient\ Progress.json | sort | uniq -c
#   12         "PROVIDER_TYPE":"Anesthesiologist",
#   10         "PROVIDER_TYPE":"Dentist",
#    2         "PROVIDER_TYPE":"Fellow",
#    1         "PROVIDER_TYPE":"Licensed Nurse",
#   27         "PROVIDER_TYPE":"Nurse Practitioner",
#    1         "PROVIDER_TYPE":"Occupational Therapist",
#    3         "PROVIDER_TYPE":"Physical Therapist",
#  137         "PROVIDER_TYPE":"Physician Assistant",
# 2501         "PROVIDER_TYPE":"Physician",
#   30         "PROVIDER_TYPE":"Registered Nurse",
#  588         "PROVIDER_TYPE":"Resource",
#    1         "PROVIDER_TYPE":"Speech-Language Pathologist",
def filter_inpatient_provider_types(
    note_jsons: Collection[note_dict],
) -> Sequence[note_dict]:
    inpatient_provider_types = {
        "Physician",
        "Physician Assistant",
        "Nurse Practitioner",
        "Fellow",
        # These were from Danielle but
        # rg "PROVIDER_TYPE\"" Inpatient\ Progress.json | sort | uniq -c
        # didn't turn any of these up
        # if we want to use any other types they're listed above
        "Resident",
        "Intern",
        "Attending",
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


# etg@DIMJ0JW5Y9T3G cns_eda_workspace % rg "PROVIDER_TYPE\"" Outpatient\ Progress.json | sort | uniq -c
#  105         "PROVIDER_TYPE":"Acupuncturist",
#    7         "PROVIDER_TYPE":"Ancillary",
#   91         "PROVIDER_TYPE":"Anesthesiologist",
#   14         "PROVIDER_TYPE":"Audiologist",
#    3         "PROVIDER_TYPE":"Case Manager",
#    4         "PROVIDER_TYPE":"Community Health Worker",
#   49         "PROVIDER_TYPE":"Coordinator",
#    6         "PROVIDER_TYPE":"Counselor",
#   14         "PROVIDER_TYPE":"Dentist",
#    1         "PROVIDER_TYPE":"Embryologist",
#   20         "PROVIDER_TYPE":"Fellow",
#  279         "PROVIDER_TYPE":"Generic Provider",
#   24         "PROVIDER_TYPE":"Genetic Counselor",
#  147         "PROVIDER_TYPE":"Licensed Dietitian/Nutritionist",
#    9         "PROVIDER_TYPE":"Licensed Nurse",
#   10         "PROVIDER_TYPE":"Medical Assistant",
# 2882         "PROVIDER_TYPE":"Nurse Practitioner",
#  104         "PROVIDER_TYPE":"Occupational Therapist",
#    2         "PROVIDER_TYPE":"Patient Care/Nursing Assistant",
#   18         "PROVIDER_TYPE":"Pharmacist",
#    2         "PROVIDER_TYPE":"Physical Therapist Assistant",
#  317         "PROVIDER_TYPE":"Physical Therapist",
#  825         "PROVIDER_TYPE":"Physician Assistant",
# 8184         "PROVIDER_TYPE":"Physician",
#   13         "PROVIDER_TYPE":"Podiatrist",
#   14         "PROVIDER_TYPE":"Psychologist",
#   63         "PROVIDER_TYPE":"Registered Dietitian",
# 5762         "PROVIDER_TYPE":"Registered Nurse",
#    3         "PROVIDER_TYPE":"Resident",
#  161         "PROVIDER_TYPE":"Resource",
#    4         "PROVIDER_TYPE":"Respiratory Therapist",
#  952         "PROVIDER_TYPE":"Social Worker",
#  105         "PROVIDER_TYPE":"Speech-Language Pathologist",
#    2         "PROVIDER_TYPE":"Spiritual Care Student",
#   20         "PROVIDER_TYPE":"Spiritual Care",
#    4         "PROVIDER_TYPE":"Technologist",
#   21         "PROVIDER_TYPE":"Therapist",
def filter_outpatient_provider_types(
    note_jsons: Collection[note_dict],
) -> Sequence[note_dict]:
    # Same as inpatient except for "Attending" and "Intern"
    # unlike with inpatient everything is accounted for
    outpatient_provider_types = {
        "Physician",
        "Physician Assistant",
        "Nurse Practitioner",
        "Fellow",
        "Resident",
    }
    filtered = [
        note_json
        for note_json in note_jsons
        if note_json.get("PROVIDER_TYPE") in outpatient_provider_types
    ]

    logger.info(
        f"Total out patient notes before provider types filtration: {len(note_jsons):,} - after: {len(filtered):,}"
    )
    return filtered


def filter_valid_mrn_and_date_notes(
    note_type: str,
    note_dicts: Collection[note_dict],
    mrn_to_earliest_date: Mapping[int, tuple[datetime.date, str]],
) -> Sequence[note_dict]:
    filtered = [
        note_json
        for note_json in note_dicts
        if has_valid_mrn_and_date(
            mrn_to_earliest_date=mrn_to_earliest_date, note_json=note_json
        )
    ]
    logger.info(
        f"Total {note_type} notes before MRN and date filtration: {len(note_dicts):,} - after: {len(filtered):,}"
    )
    return filtered


@lru_cache
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


def sample_valid_dates(
    casenum_toxdesc_subframe: pl.DataFrame,
    sample_size: int = 1,
    date_column_name: str = "NORMALIZED_DATE",
    sample_seed: int | None = SAMPLE_SEED,
) -> pl.DataFrame | None:
    with_valid_dates = casenum_toxdesc_subframe.with_columns(
        pl.col("DTS_DTTOXSTART1")
        # unlike with inpatient everything is accounted for
        .map_elements(convert_valid_date, return_dtype=pl.Date)
        .alias(date_column_name)
    ).filter(pl.col(date_column_name).is_not_null())
    if len(with_valid_dates) < sample_size:
        return None
    return with_valid_dates.sample(n=sample_size, seed=sample_seed)


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


def build_relation_filtered_frame(
    filtered_casenum_ade_date_frame: pl.DataFrame,
) -> pl.DataFrame:
    # I'll do almost any ridiculous thing
    # to appease type checkers
    date_filtered_frame = pl.concat(
        filter(
            is_not_none,
            map(
                sample_valid_dates,
                map(
                    itemgetter(1),
                    filtered_casenum_ade_date_frame.group_by(
                        "casenum",
                        "TOXDESC",
                    ),
                ),
            ),
        )
    )

    print(f"After TOXDESC etc filtering - total instances {len(date_filtered_frame)}")
    print(date_filtered_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    relation_filtered_frame = relation_category_sampling(date_filtered_frame)
    print("Exact adverse event counts (one per patient)")
    print(relation_filtered_frame["D_ATTN"].value_counts())
    print(f"After category resampling - total instances {len(relation_filtered_frame)}")
    print(relation_filtered_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    return relation_filtered_frame


def build_mrn_to_event_date_map(
    casenum_ade_date_table: str,
    casenum_dfci_mrn_table: str,
) -> Mapping[int, tuple[datetime.date, str]]:
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
    # relation_filtered_frame = build_relation_filtered_frame(casenum_filtered_frame)
    # return {
    #     get_mrn(case_number): (normalized_date, radiation_relation)
    #     for case_number, normalized_date, radiation_relation in zip(
    #         relation_filtered_frame["casenum"],
    #         relation_filtered_frame["NORMALIZED_DATE"],
    #         relation_filtered_frame["D_ATTN"],
    #     )
    # }

    return {
        get_mrn(case_number): (normalized_date, radiation_relation)
        for case_number, normalized_date, radiation_relation in zip(
            casenum_filtered_frame["casenum"],
            casenum_filtered_frame["NORMALIZED_DATE"],
            casenum_filtered_frame["D_ATTN"],
        )
    }


def collect_notes_and_write_metrics(
    casenum_ade_date_table: str,
    casenum_dfci_mrn_table: str,
    inpatient_json_path: str,
    outpatient_json_path: str,
    output_dir: str,
) -> None:
    mrn_to_selected_date = build_mrn_to_event_date_map(
        casenum_ade_date_table,
        casenum_dfci_mrn_table,
    )

    all_inpatient_notes = raw_json_parse(inpatient_json_path)
    all_outpatient_notes = raw_json_parse(outpatient_json_path)
    provider_type_filtered_inpatient_notes = filter_inpatient_provider_types(
        all_inpatient_notes
    )
    provider_type_filtered_outpatient_notes = filter_outpatient_provider_types(
        all_outpatient_notes
    )
    mrn_date_filtered_inpatient_notes = filter_valid_mrn_and_date_notes(
        note_type="inpatient",
        note_dicts=provider_type_filtered_inpatient_notes,
        mrn_to_earliest_date=mrn_to_selected_date,
    )
    mrn_date_filtered_outpatient_notes = filter_valid_mrn_and_date_notes(
        note_type="outpatient",
        note_dicts=provider_type_filtered_outpatient_notes,
        mrn_to_earliest_date=mrn_to_selected_date,
    )
    with open(os.path.join(output_dir, "filtered_inpatient.json"), mode="w") as f:
        json.dump(mrn_date_filtered_inpatient_notes, f)

    with open(
        os.path.join(output_dir, "mrn_date_filtered_outpatient.json"), mode="w"
    ) as f:
        json.dump(mrn_date_filtered_outpatient_notes, f)


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
