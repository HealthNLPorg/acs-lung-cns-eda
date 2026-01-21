import polars as pl
from collections import namedtuple
import os
import json
import argparse
from enum import Enum
from functools import partial, lru_cache
from collections.abc import Mapping, Sequence
from operator import itemgetter
from math import floor
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
    mrn_to_earliest_date: Mapping[int, datetime.date],
    target_mrn_space: Enum,
    note_json: note_dict,
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
    return dates_within_range(pt_earliest, note_date)


def raw_json_parse(json_path: str) -> list[note_dict]:
    with open(json_path) as f:
        return json.load(f)["response"]["docs"]


def filter_valid_mrn_and_date_notes(
    mrn_to_earliest_date: Mapping[int, datetime.date],
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
    casenum_mrn_frame = (
        pl.read_excel(casenum_mrn_table)
        .select("casenum", "MRN")
        .filter(pl.col("MRN").map_elements(str.isnumeric))
    )
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
) -> pl.DataFrame | None:
    with_valid_dates = casenum_toxdesc_subframe.with_columns(
        pl.col("DTS_DTTOXSTART1")
        .map_elements(convert_valid_date, return_dtype=pl.Date)
        .alias(date_column_name)
    ).filter(pl.col(date_column_name).is_not_null())
    if len(with_valid_dates) < sample_size:
        return None
    return with_valid_dates.sample(n=sample_size)


def relation_category_sampling(
    frame: pl.DataFrame,
    relation_category: str = "No Relation",
    target_ratio: float = 0.25,
) -> pl.DataFrame:
    others = frame.filter(pl.col("D_ATTN") != relation_category)
    remainder = 1.0 - target_ratio
    target_total = floor((1.0 / remainder) * len(others))
    relation_target = target_total - len(others)
    return pl.concat(
        (
            frame.filter(pl.col("D_ATTN") == relation_category),
            others.sample(n=relation_target),
        )
    )


def build_case_number_to_event_date_map(
    casenum_ade_date_table: str,
) -> Mapping[int, datetime.date]:
    # First try grouping by toxdesc, selecting by date, then doing fractional sampling
    # by d_attn
    casenum_ade_date_frame = pl.read_excel(casenum_ade_date_table).select(
        "casenum", "TOXDESC", "D_ATTN", "DTS_DTTOXSTART1"
    )
    # I'll do almost any ridiculous thing
    # to appease type checkers
    date_filtered_frame = pl.concat(
        filter(
            lambda s: s is not None,
            map(
                sample_valid_dates,
                map(
                    itemgetter(1),
                    casenum_ade_date_frame.group_by(
                        "casenum",
                        "TOXDESC",
                    ),
                ),
            ),
        )
    )
    relation_filtered_frame = relation_category_sampling(date_filtered_frame)
    return {
        case_number: normalized_date
        for case_number, normalized_date in zip(
            relation_filtered_frame["casenum"],
            relation_filtered_frame["NORMALIZED_DATE"],
        )
    }


def build_mrn_to_raw_event_date_map(
    casenum_ade_date_table: str,
    inter_site_mrn_table: str,
    casenum_mrn_table: str,
) -> tuple[Mapping[int, datetime.date], Enum]:
    case_number_to_event_date_map = build_case_number_to_event_date_map(
        casenum_ade_date_table
    )
    case_number_to_raw_mrn_map = build_case_number_to_raw_mrn_map(casenum_mrn_table)
    mrn_tuples = get_inter_site_mrn_tuples(inter_site_mrn_table)
    dfci_mrns = {
        mrn_tuple.DFCI for mrn_tuple in mrn_tuples if mrn_tuple.DFCI is not None
    }
    empi_mrns = {
        mrn_tuple.EMPI for mrn_tuple in mrn_tuples if mrn_tuple.EMPI is not None
    }
    mgh_mrns = {mrn_tuple.MGH for mrn_tuple in mrn_tuples if mrn_tuple.MGH is not None}
    unique_mrns = set(case_number_to_raw_mrn_map.values())
    print(", ".join(f"{x:_}" for x in sorted(unique_mrns)[:10]))
    print(", ".join(f"{x:_}" for x in sorted(dfci_mrns)[:10]))
    print(", ".join(f"{x:_}" for x in sorted(empi_mrns)[:10]))
    print(", ".join(f"{x:_}" for x in sorted(mgh_mrns)[:10]))
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
    filtered_inpatient_notes = filter_valid_mrn_and_date_notes(
        mrn_to_earliest_date=mrn_to_selected_date,
        target_mrn_space=target_mrn_space,
        json_path=inpatient_json_path,
    )
    filtered_outpatient_notes = filter_valid_mrn_and_date_notes(
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
