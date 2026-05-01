from itertools import chain
import random
import polars as pl
from collections import defaultdict, Counter
import os
import json
import argparse
import string
from tabulate import tabulate
from more_itertools import one, all_unique
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
    "--inpatient_provider_departments_path",
    type=str,
    help="In patient PROVIDER_DEPARTMENTS",
)

parser.add_argument(
    "--outpatient_provider_departments_path",
    type=str,
    help="Out patient PROVIDER_DEPARTMENTS",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=".",
    help="Directory for outputting table",
)

parser.add_argument(
    "--filter_to_single_date",
    action="store_true",
    help="Starting at the beginning",
)
parser.add_argument(
    "--stratify_beginning",
    action="store_true",
    help="Starting at the beginning",
)
parser.add_argument(
    "--stratify_end",
    action="store_true",
    help="Starting at the beginning",
)
parser.add_argument(
    "--sample_size",
    type=int,
    default=500,
    help="Out patient PROVIDER_DEPARTMENTS",
)
parser.add_argument("--filter_by_word_count", action="store_true")
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
note_dict = Mapping[str, str | int]


SIX_WEEKS_AFTER = datetime.timedelta(days=42)
TWO_WEEKS_BEFORE = datetime.timedelta(days=-14)


@cache
def relevant_unicode_category(category: str) -> bool:
    return category != "So" and not category.startswith("C")


@cache
def relevant_character(char: str) -> bool:
    return char in string.ascii_letters + string.digits + string.punctuation + " "


def correct_note_text(note: note_dict, text_key: str = "RPT_TEXT") -> note_dict:
    raw = note.get(text_key)
    if not isinstance(raw, str):
        raise ValueError(f"Missing note text for {text_key} and {note['DFCI_MRN']}")
    cleaned = "".join(filter(relevant_character, raw))
    if cleaned != raw:
        logger.warning("Problematic source for note: %s", note["RPT_ID"])
    return {k: v if k != text_key else cleaned for k, v in note.items()}
    # return note


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
    upper_bound: datetime.timedelta = SIX_WEEKS_AFTER,
    lower_bound: datetime.timedelta = TWO_WEEKS_BEFORE,
) -> bool:
    if note_date is None:
        # If we don't know then rule it out
        return False
    return (
        upper_bound
        >= (parse_and_normalize_date(note_date) - pt_earliest)
        >= lower_bound
    )


def table_to_provider_departments(table_path: str) -> Collection[str]:
    df = pl.read_csv(table_path)
    return {
        provider_department
        for provider_department in df.filter(pl.col("retain") == 1)
        .select("PROVIDER_DEPARTMENT_STR")
        .to_series()
    }


def print_totals(
    note_jsons: Iterable[note_dict],
    key: str,
    stage: str,
    first_n: int | None = 10,
    out_path: str | None = None,
) -> None:
    totals_by_key = Counter(note_json.get(key) for note_json in note_jsons)
    if out_path is None:
        if first_n is None:
            print(stage)
            print(
                tabulate(
                    totals_by_key.most_common(),
                    headers=[" ".join(key.split("_")).title(), "Total"],
                )
            )
            return
        print(stage)
        print(
            tabulate(
                chain(
                    totals_by_key.most_common()[: min(len(totals_by_key), first_n)],
                    [("...", "...")],
                ),
                headers=[" ".join(key.split("_")).title(), "Total"],
            )
        )
    else:
        pl.DataFrame(
            schema=[(key, pl.String), ("Total", pl.Int64)],
            data=totals_by_key.most_common(),
        ).write_csv(out_path)


def word_count_filter(
    note_jsons: Collection[note_dict],
    source: str,
    minimum_total_words: int = 500,
) -> Sequence[note_dict]:
    def has_minimum_total_words(
        note_json: note_dict, mininum_total_words: int = minimum_total_words
    ) -> bool:
        return len(str(note_json.get("RPT_TEXT", "")).split()) >= mininum_total_words

    filtered = [
        note_json for note_json in note_jsons if has_minimum_total_words(note_json)
    ]
    logger.info(
        f"Total {source} notes before minimum of {minimum_total_words} words filtration: {len(note_jsons):,} - after: {len(filtered):,}"
    )
    return filtered


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


def filter_provider_departments(
    note_jsons: Collection[note_dict],
    relevant_provider_departments: Collection[str],
    source: str,
) -> Sequence[note_dict]:
    filtered = [
        note_json
        for note_json in note_jsons
        if note_json.get("PROVIDER_DEPARTMENT_STR") in relevant_provider_departments
    ]
    logger.info(
        f"Total {source} notes before provider departments filtration: {len(note_jsons):,} - after: {len(filtered):,}"
    )
    return filtered


def filter_provider_types(
    note_jsons: Collection[note_dict], source: str
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
        f"Total {source} notes before provider types filtration: {len(note_jsons):,} - after: {len(filtered):,}"
    )
    return filtered


def get_radiation_relation_label(
    note_json: note_dict,
    mrn_to_earliest_dates: Mapping[int, Collection[tuple[datetime.date, str]]],
) -> str:
    # since has.. is the filter predicate we don't
    # have to worry if the MRN isn't in the table
    mrn_key = "DFCI_MRN"
    mrn = int(note_json[mrn_key])
    try:
        _, radiation_relation = one(
            mrn_to_earliest_dates[mrn], too_long=ValueError, too_short=IndexError
        )
        return radiation_relation
    except IndexError:
        raise ValueError(f"Missing date and radiation relation for {mrn}")
    except ValueError:
        raise ValueError(
            f"{len(mrn_to_earliest_dates[mrn])} date/radiation relation pairs for {mrn}, need to filter to single date before this point"
        )


def collection_relation_category_sampling(
    notes: Collection[tuple[note_dict, str]],
    relation_category: str = "No Relation",
    target_ratio: float = 0.25,
    sample_seed: int | None = SAMPLE_SEED,
) -> Sequence[note_dict]:
    others = [
        note
        for note, radiation_relation in notes
        if radiation_relation != relation_category
    ]
    remainder = 1.0 - target_ratio
    target_total = floor((1.0 / remainder) * len(others))
    relation_target = target_total - len(others)
    return list(
        chain(
            others,
            random.sample(
                [
                    note
                    for note, radiation_relation in notes
                    if radiation_relation == relation_category
                ],
                k=relation_target,
            ),
        )
    )


def stratification(
    notes: Collection[tuple[note_dict, str]], stratify_end: bool
) -> Sequence[note_dict]:
    if not stratify_end:
        return list(map(itemgetter(0), notes))
    return collection_relation_category_sampling(notes)


def filter_valid_mrn_and_date_notes(
    note_type: str,
    note_dicts: Collection[note_dict],
    mrn_to_earliest_dates: Mapping[int, Collection[tuple[datetime.date, str]]],
    stratify_end: bool,
) -> Sequence[note_dict]:
    filtered = [
        (
            note_json,
            get_radiation_relation_label(
                note_json=note_json, mrn_to_earliest_dates=mrn_to_earliest_dates
            ),
        )
        for note_json in note_dicts
        if has_valid_mrn_and_date(
            mrn_to_earliest_dates=mrn_to_earliest_dates, note_json=note_json
        )
    ]
    result = stratification(notes=filtered, stratify_end=stratify_end)
    logger.info(
        f"Total {note_type} notes before MRN and date filtration: {len(note_dicts):,} - after: {len(result):,}"
    )
    return result


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


def build_mrn_filtered_frame(
    casenum_ade_date_table: str,
    casenum_to_mrn: Mapping[int, int],
) -> pl.DataFrame:
    # First try grouping by toxdesc, selecting by date, then doing fractional sampling
    # by d_attn
    casenum_ade_date_frame = pl.read_excel(casenum_ade_date_table).select(
        "casenum", "TOXDESC", "D_ATTN", "DTS_DTTOXSTART1"
    )
    print(f"Before casenum filtering - total instances {len(casenum_ade_date_frame)}")
    print(casenum_ade_date_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    mrn_ade_date_frame = casenum_ade_date_frame.filter(
        pl.col("casenum").is_in(casenum_to_mrn.keys())
    ).with_columns(pl.col("casenum").replace_strict(casenum_to_mrn).alias("MRN"))
    print(f"After valid DFCI MRN filtering - total instances {len(mrn_ade_date_frame)}")
    print(mrn_ade_date_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    return mrn_ade_date_frame


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

    print(f"After date filtering - total instances {len(date_filtered_frame)}")
    print(date_filtered_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    return date_filtered_frame


def build_relation_filtered_frame(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    print(f"Before category resampling - total instances {len(frame)}")
    relation_filtered_frame = relation_category_sampling(frame)
    print("Exact adverse event counts (one per patient)")
    print(relation_filtered_frame["D_ATTN"].value_counts())
    print(f"After category resampling - total instances {len(relation_filtered_frame)}")
    print(relation_filtered_frame["D_ATTN"].value_counts(normalize=True, sort=True))
    return relation_filtered_frame


def build_mrn_to_event_dates_map(
    casenum_ade_date_table: str,
    casenum_dfci_mrn_table: str,
    filter_to_single_date: bool,
    stratify_relations: bool,
) -> Mapping[int, Collection[tuple[datetime.date, str]]]:
    result = defaultdict(set)
    casenum_dfci_mrn_df = pl.read_csv(casenum_dfci_mrn_table)
    casenum_to_dfci_mrn_map = {
        casenum: DFCI_MRN
        for casenum, DFCI_MRN in zip(
            casenum_dfci_mrn_df["casenum"], casenum_dfci_mrn_df["DFCI_MRN"]
        )
    }

    mrn_filtered_frame = build_mrn_filtered_frame(
        casenum_ade_date_table=casenum_ade_date_table,
        casenum_to_mrn=casenum_to_dfci_mrn_map,
    )
    filtered_frame = convert_and_filter_valid_dates(mrn_filtered_frame)

    if filter_to_single_date:
        filtered_frame = build_date_filtered_frame(filtered_frame)
    if stratify_relations:
        filtered_frame = build_relation_filtered_frame(filtered_frame)
    for mrn, normalized_date, radiation_relation in zip(
        filtered_frame["MRN"],
        filtered_frame["NORMALIZED_DATE"],
        filtered_frame["D_ATTN"],
    ):
        if mrn not in result:
            result[mrn].add((normalized_date, radiation_relation))
    # if stratify_relations and filter_to_single_date:
    #     exit_early = False
    #     for k, v in result.items():
    #         if len(v) > 1:
    #             logger.error(f"WHATS HAPPENING {k} {v}")
    #             exit_early = True
    #     if exit_early:
    #         print(filter_to_single_date)
    #         print(stratify_relations)
    #         exit(1)
    return result


def collect_notes_and_write_metrics(
    casenum_ade_date_table: str,
    casenum_dfci_mrn_table: str,
    inpatient_json_path: str,
    outpatient_json_path: str,
    inpatient_provider_departments_path: str,
    outpatient_provider_departments_path: str,
    filter_to_single_date: bool,
    stratify_beginning: bool,
    stratify_end: bool,
    output_dir: str,
    filter_by_word_count: bool,
    sample_size: int,
) -> None:
    inpatient_provider_departments = table_to_provider_departments(
        inpatient_provider_departments_path
    )
    outpatient_provider_departments = table_to_provider_departments(
        outpatient_provider_departments_path
    )

    mrn_to_earliest_dates = build_mrn_to_event_dates_map(
        casenum_ade_date_table,
        casenum_dfci_mrn_table,
        filter_to_single_date=filter_to_single_date,
        stratify_relations=stratify_beginning,
    )

    all_inpatient_notes = raw_json_parse(inpatient_json_path)
    all_outpatient_notes = raw_json_parse(outpatient_json_path)
    print_totals(
        all_outpatient_notes, key="PROVIDER_DEPARTMENT_STR", stage="All outpatient"
    )
    provider_type_filtered_inpatient_notes = filter_provider_types(
        all_inpatient_notes, source="in patient"
    )
    provider_type_filtered_outpatient_notes = filter_provider_types(
        all_outpatient_notes, source="out patient"
    )

    provider_department_filtered_inpatient_notes = filter_provider_departments(
        provider_type_filtered_inpatient_notes,
        source="in patient",
        relevant_provider_departments=inpatient_provider_departments,
    )
    provider_department_filtered_outpatient_notes = filter_provider_departments(
        provider_type_filtered_outpatient_notes,
        source="out patient",
        relevant_provider_departments=outpatient_provider_departments,
    )

    if filter_by_word_count:
        provider_department_filtered_inpatient_notes = word_count_filter(
            provider_department_filtered_inpatient_notes, source="in patient"
        )

        provider_department_filtered_outpatient_notes = word_count_filter(
            provider_department_filtered_outpatient_notes, source="out patient"
        )

    mrn_date_filtered_inpatient_notes = filter_valid_mrn_and_date_notes(
        note_type="inpatient",
        note_dicts=provider_department_filtered_inpatient_notes,
        mrn_to_earliest_dates=mrn_to_earliest_dates,
        stratify_end=stratify_end,
    )
    mrn_date_filtered_outpatient_notes = filter_valid_mrn_and_date_notes(
        note_type="outpatient",
        note_dicts=provider_department_filtered_outpatient_notes,
        mrn_to_earliest_dates=mrn_to_earliest_dates,
        stratify_end=stratify_end,
    )

    def get_report_id(note: note_dict) -> int:
        report_id = note.get("RPT_ID")
        if report_id is None:
            raise ValueError(f"Note with MRN {note['DFCI_MRN']} missing report ID")
        return int(report_id)

    inpatient_record_ids = [
        get_report_id(note) for note in mrn_date_filtered_inpatient_notes
    ]
    unique_inpatient_record_ids = set(inpatient_record_ids)
    if len(inpatient_record_ids) != len(unique_inpatient_record_ids):
        raise ValueError(
            f"Of {len(inpatient_record_ids)} inpatient report IDs {len(unique_inpatient_record_ids)} are unique"
        )
    outpatient_record_ids = [
        get_report_id(note) for note in mrn_date_filtered_outpatient_notes
    ]
    unique_outpatient_record_ids = set(outpatient_record_ids)
    if len(outpatient_record_ids) != len(unique_outpatient_record_ids):
        raise ValueError(
            f"Of {len(outpatient_record_ids)} outpatient report IDs {len(unique_outpatient_record_ids)} are unique"
        )
    if len(unique_outpatient_record_ids & unique_inpatient_record_ids) > 0:
        raise ValueError(
            f"Outpatient and inpatient have {len(unique_outpatient_record_ids & unique_inpatient_record_ids)} report IDs in common"
        )
    target_report_ids = random.sample(
        list(unique_outpatient_record_ids | unique_inpatient_record_ids),
        k=sample_size,
    )
    assert all_unique(target_report_ids) and len(target_report_ids) == sample_size, (
        f"{len(target_report_ids)} {len(unique_outpatient_record_ids | unique_inpatient_record_ids)}"
    )

    target_report_ids = set(target_report_ids)
    print(f"Total inpatient: {len(target_report_ids & unique_inpatient_record_ids)}")
    print(f"Total outpatient: {len(target_report_ids & unique_outpatient_record_ids)}")
    print(
        f"Total: {len(target_report_ids & unique_outpatient_record_ids) + len(target_report_ids & unique_inpatient_record_ids)}"
    )

    def to_jsonl(note_json: note_dict) -> str:
        return json.dumps(note_json) + "\n"

    def valid_note(note: note_dict) -> bool:
        return get_report_id(note) in target_report_ids

    corrected_notes = [
        correct_note_text(note)
        for note in chain(
            mrn_date_filtered_inpatient_notes,
            mrn_date_filtered_outpatient_notes,
        )
        if valid_note(note)
    ]
    with open(os.path.join(output_dir, "all.json"), mode="w", encoding="utf-8") as f:
        f.writelines(
            map(
                to_jsonl,
                corrected_notes,
            )
        )
    ctakes_ready = os.path.join(
        output_dir,
        "ctakes_ready",
    )
    for fn in os.listdir(ctakes_ready):
        os.remove(os.path.join(ctakes_ready, fn))
    for corrected_note in corrected_notes:
        with open(
            os.path.join(
                ctakes_ready,
                f"{get_report_id(corrected_note)}.txt",
            ),
            mode="w",
            encoding="utf-8",
        ) as f:
            corrected_text = corrected_note.get("RPT_TEXT")
            if corrected_text is None or not isinstance(corrected_text, str):
                raise ValueError(
                    f"Missing corrected text for {get_report_id(corrected_note)}"
                )
            f.write(corrected_text)


def main():
    args = parser.parse_args()
    collect_notes_and_write_metrics(
        args.casenum_ade_date_table,
        args.casenum_dfci_mrn_table,
        args.inpatient_json_path,
        args.outpatient_json_path,
        args.inpatient_provider_departments_path,
        args.outpatient_provider_departments_path,
        args.filter_to_single_date,
        args.stratify_beginning,
        args.stratify_end,
        args.output_dir,
        args.filter_by_word_count,
        args.sample_size,
    )


# RPT_TEXT
if __name__ == "__main__":
    main()
