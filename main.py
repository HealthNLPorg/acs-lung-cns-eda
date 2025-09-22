import polars as pl
import os
import json  # if proves to be too slow or memory consuming look into https://github.com/ICRAR/ijson + https://github.com/lloyd/yajl
from collections import Counter
import argparse
from functools import reduce, partial, lru_cache
from collections.abc import Iterable
from operator import itemgetter
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
    default=["SUBJECT", "PROVIDER_TYPE", "SPECIALTY_NAME"],
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


def write_unique_keys(
    output_dir: str, category: str, note_json_list: list[dict[str, str | int]]
) -> None:
    with open(os.path.join(output_dir, f"{category}_unique_keys.txt"), mode="w") as f:
        for key in sorted(
            set(chain.from_iterable(note_json.keys() for note_json in note_json_list))
        ):
            f.write(f"{key}\n")


def write_field_totals(
    output_dir: str, fields: list[str], note_json_list: list[dict[str, str | int]]
) -> None:
    def normalize(field: str) -> str:
        return " ".join(field.split("_")).title()

    field_to_value_totals = {field: Counter() for field in fields}
    for note_json in note_json_list:
        for field, counter in field_to_value_totals.items():
            field_value = note_json.get(field)
            counter[
                normalize(field_value)
                if isinstance(field_value, str)
                else str(field_value)
            ] += 1
    for field, counter in field_to_value_totals.items():
        pl.DataFrame(
            sorted(counter.items(), key=itemgetter(1), reverse=True),
            schema=[normalize(field), "Total"],
            orient="row",
        ).write_csv(
            os.path.join(output_dir, f"{field.lower()}_totals.tsv"), separator="\t"
        )


def write_totals(output_dir: str, fields: list[str], notes_dir: str) -> None:
    core_dirname = "_".join(os.path.basename(notes_dir).split()).lower()
    target_dir = os.path.join(output_dir, f"{core_dirname}_metrics")
    mkdir(target_dir)

    def get_notes_ls(fn: str) -> list[dict[str, str | int]]:
        with open(os.path.join(notes_dir, fn)) as f:
            return json.load(f)["response"]["docs"]

    all_notes_json_ls = list(
        chain.from_iterable(
            get_notes_ls(fn) for fn in os.listdir(notes_dir) if fn.endswith("json")
        )
    )
    write_unique_keys(
        target_dir,
        core_dirname,
        all_notes_json_ls,
    )
    write_field_totals(target_dir, fields, all_notes_json_ls)


def note_is_qualified(
    mrn_to_earliest_date: dict[int, str], note_json: dict[str, str | int]
) -> bool:
    mrn = int(note_json["DFCI_MRN"])
    pt_earliest = mrn_to_earliest_date.get(mrn)
    note_date = note_json.get("EVENT_DATE")
    return is_before(pt_earliest, note_date)


# [
#     "PROVIDER_TYPE",
#     "EVENT_DATE",
#     "INP_RPT_TYPE_CD",
#     "INP_RPT_TYPE",
#     "PROVIDER_DEPARTMENT",
#     "RPT_TEXT_NO_HTML",
#     "RPT_TYPE_STR",
#     "IMG_EXAM_END_DATE",
#     "PROVIDER_TYPE_STR",
#     "PROVIDER_DEPARTMENT_STR",
#     "LAB_STATUS_CD",
#     "ENCOUNTER_TYPE_DESC_STR",
#     "ORD_STATUS_DESC_STR",
#     "id",
#     "PROVIDER_CRED",
#     "DEPT_ID",
#     "LAST_INDEX_DATE",
#     "IMPRESSION_TEXT",
#     "PRACTICE_NAME",
#     "ENCOUNTER_TYPE_CD",
#     "PRACTICE_ID",
#     "RPT_STATUS_STR",
#     "INSTITUTION_STR",
#     "AUTHOR_NAME",
#     "AUTHOR_NAME_STR",
#     "ORD_STATUS_DESC",
#     "SUBJECT_STR",
#     "NARRATIVE_TEXT",
#     "RPT_DATE",
#     "RPT_TYPE_CD",
#     "SOURCE",
#     "ENCOUNTER_TYPE_DESC",
#     "DFCI_MRN",
#     "LAB_STATUS_DESC_STR",
#     "LAST_UPD_PRACTICE_NAME",
#     "RPT_TEXT",
#     "EDW_LAST_MOD_DT",
#     "AUTHOR_ID",
#     "EMR_SOURCE",
#     "SPECIALTY_NAME",
#     "DATA_SOURCE",
#     "PROC_DESC_STR",
#     "RPT_TYPE",
#     "PRACTICE_NAME_STR",
#     "SOURCE_STR",
#     "PROVIDER_CRED_STR",
#     "LAST_UPD_PRACTICE_ID",
#     "RPT_ID",
#     "SPECIALTY_NAME_STR",
#     "SPEC_TAKEN_DATE",
#     "ORD_STATUS_CD",
#     "ENCOUNTER_ID",
#     "RPT_STATUS_CD",
#     "PROVIDER_NAME",
#     "ACCESSION_ID",
#     "LAST_UPD_PRACTICE_NAME_STR",
#     "PROC_DESC",
#     "INSTITUTION",
#     "RPT_STATUS",
#     "INP_RPT_TYPE_STR",
#     "LAB_STATUS_DESC",
#     "_version_",
#     "SPECIALTY_ID",
#     "ADDENDUM_TEXT",
#     "SUBJECT",
#     "PROVIDER_NAME_STR",
#     "PROC_ID",
# ]
# CSV rows for reference - just in imaging?


def get_qualified_notes_from_csv(
    mrn_to_earliest_date: dict[int, str], csv_path: str
) -> list[dict[str, str | int]]:
    local_qualified = partial(note_is_qualified, mrn_to_earliest_date)
    note_json_list = pl.read_csv(csv_path).to_dicts()
    return [note_json for note_json in note_json_list if local_qualified(note_json)]


def get_qualified_notes_from_json(
    mrn_to_earliest_date: dict[int, str], json_path: str
) -> list[dict[str, str | int]]:
    with open(json_path) as f:
        note_json_list = json.load(f)["response"]["docs"]
    local_qualified = partial(note_is_qualified, mrn_to_earliest_date)
    return [note_json for note_json in note_json_list if local_qualified(note_json)]


def get_dir_totals(
    mrn_to_earliest_date: dict[int, str], notes_dir: str
) -> Iterable[Counter[int]]:
    for root, dirs, files in os.walk(notes_dir):
        root_total = 0
        for fn in files:
            if fn.lower().endswith("json"):
                file_counter = get_file_totals_json(
                    mrn_to_earliest_date, os.path.join(root, fn)
                )
                yield file_counter
                root_total += sum(file_counter.values())
            if fn.lower().endswith("csv"):
                file_counter = get_file_totals_csv(
                    mrn_to_earliest_date, os.path.join(root, fn)
                )
                yield file_counter
                root_total += sum(file_counter.values())
        if root_total > 0:
            logger.info(
                f"Total qualifying files in {os.path.basename(root)}: {root_total}"
            )


def get_totals(
    mrn_to_earliest_date: dict[int, str], notes_dir: str
) -> Iterable[tuple[int, int]]:
    full_counter = reduce(
        Counter.__add__, get_dir_totals(mrn_to_earliest_date, notes_dir)
    )
    # return sorted by totals
    return sorted(full_counter.items(), key=itemgetter(1), reverse=True)


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

    final_sorted_df = pl.DataFrame(
        get_totals(mrn_to_earliest_date, notes_dir),
        schema=["MRN", "TOTAL_AFTER_EARLIEST"],
        orient="row",
    )
    final_sorted_df.write_csv(os.path.join(output_dir, "totals.csv"), separator=",")


def main():
    args = parser.parse_args()
    collect_notes_and_write_metrics(
        args.pt_record_csv, args.notes_dir, args.output_dir, args.fields
    )


if __name__ == "__main__":
    main()
