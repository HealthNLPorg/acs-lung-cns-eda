import re
import os
from more_itertools import partition
import logging
from operator import itemgetter
from itertools import chain
import random
from collections import Counter
from typing import cast
import argparse
import json
from main import save_jsonl

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--notes_jsonl_path",
    type=str,
    help="JSONL containing notes",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="YE OLDE",
)
parser.add_argument(
    "--relevant_departments_path",
    type=str,
    help='If a line ends with an "x" we want to retain it',
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
note_dict = dict[str, int | str]


def __load_json(line: str) -> note_dict:
    try:
        return cast(note_dict, json.loads(line))
    except Exception:
        ValueError(f"Can parse the following line as valid JSON:\n{line}")
        return {}


def __load_note_dicts(notes_jsonl_path):
    with open(notes_jsonl_path, mode="r") as f:
        return [__load_json(line) for line in f]


def __get_provider_dept_total(line: str) -> int:
    return int(line.lstrip().split()[0])


def __get_provider_dept_name(line: str) -> str | None:
    # target_column_regex = r"\"([A-Z\s0-9\/]+)\""
    target_column_regex = r"\"([^\"\']+)\""
    matches = re.findall(target_column_regex, line)
    match len(matches):
        case 1:
            return matches[0]
        case 0:
            ValueError(f"No matches found for {target_column_regex} in:\n{line}")
            return None
        case _:
            ValueError(
                f"More than one match {matches} found for {target_column_regex} in:\n{line}"
            )
            return matches[0]


def __get_type_to_total(relevant_departments_path: str) -> dict[str, int]:
    with open(relevant_departments_path, mode="r") as f:
        return {
            __get_provider_dept_name(line): __get_provider_dept_total(line)
            for line in f
            if line.rstrip().endswith("x")
            and __get_provider_dept_name(line) is not None
        }


def __select_from_note_pool(
    note_json_list: list[note_dict],
    type_to_total: dict[str, int],
    type_key: str = "PROVIDER_DEPARTMENT",
    type_total_threshold: int = 10,
    target_total: int = 250,
) -> tuple[list[note_dict], dict[str, int]]:
    logger.info(f"Total notes: {len(note_json_list)}")
    logger.info(f"Total departments: {len(type_to_total)}")

    def __under_threshold(item: tuple[str, int]) -> bool:
        return item[-1] < type_total_threshold

    _to_subsample, _to_completely_retain = partition(
        __under_threshold, type_to_total.items()
    )
    subsample_dict = dict(_to_subsample)
    completely_retain_dict = dict(_to_completely_retain)
    logger.info("To retain:")
    for dept, total in sorted(
        completely_retain_dict.items(), key=itemgetter(1), reverse=True
    ):
        print(f"{dept}\t{total}")
    logger.info("To subsample:")
    for dept, total in sorted(subsample_dict.items(), key=itemgetter(1), reverse=True):
        print(f"{dept}\t{total}")
    fully_retained = [
        note_json
        for note_json in note_json_list
        if note_json.get(type_key) is not None
        and note_json.get(type_key) in completely_retain_dict
    ]
    logger.info(f"Total retained notes: {len(fully_retained)}")
    to_subsample = [
        note_json
        for note_json in note_json_list
        if note_json.get(type_key) is not None
        and note_json.get(type_key) in subsample_dict
    ]
    logger.info(f"Total notes to subsample: {len(to_subsample)}")
    difference = target_total - sum(
        total
        for _type, total in type_to_total.items()
        if _type in completely_retain_dict
    )
    logger.info(f"Target difference: {difference}")
    raw_list = list(chain(fully_retained, random.sample(to_subsample, difference)))
    ad_hoc_totals = Counter(map(itemgetter(type_key), raw_list))

    def __get_ad_hoc_total(note_json: note_dict) -> int:
        return ad_hoc_totals[note_json[type_key]]

    return sorted(raw_list, key=__get_ad_hoc_total), ad_hoc_totals


def resample_notes(
    notes_jsonl_path: str,
    relevant_departments_path: str,
    output_dir: str,
    type_key: str = "PROVIDER_DEPARTMENT",
    type_total_threshold: int = 10,
    target_total: int = 250,
) -> None:
    note_json_list = __load_note_dicts(notes_jsonl_path)
    type_to_total = __get_type_to_total(relevant_departments_path)
    sorted_notes, new_types_to_totals = __select_from_note_pool(
        note_json_list, type_to_total
    )
    save_jsonl(
        output_dir,
        f"{type_key.lower()}_below_{type_total_threshold}_first",
        sorted_notes,
    )

    def __to_row(dept_total: tuple[str, int]) -> str:
        dept, total = dept_total
        return f"{dept}\t{total}\n"

    with open(
        os.path.join(output_dir, f"new_{type_key.lower()}_counts.tsv"), mode="w"
    ) as f:
        f.writelines(
            map(
                __to_row,
                sorted(new_types_to_totals.items(), key=itemgetter(1), reverse=True),
            )
        )


def main() -> None:
    args = parser.parse_args()
    resample_notes(
        args.notes_jsonl_path, args.relevant_departments_path, args.output_dir
    )


if __name__ == "__main__":
    main()
