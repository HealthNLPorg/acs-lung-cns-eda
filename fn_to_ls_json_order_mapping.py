import polars as pl
import os
import argparse
import json
from collections import deque
from operator import itemgetter
from more_itertools import unzip

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--json_corpus",
    type=str,
    help="JSON containing the annotated/pre-annotated Label Studio corpus",
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Output dir",
)


def parse_report_id(file_annotation: dict, key: str = "file_upload") -> int:
    file_upload = file_annotation.get(key)
    if file_upload is None:
        ValueError(f"Cannot find file_upload field in {file_annotation}")
        return -1
    return int(file_upload.split("."))


def store_order(json_corpus: str, output_dir: str) -> None:
    report_id_placement_pairs = deque()
    with open(json_corpus, mode="r") as f:
        corpus = json.load(f)
    for idx, file_annotation in enumerate(corpus):
        report_id_placement_pairs.append((parse_report_id(file_annotation), idx))
    report_id_iter, placement_iter = unzip(
        sorted(report_id_placement_pairs, key=itemgetter(0))
    )
    pl.DataFrame({"report_id": report_id_iter, "placement": placement_iter}).write_csv(
        os.path.join(output_dir, "report_id_to_placement.tsv"), separator="\t"
    )


def main():
    args = parser.parse_args()
    store_order(args.json_corpus, args.output_dir)


if __name__ == "__main__":
    main()
