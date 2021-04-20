"""
The purpose of the file is to download image files from S3 so to a
local directory so they can be used for labelling and training of
ML-models.

Some possible improvements:
- Avoid hardcoded parameters.
- Generalize to more stations.
- Move training to AWS (meaning downloading locally will be obsolete).
"""

import os
import re
from datetime import datetime
from pathlib import Path

import boto3

from common import datetime_from_string

# HARDCODED PARAMETERS
BUCKET = "ok-origo-dataplatform-prod"
PREFIX = "raw/red/REN/station_id_41"
TARGET_DIR = "./actual_images"


def valid_by_datetime(key, date_from=None, date_to=None):
    """
    Return True is the key contains a date which is within the
    specified range, if not, return False.
    """

    if not (isinstance(date_from, datetime) or date_from is None):
        raise TypeError(
            f"date_from is type {type(date_from)}, not the expected datetime or NoneType"
        )
    if not (isinstance(date_to, datetime) or date_to is None):
        raise TypeError(
            f"date_to is a {type(date_to)}, not the expected datetime or NoneType"
        )

    key_dt = datetime_from_string(key)

    if (date_from and key_dt < date_from) or (date_to and key_dt > date_to):
        return False

    return True


def key_downloaded(key, target_dir):
    """
    Given an S3-key, check if this file has already been downloaded to
    the local target_dir.
    """

    filename = key.split(r"/")[-1]
    my_file = Path(os.path.join(target_dir, filename))

    return my_file.is_file()


def download_file(key, target_dir):
    """
    Given an S3-key and a local target_dir, download the file there.
    """

    filename = key.split(r"/")[-1]
    dst = os.path.join(target_dir, filename)

    s3 = boto3.client("s3")
    s3.download_file(BUCKET, key, dst)


if __name__ == "__main__":

    print(f"Reading images at bucket {BUCKET}, prefix {PREFIX}.")
    client = boto3.client("s3")
    keys = [
        e["Key"]
        for p in client.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=PREFIX
        )
        for e in p["Contents"]
    ]

    # Find all the images which have not yet be downloaded to the target folder
    new_keys = [k for k in keys if not key_downloaded(k, TARGET_DIR)]

    # Filter on dates?
    # Specify date_from and date_to. If not given, both defaults to None.
    date_from = datetime(2021, 3, 5, 8, 0, 0)
    valid_keys = [key for key in new_keys if valid_by_datetime(key, date_from)]
    print(
        f"A total of {len(valid_keys)} new images in the specified period have been found and will be downloaded."
    )

    # Download
    print("Press Ctrl+C to abort.")
    if not os.path.exists(TARGET_DIR):
        print(f"Made folder {TARGET_DIR}")
        os.makedirs(TARGET_DIR)
    for i, vk in enumerate(valid_keys):
        download_file(vk, TARGET_DIR)
        print(f"File {i + 1}/{len(valid_keys)} downloaded: {vk}")

    print("Done. Now get some coffee and have a blast labelling images!")
