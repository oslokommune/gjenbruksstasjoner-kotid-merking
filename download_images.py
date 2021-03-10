"""
The purpose of the file is to download image files from S3 so to a
local folder so they can be used for labelling and training of
ML-models.

Some possible improvements:
- Avoid hardcoded parameters.
- Generalize to more stations.
- Move training to AWS (meaning downloading locally will be obsolete).
"""

import sys
import os
import re
from datetime import datetime

from pathlib import Path
import boto3

# HARDCODED PARAMETERS
BUCKET = "ok-origo-dataplatform-prod"
PREFIX = "raw/red/REN/station_id_41"
TARGET_FOLDER = r"./actual images"


def date_from_key(key: str) -> datetime:
    """
    Get a S3-key as as a string, return a datetime object generated
    from the key. Raise an ValueError if any other number than one
    pattern is found.
    """

    matching_datetimes = re.findall(r"\d{8}T\d{6}", key)
    nfindings = len(matching_datetimes)
    if not nfindings == 1:
        raise ValueError(
            f"1 expected, but found {nfindings} datetime strings within this key:\n{key}"
        )

    return datetime.strptime(matching_datetimes[0], "%Y%m%dT%H%M%S")


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

    key_dt = date_from_key(key)

    if date_from is not None:
        if key_dt < date_from:
            return False

    if date_to is not None:
        if key_dt > date_to:
            return False

    return True


def key_downloaded(key, target_folder):
    """
    Given a S3-key, check if this file has already been downloaded to
    the local target_folder.
    """

    filename = key.split(r"/")[-1]
    my_file = Path(os.path.join(target_folder, filename))

    if my_file.is_file():
        return True
    else:
        return False


def download_file(key, target_folder):
    """
    Given a S3-key and a local target_folder, download the file there.
    """

    filename = key.split(r"/")[-1]
    dst = os.path.join(target_folder, filename)

    s3 = boto3.client("s3")
    try:
        s3.download_file(BUCKET, key, dst)
    except FileNotFoundError:
        print(f"ERROR: The target folder '{target_folder}' does not exist.")
        sys.exit(1)


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
    new_keys = [k for k in keys if not key_downloaded(k, TARGET_FOLDER)]

    # Filter on dates?
    # Specifiy date_from and date_to. If not given, both defaults to None.
    date_from = datetime(2021, 3, 5, 8, 0, 0)
    valid_keys = [key for key in new_keys if valid_by_datetime(key, date_from)]
    print(
        f"A total of {len(valid_keys)} new images in the specified period have been found and will be downloaded."
    )

    # Download
    print("Press Ctrl+Z to abort.")
    counter = 0
    for vk in valid_keys:
        download_file(vk, TARGET_FOLDER)
        counter += 1
        print(f"File {counter}/{len(valid_keys)} downloaded: {vk}")

    print("Done. Now get some coffee and have a blast labelling images!")
