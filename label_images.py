import os
import datetime

import cv2
import pandas as pd
import numpy as np

from common import datetime_from_string

np.random.seed(1)

# HARDCODED PARAMETERS
PICTURES_DIR = "./actual_images"
LABELS_FILENAME = "labels_data.csv"


def update_label_data(pictures_dir: str, labels_data: str) -> pd.DataFrame:
    """
    Read the content of an (assumed) existing .csv-file of image
    filenames and possible previous labeling data. Merge this with a
    table of empty rows for the new images which have been downloaded.
    """

    COLUMN_NAMES = [
        "timestamp",
        "open",
        "set_type",
        "relevant",
        "queue_full",
        "queue_empty",
        "queue_end_pos",
        "lanes",
        "labelled",
    ]

    try:
        df1 = pd.read_csv(labels_data, index_col=0)
        df1["timestamp"] = [datetime_from_string(nfn) for nfn in list(df1.index)]
    except FileNotFoundError:
        df1 = pd.DataFrame(columns=COLUMN_NAMES)

    # Which files have been downloaded, but not added to the list?
    file_names = os.listdir(pictures_dir)
    image_files = [fn for fn in file_names if fn[-4:] == ".jpg"]
    new_file_names = list(set(image_files).difference(set(df1.index)))

    # Add rows for new files to df2
    df2 = pd.DataFrame(columns=COLUMN_NAMES, index=new_file_names)
    df2["timestamp"] = [datetime_from_string(nfn) for nfn in new_file_names]
    df2["open"] = df2["timestamp"].map(within_opening_hours)
    df2["labelled"] = False

    # Merge and assign set type
    df = pd.concat((df1, df2), axis=0)
    df = df.sort_values(by="timestamp", ascending=True)
    df["set_type"] = assign_image_to_set(df.shape[0])

    return df


def print_help():
    """
    Print a help string.
    """

    HELP_STRING = """
    H: Help (this table)
    Mouse Left click: Set end of queue.
    E: Empty (no cars).
    F: Full (can't see end of line).
    1: One lane.
    2: Two lanes.
    R: Relevant or not (toggle).
    C: Clear all.
    Space: Save labelling and move to next image.
    """
    print(HELP_STRING)


def within_opening_hours(dt: datetime.datetime):
    """
    Return True if dt is within the opening hours for Haraldrud. Note
    that these functions are based on a rule of thumb which is
    normally correct. This was considered GOOD ENOUGH enough, since the
    purpose is to not spend time of images outside the opening hours.
    """

    # Post-Covid-19 restart (approximately April 2020)
    if dt.year == 2020 and dt.month == 4:
        return (dt.hour >= 10) and (dt.hour < 17) and (dt.weekday() in [0, 1, 2, 3, 4])

    # Later (does not handle holidays, but this is not crucial)
    if dt.weekday() in [0, 1, 2, 3]:
        # Mon - Thu
        return (dt.hour >= 8) and (dt.hour <= 20)
    elif dt.weekday() in [4, 5]:
        # Fri - Sat
        return (dt.hour >= 9) and (dt.hour <= 15)
    elif dt.weekday() in [6]:
        # Sun
        return False


def assign_image_to_set(
    number_of_images: int, train=0.7, valid=0.15, test=0.15
) -> np.array:
    """
    Based on the number_of_images, return an equal length array which
    split in the sets Train, Valid or Train.

    The intended purpose is to assign images to these sets. Since the
    rows (outside this function) are sorted by date and the random
    seed is set, the sets will be consistent when (only new) pictures
    are added.
    """

    if (train + valid + test) != 1.0:
        raise ValueError("The proportions of image sets does not add to 100%.")

    np.random.seed(1)
    rnd = np.random.random(number_of_images)
    assigned_set = np.where(
        rnd < train, "Train", np.where(rnd >= (1 - test), "Test", "Valid")
    )

    return assigned_set


class MouseCoordinates:
    """
    Used to register mouse clicks, i.e. where the end of the queue is
    located.
    """

    def __init__(self):

        self.x = np.nan
        self.y = np.nan
        self.clicked = False

    def set_end_of_queue(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            self.clicked = True


def get_text_y():
    """
    Simple generator used to give y coordinate when displaying status.
    """

    y = 40
    while True:
        yield y
        y += 20


def draw_queue_end(
    im: np.ndarray, fn: str, df: pd.DataFrame, text_spec: dict
) -> np.ndarray:
    """
    Given the filename and the information of the image stored in df,
    draw relevant information on top of the image im and return it.
    """

    if (
        df.loc[fn, "queue_full"] is True
    ):  # keep as-is. Dropping "is True" gives an undesired result.
        im = cv2.putText(
            im, "?", (1150, 150), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3
        )
        queue_text = "Koens ende er ikke synlig"
    elif df.loc[fn, "queue_empty"] is True:
        im = cv2.putText(
            im, "_", (10, 150), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3
        )
        queue_text = "Koen er tom."
    elif not np.isnan(df.loc[fn, "queue_end_pos"]):
        x = int(df.loc[fn, "queue_end_pos"])
        im = cv2.line(im, (x, 0), (x, im.shape[0]), (255, 255, 255), 2)
        queue_text = "x={0}".format(df.loc[fn, "queue_end_pos"])
    else:
        queue_text = ""

    im = cv2.putText(
        im,
        "Slutt ko: {0}".format(queue_text),
        (text_spec["x_pos"], next(text_spec["y_generator"])),
        text_spec["font"],
        1,
        text_spec["color"],
        1,
    )

    return im


def get_weekday(filename: str) -> str:
    """
    Returns the weekday given a filename.
    """

    d = datetime_from_string(filename).date()
    WEEKDAYS = (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )
    return WEEKDAYS[d.weekday()]


def get_next_file_name(df):
    """
    Get the next image to be labelled.
    """

    df_tmp = df.loc[df.loc[:, "relevant"].isnull(), :].copy()
    return df_tmp.index[-1]


def check_entry(df, filename):
    """
    Check if the labelling done to this image is sufficient
    Lanes are not mandatory
    """

    if df.loc[filename, "relevant"] is False:
        return True

    if df.loc[filename, "relevant"] is True:
        if df.loc[filename, "queue_full"] is True:
            return True
        elif df.loc[filename, "queue_empty"] is True:
            return True
        elif not np.isnan(df.loc[filename, "queue_end_pos"]):
            return True

    return False


def label_images(df, pictures_folder):
    """
    This main loop for reading new unlabeled images, labelling them
    and writing the updated information to the .csv file.
    """

    fn = get_next_file_name(df)

    mc = MouseCoordinates()
    WIN_NAME = "Label the image"
    cv2.namedWindow(WIN_NAME)
    try:
        cv2.setMouseCallback(WIN_NAME, mc.set_end_of_queue)
        pass
    except cv2.error:
        #
        print(
            "Ubuntu 18.04 - USE: sudo apt install libcanberra-gtk-module libcanberra-gtk3-module"
        )
        raise

    load_new = True
    while True:

        if load_new:
            fn = get_next_file_name(df)
            print(f"Image file: {fn}")
            full_path = os.path.join(pictures_folder, fn)
            im_org = cv2.imread(full_path, cv2.IMREAD_COLOR)
            load_new = False

        im = im_org.copy()

        # Define the size and style of the status legend
        text_spec = {
            "color": (0, 0, 0),
            "font": cv2.FONT_HERSHEY_PLAIN,
            "x_pos": 10,
            "y_generator": get_text_y(),
        }
        im = cv2.rectangle(im, (0, 20), (400, 100), (255, 255, 255), -1)

        # Handle mouse clicks
        if mc.clicked:
            df.loc[fn, "queue_end_pos"], _ = mc.x, mc.y
            df.loc[fn, "queue_full"] = False
            df.loc[fn, "queue_empty"] = False
            mc.clicked = False

        # Draw end of queue end
        im = draw_queue_end(im, fn, df, text_spec)

        # Draw number of lanes (to legend)
        if ~np.isnan(df.loc[fn, "lanes"]):
            lanes_text = int(df.loc[fn, "lanes"])
        else:
            lanes_text = ""
        im = cv2.putText(
            im,
            "Lanes: {0}".format(lanes_text),
            (text_spec["x_pos"], next(text_spec["y_generator"])),
            text_spec["font"],
            1,
            text_spec["color"],
            1,
        )

        # Draw relevance (to legend)
        if ~np.isnan(df.loc[fn, "relevant"]):
            relevant_text = df.loc[fn, "relevant"]
        else:
            relevant_text = ""
        im = cv2.putText(
            im,
            "Relevant: {0}".format(relevant_text),
            (text_spec["x_pos"], next(text_spec["y_generator"])),
            text_spec["font"],
            1,
            text_spec["color"],
            1,
        )

        cv2.imshow(WIN_NAME, im)
        k = cv2.waitKey(1)

        # Handle keys pressed
        if k % 256 == 32:  # Space - Save the labelling data
            if check_entry(df, fn):
                print("Successful entry.")
                df.loc[fn, "labelled"] = True
                df.to_csv(LABELS_FILENAME, index=True)
                load_new = True
                continue
            else:
                print("Labelling not done, please review.")

        if k % 256 == 27:  # Escape - Quit the program
            break

        if k % 256 == 99:  # c = clear all
            df.loc[fn, "queue_end_pos"] = np.nan
            df.loc[fn, "queue_full"] = np.nan
            df.loc[fn, "queue_empty"] = np.nan
            df.loc[fn, "relevant"] = np.nan
            df.loc[fn, "lanes"] = np.nan

        # Absolutes - mouse positions not used
        if k % 256 == 101:  # e = empty queue
            df.loc[fn, "queue_end_pos"] = np.nan
            df.loc[fn, "queue_full"] = False
            df.loc[fn, "queue_empty"] = True
        elif k % 256 == 102:  # f = full queue (can't see the end of the line)
            df.loc[fn, "queue_end_pos"] = np.nan
            df.loc[fn, "queue_full"] = True
            df.loc[fn, "queue_empty"] = False

        if k % 256 == 104:  # h = help
            print_help()

        if k % 256 in [49, 50, 51]:
            if k % 256 == 49:  # 1 = One lane
                df.loc[fn, "lanes"] = 1
            elif k % 256 == 50:  # 2 = Two lanes
                df.loc[fn, "lanes"] = 2
            elif k % 256 == 51:  # 3 = Not relevant (e.g. if no cars)
                df.loc[fn, "lanes"] = np.nan

        if k % 256 == 114:  # r = Relevant (toggle)
            if df.loc[fn, "relevant"] is True:
                df.loc[fn, "relevant"] = False
            elif df.loc[fn, "relevant"] is False:
                df.loc[fn, "relevant"] = True
            elif np.isnan(df.loc[fn, "relevant"]):
                df.loc[fn, "relevant"] = True

        # Can be de-commented during development.
        # Which key number was pressed?
        # if k % 256 < 255:
        #     print(k % 256)

    cv2.destroyAllWindows()

    return df


if __name__ == "__main__":

    df = update_label_data(PICTURES_DIR, LABELS_FILENAME)
    df = label_images(df, PICTURES_DIR)

    print("That was fun!")
