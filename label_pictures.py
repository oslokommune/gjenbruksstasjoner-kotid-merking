import sys
import os
import re
import datetime

import cv2
import pandas as pd
import numpy as np

np.random.seed(1)

PICTURES_FOLDER = r"./actual_images"
LABELS_FILENAME = "labels_data.csv"  # .csv to open for manual manipulation


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


def get_datetime(filename):
    """"""

    try:
        dt_str = re.findall(r"(\d{8}T\d{6})", filename)[0]
    except IndexError:
        print(filename)
        sys.exit(1)

    dt = datetime.datetime.strptime(dt_str, "%Y%m%dT%H%M%S")

    return dt


def within_opening_hours(dt):

    """These functions are roughly correct. Consider tweaking if more exactlness is needed."""

    # Post-Covid-19 restart (approximately April 2020)
    if dt.year == 2020 and dt.month == 4:
        if (dt.hour >= 10) and (dt.hour < 17) and (dt.weekday() in [0, 1, 2, 3, 4]):
            return True
        else:
            return False

    # Otherwise (this is not exact, but good enough...)
    if dt.weekday() in [0, 1, 2, 3]:
        # Mon - Thu
        if (dt.hour >= 8) and (dt.hour <= 20):
            return True
        else:
            return False

    elif dt.weekday() in [4, 5]:
        # Fri + Sat
        if (dt.hour >= 9) and (dt.hour <= 15):
            return True
        else:
            return False

    elif dt.weekday() in [6]:
        # Sun
        return False


def assign_image_to_set(number_of_images, train=0.7, valid=0.15, test=0.15):

    assert (train + valid + test) == 1.0

    np.random.seed(1)

    rnd = np.random.random(number_of_images)
    assigned_set = np.where(
        rnd < train, "Train", np.where(rnd >= (1 - test), "Test", "Valid")
    )

    return assigned_set


def update_label_data(pictures_folder, labels_data):

    file_names = os.listdir(pictures_folder)
    file_names = [fn for fn in file_names if fn[-4:] == ".jpg"]

    # Read the old df1
    try:
        df1 = pd.read_csv(labels_data, index_col=0)
        df1["timestamp"] = [
            get_datetime(nfn) for nfn in list(df1.index)
        ]  # Format lost while saving to .csv
    except FileNotFoundError:

        df1 = pd.DataFrame(
            columns=[
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
        )

    # Which files have been downloaded, but not added to the list?
    new_file_names = list(set(file_names).difference(set(df1.index)))

    # Create df2 - to be appended
    timestamp = [get_datetime(nfn) for nfn in new_file_names]
    df2 = pd.DataFrame({"timestamp": timestamp}, index=new_file_names)
    df2["open"] = df2["timestamp"].map(within_opening_hours)
    df2["set_type"] = np.nan
    df2["relevant"] = np.nan
    df2["queue_full"] = np.nan
    df2["queue_empty"] = np.nan
    df2["queue_end_pos"] = np.nan
    df2["lanes"] = np.nan
    df2["labelled"] = False

    # Append
    print(df1.head())
    print(df2.head())
    df = pd.concat((df1, df2), axis=0)
    print(df.tail())
    df = df.sort_values(by="timestamp", ascending=True)
    df["set_type"] = assign_image_to_set(df.shape[0])

    return df


class MouseCoordinates(object):
    def __init__(self):

        self.x = np.nan
        self.y = np.nan
        self.clicked = False

    def set_end_of_queue(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            self.clicked = True

        if event == cv2.EVENT_RBUTTONDOWN:
            self.x = np.nan
            self.y = np.nan
            self.clicked = True


def get_text_y():

    # Simple generator used during print status
    y = 40
    while True:
        yield y
        y += 20


def draw_queue_end(im, fn, df, text_spec):

    color = text_spec["color"]
    font = text_spec["font"]
    x = text_spec["x_pos"]
    yg = text_spec["y_generator"]

    if df.loc[fn, "queue_full"] is True:
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
    Yes, the number of lines here can probably be cut by 75%.
    """

    comp_exp = re.compile(r"_(\d{8})T")
    match_obj = comp_exp.findall(filename)
    if len(match_obj) == 0:
        raise ValueError(f"No date found in {filename}.")
    elif len(match_obj) > 1:
        print(match_obj)
        raise ValueError(f"Wut?? {filename}")

    datestring = match_obj[0]
    date = datetime.date(
        int(datestring[0:4]), int(datestring[4:6]), int(datestring[6:8])
    )
    if date.weekday() == 0:
        return "Monday"
    elif date.weekday() == 1:
        return "Tuesday"
    elif date.weekday() == 2:
        return "Wednesday"
    elif date.weekday() == 3:
        return "Thurday"
    elif date.weekday() == 4:
        return "Friday"
    elif date.weekday() == 5:
        return "Saturday"
    elif date.weekday() == 6:
        return "Sunday"
    else:
        raise Exception("Wut?")


def get_next_file_name(fn, df):

    # Every entry where relevant is True/False won't be included here.

    df_tmp = df.loc[df.loc[:, "relevant"].isnull(), :].copy()

    SPECIAL_CASE = False
    if SPECIAL_CASE:
        # Implemented at a quick workaround 2020-Nov-12 to train on images from Monday + Saturday.
        # Can be generalized among other things for å more targeted training.
        VALID_WEEKDAYS = ["Monday", "Saturday"]
        df_tmp["Weekday"] = df_tmp.index.map(get_weekday)
        df_tmp = df_tmp.loc[df_tmp["Weekday"].isin(VALID_WEEKDAYS), :].copy()

    return df_tmp.index[-1]


def check_entry(df, filename):

    # Check if the labelling done to this image is sufficient
    # Lanes are not mandatory

    # Pics with no relevant information don't need this information filled in
    if df.loc[filename, "relevant"] is False:
        return True

    if df.loc[filename, "relevant"] is True:

        if df.loc[filename, "queue_full"] is True:  # and \
            # (not np.isnan(df.loc[filename, "lanes"])):

            return True

        elif df.loc[filename, "queue_empty"] is True:

            return True

        elif not np.isnan(df.loc[filename, "queue_end_pos"]):  #  and \
            # (not np.isnan(df.loc[filename, "lanes"])):

            return True

    return False


def label_images(df, pictures_folder):

    fn = get_next_file_name(np.nan, df)

    mc = MouseCoordinates()
    WIN_NAME = "Label the image"
    cv2.namedWindow(WIN_NAME)
    try:
        cv2.setMouseCallback(WIN_NAME, mc.set_end_of_queue)
        pass
    except cv2.error:
        print(
            "Ubuntu 18.04 - USE: sudo apt install libcanberra-gtk-module libcanberra-gtk3-module"
        )
        raise

    load_new = True
    while True:

        if load_new:
            fn = get_next_file_name(fn, df)
            print(fn)
            full_path = os.path.join(pictures_folder, fn)
            im_org = cv2.imread(full_path, cv2.IMREAD_COLOR)
            load_new = False

        im = im_org.copy()

        # Size status rectange
        text_spec = {
            "color": (0, 0, 0),
            "font": cv2.FONT_HERSHEY_PLAIN,
            "x_pos": 10,
            "y_generator": get_text_y(),
        }
        im = cv2.rectangle(im, (0, 20), (400, 100), (255, 255, 255), -1)

        # Handle mouse clicks
        if mc.clicked:
            df.loc[fn, "queue_end_pos"], _ = mc.x, mc.y  # x_queue_end, y_queue_end
            df.loc[fn, "queue_full"] = False
            df.loc[fn, "queue_empty"] = False
            mc.clicked = False

        # Draw queue end
        im = draw_queue_end(im, fn, df, text_spec)

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
        if k % 256 == 32:  # Space
            if check_entry(df, fn):
                print("Successfull entry.")
                df.loc[fn, "labelled"] = True
                print(df.dtypes)
                df.to_csv(LABELS_FILENAME, index=True)
                load_new = True
            else:
                print("Labelling not done, please review.")

        if k % 256 == 27:  # Escape
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
            elif k % 256 == 51:  # 3 = Unclear or not relevant (e.g. if no cars)
                df.loc[fn, "lanes"] = np.nan

        if k % 256 == 114:  # r = Relevant (toggle)
            if df.loc[fn, "relevant"] is True:
                df.loc[fn, "relevant"] = False
            elif df.loc[fn, "relevant"] is False:
                df.loc[fn, "relevant"] = True
            elif np.isnan(df.loc[fn, "relevant"]):
                df.loc[fn, "relevant"] = True
            else:
                print("Value of relevant: {0}".format(df.loc[fn, "relevant"]))
                print("dtype of relevant: {0}".format(df.loc[fn, "relevant"].dtype))
                raise Exception("WUT?")

        if k % 256 < 255:
            print(k % 256)

    cv2.destroyAllWindows()

    return df


def test():

    dt1 = datetime.datetime(2020, 4, 15, 12, 0, 0)

    print(dt1.month)
    print(dt1.year)


if __name__ == "__main__":

    # test(); import sys; sys.exit(1)

    print(LABELS_FILENAME)
    df = update_label_data(PICTURES_FOLDER, LABELS_FILENAME)
    df = label_images(df, PICTURES_FOLDER)

    print("That was fun!")
