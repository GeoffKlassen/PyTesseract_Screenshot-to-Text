from ScreenshotClass import Screenshot
from ParticipantClass import Participant
from RuntimeValues import *
import os
import re
import numpy as np
import pandas as pd
import pytesseract
import cv2
from collections import namedtuple, Counter
import requests
from PIL import Image
from io import BytesIO
import warnings
from datetime import datetime
import time

"""
    Variables defined for use as column names
"""
SCREENTIME = 'screentime'
PICKUPS = 'pickups'
NOTIFICATIONS = 'notifications'

HEADING_COLUMN = 'heading'
SCREENTIME_HEADING = SCREENTIME
LIMITS_HEADING = 'limits'
MOST_USED_HEADING = 'most used'
PICKUPS_HEADING = PICKUPS
FIRST_PICKUP_HEADING = 'first pickup'
FIRST_USED_AFTER_PICKUP_HEADING = 'first used after pickup'
NOTIFICATIONS_HEADING = NOTIFICATIONS
HOURS_AXIS_HEADING = 'hours row'
DAY_OR_WEEK_HEADING = 'day or week'

NO_TEXT = '-99999'
NO_NUMBER = -99999
NO_CONF = -1

TODAY = 'today'
YESTERDAY = 'yesterday'
DAY_OF_THE_WEEK = 'weekday'
WEEK = 'week'

PARTICIPANT_ID = 'participant_id'
IS_RESEARCHER = 'is_researcher'
RESPONSE_DATE = 'response_date'
IMG_RESPONSE_TYPE = 'img_response_type'
IMG_URL = 'img_url'


def compile_list_of_urls(df, screentime_cols, pickups_cols, notifications_cols,
                         time_col='Record Time', id_col='Participant ID', label_col='Participant Label'):
    """Create a dataframe of URLs from a provided dataframe of survey responses.
    Args:
        df:                 The dataframe with columns that contain URLs
        screentime_cols:    An array of the column names in df that are for URLs of screentime images
        pickups_cols:       An array of the column names in df that are for URLs of pickups (unlocks) images
        notifications_cols: An array of the column names in df that are for URLs of notifications images
        time_col:           The name of the column in df that lists the date & time stamp of submission
                            (For Avicenna CSVs, this is usually 'Record Time')
        id_col:             The name of the column in df that contains the ID of the user
                            (For Avicenna CSVs, this is usually 'Participant ID')
        label_col:          The name of the column in df that indicates whether the row is for a IS_RESEARCHER
                            (For Avicenna CSVs, this is usually 'Participant Label')
    Returns:
        pd.Dataframe:       A dataframe of image URLs, along with the user ID, response date, and category the image was
                            submitted in.

    NOTE: iOS & Android Dashboards have three categories of usage statistics: screentime, pickups (aka unlocks), and
    notifications. Because of this, the dataframe provided may have multiple columns of URLs (e.g. one column for each
    usage category, multiple columns for only one category, or multiple columns for all three categories).
    """
    url_df = pd.DataFrame(columns=[PARTICIPANT_ID, IS_RESEARCHER, RESPONSE_DATE, IMG_RESPONSE_TYPE, IMG_URL])
    for i in df.index:
        user_id = df[id_col][i]

        # Extract the response date from the date column, in date format
        try:
            response_date = datetime.strptime(str(df[time_col][i])[0:10], "%Y-%m-%d")  # YYYY-MM-DD = 10 chars
        except ValueError:  #
            response_date = datetime.strptime("1999-01-01", "%Y-%m-%d")  # TODO: try None?

        for col_name in screentime_cols + pickups_cols + notifications_cols:
            # Cycle through each column in which a user might have submitted a screenshot and add its URL to the list
            url = str(df[col_name][i])
            if not url.startswith("https://file.avicennaresearch.com"):
                continue

            if col_name in screentime_cols:
                img_response_type = SCREENTIME
            elif col_name in pickups_cols:
                img_response_type = PICKUPS
            elif col_name in notifications_cols:
                img_response_type = NOTIFICATIONS
            else:
                img_response_type = None
            # Note: For the Boston Children's Hospital data, all images are of type SCREENTIME

            new_row = {PARTICIPANT_ID: user_id,
                       IS_RESEARCHER: True if df.loc[i, id_col] == IS_RESEARCHER else False,
                       RESPONSE_DATE: response_date.date(),
                       IMG_RESPONSE_TYPE: img_response_type,
                       IMG_URL: url}
            url_df = pd.concat([url_df, pd.DataFrame([new_row])], ignore_index=True)

    return url_df


def load_and_process_image(screenshot, white_threshold=200, black_threshold=60):
    """
    Very helpful tutorial: https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
     Load the image from the file and some image processing to make the image "readable" for tesseract.
    """

    def _download_image_from_avicenna(img_path):

        empty_arr = np.array([])
        if "https" not in img_path and "avicennaresearch" not in img_path:
            print("Skipping. Invalid URL.")
            return empty_arr
        test_url = img_path.replace("https", "http")
        print("Downloading image...")
        # The first request without auth headers to check what the redirected URL is.
        response = requests.get(test_url, verify=True)
        actual_url = response.url
        if "https" in actual_url:
            response = requests.get(img_path, verify=True, auth=(user, passw))
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if use_downloaded_images:
                    img = img.convert('RGB')  # We'll standardize the image to RGB format.
                    img.save(img_local_path)
                    print(f"Image saved to local folder '{dir_for_downloaded_images}'.")
                # _adjust_size(img)   # TODO verify if img properly adjusted.
                img_array = np.array(img, dtype='uint8')
                return img_array
            else:
                print("Invalid status code: %s" % response.status_code)
                return empty_arr
        else:
            print("Skipping. We only want to be connecting to Avicenna using the https protocol.")
            return empty_arr

    def remove_color_blocks_from_image(img, lightmode):
        """
            - Code used here is from: https://stackoverflow.com/a/72068384
            - Finding upper and lower limits HSV values for a given RGB color: https://www.youtube.com/watch?v=x4qPhYamRDI
        """
        # in our IOS image, there are often a few colors in the bar graph. Here, we are defining
        # Lower and upper limits of all colors. Format: {'color': [color_lower_limit, color_upper_limit], ...}
        # color_dict = {'orange': (np.array([12, 100, 100], np.uint8), np.array([24, 255, 255], np.uint8)),
        #               'blue': (np.array([80, 100, 100], np.uint8), np.array([115, 255, 255], np.uint8)),
        #               'red': (np.array([0, 100, 100], np.uint8), np.array([14, 255, 255], np.uint8))}
        color_dict = {'all': (np.array([0, 100, 100], np.uint8), np.array([180, 255, 255], np.uint8))}

        for color in color_dict.keys():
            color_min = color_dict[color][0]
            color_max = color_dict[color][1]
            #  to HSV colourspace and get mask of blue pixels
            HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(HSV, color_min, color_max)
            # Try dilating (enlarging) the mask.
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)

            if lightmode:
                img[mask > 0] = [255, 255, 255]  # Make all pixels in mask white
            else:
                img[mask > 0] = [0, 0, 0]  # Make all pixels in mask black

    if use_downloaded_images:
        img_local_path = os.path.join(dir_for_downloaded_images, screenshot.filename)
        if os.path.exists(img_local_path):
            print("Opening local image...")
            image = Image.open(img_local_path)
            image = image.convert('RGB')
            image = np.array(image, dtype='uint8')
        else:
            image = _download_image_from_avicenna(screenshot.url)
    else:
        image = _download_image_from_avicenna(screenshot.url)

    if image.size == 0:
        # We'll just return the empty array...
        return image, image
    # check if image is in lightmode or dark mode
    threshold = 127  # Halfway between 0 and 255
    is_light_mode = np.mean(image) > threshold

    remove_color_blocks_from_image(image, is_light_mode)

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To gray scale.
    if is_light_mode:
        # threshold arguments: image, threshold value, default to use if value > threshold. type of threshold to apply.
        #       for last argument: can be cv2.THRESH_BINARY or cv2.THRESH_OTSU
        # IMPORTANT: Looks like the best threshold value for a light mode image is 206. # GK: 230 makes text thicker,
        # but also makes the 'thermometer' shaped bars (below each app name) more likely to show up
        (_, bw_image) = cv2.threshold(grey_image, white_threshold, 255, cv2.THRESH_BINARY)
        return grey_image, bw_image
    else:
        # Settings for dark mode.
        (_, bw_image) = cv2.threshold(grey_image, black_threshold, 180, cv2.THRESH_BINARY)

    return grey_image, bw_image


def merge_df_rows_by_height(df):
    # Sometimes two 'words' that are side-by-side on the screenshot end up on their own lines in the df.
    # Merge these rows into a single row (combine their 'text' values, average their 'conf' values, and
    # increase the 'width' value of the first word).

    df['right'] = df['left'] + df['width']
    df = df.sort_values(by=['top', 'left']).reset_index(drop=True)
    rows_to_drop = []
    for i in df.index:
        if i == 0:
            continue
        if abs(df.loc[i]['top'] - df.loc[i - 1]['top']) < 15:  # If two rows' heights are within 15 pixels of each other
            # TODO replace 15 with a percentage of the screenshot width (define moe = x% of screenshot width)
            if df.loc[i]['left'] > df.loc[i - 1]['left']:
                df.at[i - 1, 'text'] = df.loc[i - 1]['text'] + ' ' + df.loc[i]['text']
                df.at[i - 1, 'width'] = max(df.loc[i]['right'], df.loc[i - 1]['right']) - min(df.loc[i]['left'],
                                                                                              df.loc[i - 1]['right'])

                df.at[i - 1, 'conf'] = (df.loc[i - 1]['conf'] + df.loc[i]['conf']) / 2
                rows_to_drop.append(i)
            elif df.loc[i - 1]['left'] > df.loc[i]['left']:
                df.at[i, 'text'] = df.loc[i]['text'] + ' ' + df.loc[i - 1]['text']
                df.at[i, 'width'] = max(df.loc[i]['right'], df.loc[i - 1]['right']) - min(df.loc[i]['left'],
                                                                                          df.loc[i - 1]['right'])

                df.at[i, 'conf'] = (df.loc[i]['conf'] + df.loc[i - 1]['conf']) / 2
                rows_to_drop.append(i - 1)

    if 'level_0' in df.columns:
        df.drop(columns=['level_0'], inplace=True)
    consolidated_df = df.drop(index=rows_to_drop).reset_index()

    return consolidated_df


def merge_df_rows_by_line_num(df):
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']

    df_rows_merged = df.groupby(['page_num',
                                 'block_num',
                                 'par_num',
                                 'line_num']).agg({'left': 'min',
                                                   'top': 'min',
                                                   'width': 'last',
                                                   'height': 'max',
                                                   'conf': 'mean',
                                                   'text': lambda x: ' '.join(x),
                                                   'right': 'max',
                                                   'bottom': 'max'})
    df_rows_merged.columns = ['left', 'top', 'width', 'height', 'conf', 'text', 'right', 'bottom']
    df_rows_merged = df_rows_merged.reset_index()
    df_rows_merged['width'] = df_rows_merged['right'] - df_rows_merged['left']
    df_rows_merged['height'] = df_rows_merged['bottom'] - df_rows_merged['top']

    df_rows_merged.drop(['right', 'bottom'], axis=1, inplace=True)

    df_nearby_rows_combined = merge_df_rows_by_height(df_rows_merged)

    return df_nearby_rows_combined


def extract_text_from_image(img, cmd_config='', remove_chars='[^a-zA-Z0-9+é]+'):
    def ensure_text_is_string(value):
        try:
            # Try converting the value to a float
            number = float(value)
            # If successful, return the whole number part as a string
            return str(int(number))
        except ValueError:
            # If conversion fails (i.e., value is not a number), return value
            return str(value)

    if len(cmd_config) > 0:
        df_words = pytesseract.image_to_data(img, output_type='data.frame', config=cmd_config)
    else:
        df_words = pytesseract.image_to_data(img, output_type='data.frame')
    df_words = df_words.replace({remove_chars: ''}, regex=True)
    df_words = df_words.replace({r'é': 'e'}, regex=True)  # For app name "Pokémon GO", etc.
    df_words = df_words[df_words['conf'] > 0]
    df_words = df_words[(df_words['text'] != '') & (df_words['text'] != ' ')]
    df_words = df_words.fillna('')
    df_words['text'] = (df_words['text'].apply(ensure_text_is_string))
    df_words = df_words[~df_words['text'].str.contains('^[aemu]+$')] if df_words.shape[0] > 0 else df_words

    df_lines = merge_df_rows_by_line_num(df_words)

    return df_words, df_lines


def show_text_on_image(df, img, draw_boxes=True):
    # Show where text is found on an image (mostly for debugging)

    img_height, img_width = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    scale = 1080 / img_height if img_height > 1080 else 1  # Ensures the image fits on a 1920x1080 screen
    df[['left', 'width', 'top', 'height']] = df[['left', 'width', 'top', 'height']].astype(int)

    if draw_boxes:
        m = 5  # A small margin to expand the borders of the boxes away from the text
        for i in df.index:
            start_point = (df['left'][i] - m, df['top'][i] - m)
            end_point = (df['left'][i] + df['width'][i] + m,
                         df['top'][i] + df['height'][i] + m)

            conf_p = df['conf'][i] / 100  # confidence value for each text item, expressed as a proportion from 0 to 1
            box_color_to_paint = (int(255 * conf_p), int(127 * conf_p), 255 - int(255 * conf_p))

            thick = int(2 / scale) if df['conf'][i] >= conf_limit else int(1 / scale)
            cv2.rectangle(img, start_point, end_point, box_color_to_paint, thickness=thick)
            # Border colour indicates confidence level. More blue = more confident; more red = less confident.

    print("\nText found in image:")
    print(df[['left', 'top', 'width', 'height', 'conf', 'text']])

    scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Text found in image", scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def determine_language_of_image(participant, df):
    user_default_lang = participant.language if participant.language is not None else default_language
    print("Detecting language: ", end='')

    if df.shape[0] <= 1:
        print(f"No text detected. Defaulting to {user_default_lang}.")
        return user_default_lang
    for key, val in LANGUAGE_KEYWORDS.items():
        if df['text'].str.contains('|'.join(val)).any():
            print(f"{key}")
            return key

    print(f"Unknown. Defaulting to {user_default_lang}.")
    return user_default_lang


def get_or_create_participant(users, test_user_id):
    for u in users:
        if u.id == test_user_id:
            print(f"Found existing user: {u}")
            return u

    # If no matching participant is found, create a new one
    new_user = Participant(test_user_id)
    users.append(new_user)
    print(f"New user created: {new_user}")
    return new_user


if __name__ == '__main__':
    if not os.path.exists(dir_for_downloaded_images):
        os.makedirs(dir_for_downloaded_images)

    # Read in the list of URLs for the appropriate Study (as specified in RuntimeValues.py)
    url_list = pd.DataFrame()
    for survey in survey_list:
        print(f"Compiling URLs from {survey[CSV_FILE]}...", end='')
        survey_csv = pd.read_csv(survey[CSV_FILE])
        current_list = compile_list_of_urls(survey_csv, survey[SCREEN_COLS], survey[PICKUP_COLS], survey[NOTIF_COLS])
        print(f"Done.\n{current_list.shape[0]} URLs found.")
        url_list = pd.concat([url_list, current_list], ignore_index=True)

    num_urls = url_list.shape[0]
    print(f'Total URLs from all surveys: {num_urls}')

    # Cycle through the images, creating a screenshot object for each one
    screenshots = []
    participants = []
    for index in url_list.index:
        if not (test_lower_bound <= index+1 <= test_upper_bound):
            # Only extract data from the images within the bounds specified in RuntimeValues.py
            continue

        current_screenshot = Screenshot(url=url_list[IMG_URL][index],
                                        user_id=url_list[PARTICIPANT_ID][index],
                                        date=url_list[RESPONSE_DATE][index],
                                        category=url_list[IMG_RESPONSE_TYPE][index])
        print(f"File {index+1} of {num_urls}:\n{current_screenshot}")

        screenshots.append(current_screenshot)
        current_participant = get_or_create_participant(participants, current_screenshot.user_id)

        # Download the image (if not using local images) or open the local image
        grey_image, bw_image = load_and_process_image(current_screenshot, white_threshold=226)
        current_screenshot.set_image(grey_image)

        current_screenshot.is_light_mode = True if np.mean(grey_image) > 170 else False
        # Light-mode images have an average pixel brightness above 170 (scale 0 to 255).
        text_df_single_words, text_df = extract_text_from_image(bw_image)

        if show_images:
            show_text_on_image(text_df, bw_image, draw_boxes=True)

        if text_df.shape[0] == 0:
            print(f"No text found.  Setting {current_screenshot.category_submitted} values to {NO_NUMBER}.")
            # update_all_columns_for_empty_screenshot(reason='No text found')
            continue

        current_screenshot.set_dimensions(bw_image.shape)
        image_language = determine_language_of_image(current_participant, text_df)
        current_screenshot.set_language(image_language)
        current_participant.set_language(image_language)