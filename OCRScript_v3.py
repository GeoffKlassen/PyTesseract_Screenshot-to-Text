import Android
import iOS
from iOS import *
from Android import *
from ScreenshotClass import *
from ParticipantClass import *
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
DEVICE_ID = 'device_id'
IS_RESEARCHER = 'is_researcher'
RESPONSE_DATE = 'response_date'
IMG_RESPONSE_TYPE = 'img_response_type'
IMG_URL = 'img_url'

IOS = 'iOS'
ANDROID = 'Android'
UNKNOWN = 'Unknown'


def compile_list_of_urls(df, url_cols,
                         date_col='Record Time', id_col='Participant ID', device_id_col='Device ID'):
    """Create a DataFrame of URLs from a provided DataFrame of survey responses.
    :param df:                 The dataframe with columns that contain URLs
    :param url_cols:           A dictionary of column names for screentime URLs, pickups URLs, and notifications URLs
    :param date_col:           The name of the column in df that lists the date & time stamp of submission (For Avicenna CSVs, this is usually 'Record Time')
    :param id_col:             The name of the column in df that contains the Avicenna ID of the user (For Avicenna CSVs, this is usually 'Participant ID')
    :param device_id_col:      The name of the column in df that contains the ID of the device from which the user responded (For Avicenna CSVs, this is usually 'Device ID')

    :return: A DataFrame of image URLs, with user ID, response date, and category the image was submitted in.

    NOTE: iOS & Android Dashboards have three categories of usage statistics: screentime, pickups (a.k.a. unlocks), and
    notifications. Because of this, the table provided from the study may have multiple columns of URLs (e.g. one column
    for each usage category, multiple columns for only one category, or multiple columns for all three categories).
    """
    url_df = pd.DataFrame(columns=[PARTICIPANT_ID, DEVICE_ID, IS_RESEARCHER, RESPONSE_DATE, IMG_RESPONSE_TYPE, IMG_URL])
    for i in df.index:
        user_id = df[id_col][i]
        device_id = df[device_id_col][i]

        # Extract the response date from the date column, in date format
        try:
            response_date = datetime.strptime(str(df[date_col][i])[0:10], "%Y-%m-%d")  # YYYY-MM-DD = 10 chars
        except ValueError:  #
            response_date = datetime.strptime("1999-01-01", "%Y-%m-%d")  # TODO: try None?

        for key, col_list in url_cols.items():
            for col_name in col_list:
                url = str(df[col_name][i])
                img_response_type = key
                if not url.startswith("https://file.avicennaresearch.com"):
                    continue
                # Note: For the Boston Children's Hospital data, all images are of type SCREENTIME

                new_row = {PARTICIPANT_ID: user_id,
                           DEVICE_ID: device_id,
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
                    print(f"Image saved to local folder '{dir_for_downloaded_images}\\{screenshot.device_os}'.")
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

    if not os.path.exists(dir_for_downloaded_images):
        os.makedirs(dir_for_downloaded_images)
    if not os.path.exists(f"{dir_for_downloaded_images}\\{screenshot.device_os}"):
        os.makedirs(f"{dir_for_downloaded_images}\\{screenshot.device_os}")

    if use_downloaded_images:
        img_local_path = os.path.join(dir_for_downloaded_images, screenshot.device_os, screenshot.filename)
        if os.path.exists(img_local_path):
            print(f"Opening local image in '{dir_for_downloaded_images}\\{screenshot.device_os}'...")
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

    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To gray scale.
    if is_light_mode:
        # threshold arguments: image, threshold value, default to use if value > threshold. type of threshold to apply.
        #       for last argument: can be cv2.THRESH_BINARY or cv2.THRESH_OTSU
        # IMPORTANT: Looks like the best threshold value for a light mode image is 206. # GK: 230 makes text thicker,
        # but also makes the 'thermometer' shaped bars (below each app name) more likely to show up
        (_, bw_img) = cv2.threshold(grey_img, white_threshold, 255, cv2.THRESH_BINARY)
        return grey_img, bw_img
    else:
        # Settings for dark mode.
        (_, bw_img) = cv2.threshold(grey_img, black_threshold, 180, cv2.THRESH_BINARY)

    return grey_img, bw_img


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


def show_image(df, img, draw_boxes=True):
    """
    Opens a new window of the image currently being analyzed for data. Press any key to close the window.

    :param df: The dataframe that contains text extracted from the image (img) 
    :param img: The image from which the text in df is extracted
    :param draw_boxes: If True, coloured boxes are drawn around the text located in img.
    :return: None
    """
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
    backup_lang = participant.language if participant.language is not None else default_language
    backup_lang_msg = f"Setting image language to {'study' if participant.language is None else 'user'} default ({backup_lang})."
    user_lang_exists = True if participant.language is not None else False

    print("Detecting language: ", end='')
    if df.shape[0] <= 1:
        print(f"No text detected. {backup_lang_msg}")
        return backup_lang, False
    for key, val in LANGUAGE_KEYWORDS.items():
        if df['text'].str.contains('|'.join(val)).any():
            print(f"{key}")
            return key, True

    print(f"Unknown. {backup_lang_msg}")
    return backup_lang, user_lang_exists


def get_or_create_participant(users, user_id, dev_id):
    for u in users:
        if u.user_id == user_id:
            print(f"Found existing user: {u}")
            return u

    # If no matching participant is found, create a new one
    new_user = Participant(user_id=user_id, device_id=dev_id, device_os=get_os(dev_id))
    users.append(new_user)
    print(f"New user created: {new_user}")
    return new_user


def get_os(dev_id):
    """ Returns the (very likely) device OS based on the Device ID.

    Args:
        dev_id:    The Device ID of the device used to submit an image.

    Returns:
        The operating system (iOS or Android)

    As of February 7, 2025, all iPhone Device IDs in Avicenna CSVs appear as 32-digit hexadecimal numbers, and
    all Android device IDs in Avicenna CSVs appear as 16-digit hexadecimal numbers.
    When a user submits a survey response via a web browser, the Device ID will reflect that browser:
    W_MACOSX_SAFARI             - macOS (MacBook or iMac)
    W_IOS_MOBILESAFARI          - iOS Safari browser
    W_ANDROID_CHROMEMOBILE      - Android Chrome browser
    W_ANDROID_FIREFOXMOBILE     - Android Firefox browser
    """
    if dev_id is None:
        return None
    elif not bool(re.compile(r'^[0-9a-fA-F]+$').match(dev_id)):
        # If the Device ID is not a hexadecimal string, we cannot guarantee the survey response is from a phone.
        return UNKNOWN
    elif len(dev_id) == 16:
        return ANDROID
    elif len(dev_id) == 32:
        return IOS
    else:
        # If the Device ID is a hexadecimal string that is not 16- or 32-digits, we can't guarantee the phone OS.
        return UNKNOWN


# TODO This function has a more recent version in the CHB code.
#  Use that, and merge in the Android values for value_format.
"""
def choose_between_two_values(text1, conf1, text2, conf2):
    text1 = str(text1)
    text2 = str(text2)
    value_format = (USE misread_time_formats FROM IOS.PY) if \
        dashboard_category == SCREENTIME else r'^[0-9A]+$'
    format_name = 'time' if dashboard_category == SCREENTIME else 'number'

    if conf1 != NO_CONF and conf2 != NO_CONF:
        if bool(re.search(value_format, text1)) and not bool(re.search(value_format, text2)):
            print(f"Only 1st scan matches a proper {format_name} format. Keeping 1st scan.")
            return text1, conf1
        elif not bool(re.search(value_format, text1)) and bool(re.search(value_format, text2)):
            print(f"Only 2nd scan matches a proper {format_name} format. Using 2nd scan.")
            return text2, conf2
        elif len(text1) > len(text2):
            print("1st scan is longer than 2nd scan. Keeping 1st scan.")
            return text1, conf1
        elif len(text1) < len(text2):
            print("2nd scan is longer than 1st scan. Using 2nd scan.")
            return text2, conf2
        else:
            if conf1 > conf2:
                print("1st scan has higher confidence. Keeping 1st scan.")
                return text1, conf1
            else:
                print("2nd scan has higher confidence. Using 2nd scan.")
                return text2, conf2
    elif conf1 != NO_CONF:
        print("No text found on 2nd scan. Keeping 1st scan.")
        return text1, conf1
    elif conf2 != NO_CONF:
        print("No text found on 1st scan. Using 2nd scan.")
        return text2, conf2
    else:
        print("No text found on 1st or 2nd scan.")
        return NO_NUMBER, NO_CONF
"""


if __name__ == '__main__':
    # Read in the list of URLs for the appropriate Study (as specified in RuntimeValues.py)
    url_list = pd.DataFrame()
    for survey in survey_list:
        print(f"Compiling URLs from {survey[CSV_FILE]}...", end='')
        survey_csv = pd.read_csv(survey[CSV_FILE])
        current_list = compile_list_of_urls(survey_csv, survey[URL_COLUMNS])
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

        print(f"\n\nFile {index + 1} of {num_urls}: {url_list[IMG_URL][index]}")

        # Load the participant for the current screenshot if they already exist, or create a new participant if not
        current_participant = get_or_create_participant(users=participants,
                                                        user_id=url_list[PARTICIPANT_ID][index],
                                                        dev_id=url_list[DEVICE_ID][index])

        current_screenshot = Screenshot(url=url_list[IMG_URL][index],
                                        user_id=url_list[PARTICIPANT_ID][index],
                                        device_os=get_os(url_list[DEVICE_ID][index]),
                                        date=url_list[RESPONSE_DATE][index],
                                        category=url_list[IMG_RESPONSE_TYPE][index])

        # Add the current screenshot to the list of all screenshots
        screenshots.append(current_screenshot)
        # Download the image (if not using local images) or open the local image
        grey_image, bw_image = load_and_process_image(current_screenshot, white_threshold=226)
        current_screenshot.set_image(grey_image)

        current_screenshot.is_light_mode = True if np.mean(grey_image) > 170 else False
        # Light-mode images have an average pixel brightness above 170 (scale 0 to 255).
        text_df_single_words, text_df = extract_text_from_image(bw_image)

        if show_images:
            show_image(text_df, bw_image, draw_boxes=True)

        if text_df.shape[0] == 0:
            print(f"No text found.  Setting {current_screenshot.category_submitted} values to {NO_NUMBER}.")
            # update_all_columns_for_empty_screenshot(reason='No text found')
            continue

        current_screenshot.set_dimensions(bw_image.shape)

        # Get the language of the image, and assign that language to the screenshot & user (if a language was detected)
        image_language, language_was_detected = determine_language_of_image(current_participant, text_df)
        if language_was_detected:
            current_screenshot.set_language(image_language)
            current_participant.set_language(image_language)

        if current_screenshot.device_os == ANDROID:
            ## Perhaps all you need to do is copy the code for extracting Android data into the Android.py file ??
            Android.main()
            # use functions from Android.py
            # Return the extracted data

        elif current_screenshot.device_os == IOS:
            iOS.main()

            ## use functions from iOS.py
            # Return the extracted data
