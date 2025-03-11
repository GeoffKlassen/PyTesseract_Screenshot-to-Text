import hashlib
from os import truncate

from pandas.core.methods.selectn import SelectNSeries

import AndroidFunctions as Android
import ConvenienceVariables
import ScreenshotClass
import iOSFunctions as iOS
from RuntimeValues import *
from RuntimeValues import app_area_scale_factor
from ScreenshotClass import Screenshot
from ParticipantClass import Participant
from ConvenienceVariables import *
import os
import re
import numpy as np
import pandas as pd
import pytesseract
import cv2
import requests
from PIL import Image
from io import BytesIO
import warnings
from datetime import datetime, timedelta
import time


def compile_list_of_urls(df, url_cols,
                         date_col='Record Time', id_col='Participant ID', device_id_col='Device ID'):
    """Create a DataFrame of URLs from a provided DataFrame of survey responses.
    :param df:             The dataframe with columns that contain URLs
    :param url_cols:       A dictionary of column names for screentime URLs, pickups URLs, and notifications URLs
    :param date_col:       The name of the column in df that lists the date & time stamp of submission (For Avicenna CSVs, this is usually 'Record Time')
    :param id_col:         The name of the column in df that contains the Avicenna ID of the user (For Avicenna CSVs, this is usually 'Participant ID')
    :param device_id_col:  The name of the column in df that contains the ID of the device from which the user responded (For Avicenna CSVs, this is usually 'Device ID')

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
        """

        :param img_path:
        :return:
        """
        empty_arr = np.array([])
        if "https" not in img_path and "avicennaresearch" not in img_path:
            print("Skipping. Invalid URL.")
            return empty_arr
        test_url = img_path.replace("https", "http")
        print("Downloading image from URL...")
        # The first request without auth headers to check what the redirected URL is.
        response = requests.get(test_url, verify=True)
        actual_url = response.url
        if "https" in actual_url:
            response = requests.get(img_path, verify=True, auth=(user, passw))
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # We'll standardize the image to RGB format.
                if save_downloaded_images:
                    img.save(img_local_path)
                    print(f"Image saved to '{dir_for_downloaded_images}\\{screenshot.device_os_submitted}\\{screenshot.filename}'.")
                    img = Image.open(img_local_path)
                else:
                    img.save(img_temp_path)
                    img = Image.open(img_temp_path)
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

    if use_downloaded_images or save_downloaded_images:
        if not os.path.exists(dir_for_downloaded_images):
            os.makedirs(dir_for_downloaded_images)
        if not os.path.exists(f"{dir_for_downloaded_images}\\{screenshot.device_os_submitted}"):
            os.makedirs(f"{dir_for_downloaded_images}\\{screenshot.device_os_submitted}")
    img_local_path = os.path.join(dir_for_downloaded_images, screenshot.device_os_submitted, screenshot.filename)
    img_temp_path = os.path.join(dir_for_downloaded_images, screenshot.device_os_submitted, "temp.jpg")

    if use_downloaded_images:
        if os.path.exists(img_local_path):
            print(f"Opening local image '{dir_for_downloaded_images}\\{screenshot.device_os_submitted}\\{screenshot.filename}'...")
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
    brown_bg_threshold = 160
    white_bg_threshold = 170
    average_pixel_colour = np.mean(image)
    if average_pixel_colour > white_bg_threshold:
        bg_colour = WHITE
    elif average_pixel_colour < brown_bg_threshold:
        bg_colour = BLACK
    else:
        bg_colour = BROWN

    remove_color_blocks_from_image(image, (bg_colour == WHITE))

    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To gray scale.
    if bg_colour == WHITE:
        # threshold arguments: image, threshold value, default to use if value > threshold. type of threshold to apply.
        #       for last argument: can be cv2.THRESH_BINARY or cv2.THRESH_OTSU
        # IMPORTANT: Looks like the best threshold value for a light mode image is 206. # GK: 230 makes text thicker,
        # but also makes the 'thermometer' shaped bars (below each app name) more likely to show up
        bw_threshold = white_threshold if screenshot.device_os_detected == IOS else 200
        max_value = 255
    elif bg_colour == BLACK:
        # Settings for dark mode.
        bw_threshold = black_threshold if screenshot.device_os_detected == ANDROID else 70
        max_value = 180
    else:
        # Settings for 'brown' mode.
        bw_threshold = 140
        max_value = 255

    (_, bw_img) = cv2.threshold(grey_img, bw_threshold, max_value, cv2.THRESH_BINARY)

    return grey_img, bw_img


def levenshtein_distance(s1, s2):
    """
    Determines the number of character insertions/deletions/substitutions required to transform s1 into s2.
    :param s1: (String) One of the strings
    :param s2: (String) The other string
    :return: (int) The distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]


def get_date_regex(lang, fmt=DATE_FORMAT):
    """
    Creates the date regular expression to use when looking for text in a screenshot that matches a date format.
    :param lang: The language to use for month abbreviations
    :param fmt: The date type to create a regex for (DATE_FORMAT or DATE_RANGE_FORMAT)
    :return: The full date regex of all possible date formats for the given language
    """
    patterns = []
    for _format in fmt[lang]:
        # Replace the 'MMM's in DATE_FORMAT with the appropriate 3-4 letter abbreviations for the months.
        patterns.append(re.sub('MMM', ''.join(['(', '|'.join(MONTH_ABBREVIATIONS[lang]), ')']), _format))
    date_regex = '|'.join(patterns)
    return date_regex


def merge_df_rows_by_height(df):
    """

    :param df:
    :return:
    """
    # Sometimes two 'words' that are side-by-side on the screenshot end up on their own lines in the df.
    # Merge these rows into a single row (combine their 'text' values, average their 'conf' values, and
    # increase the 'width' value of the first word).

    df['right'] = df['left'] + df['width']
    df = df.sort_values(by=['top', 'left']).reset_index(drop=True)
    rows_to_drop = []
    for i in df.index:
        if i == 0:
            continue
        if abs(df.loc[i]['top'] - df.loc[i - 1]['top']) < 15:  # and (df.loc[i]['text'] != "X"):  # Not sure why I did the "X" check
            # If two rows' heights are within 15 pixels of each other and the current row is not "X" (Twitter)
            # TODO replace 15 with a percentage of the screenshot width (define moe = x% of screenshot width)
            if df.loc[i]['left'] > df.loc[i - 1]['left']:
                df.at[i - 1, 'text'] = df.loc[i - 1]['text'] + ' ' + df.loc[i]['text']
                # df.at[i - 1, 'width'] = max(df.loc[i]['right'], df.loc[i - 1]['right']) - min(df.loc[i]['left'],
                #                                                                               df.loc[i - 1]['right'])
                df.at[i - 1, 'width'] = df['right'][i] - df['left'][i - 1]
                df.at[i - 1, 'conf'] = (df.loc[i - 1]['conf'] + df.loc[i]['conf']) / 2
                rows_to_drop.append(i)
            elif df.loc[i - 1]['left'] > df.loc[i]['left']:
                df.at[i, 'text'] = df.loc[i]['text'] + ' ' + df.loc[i - 1]['text']
                # df.at[i, 'width'] = max(df.loc[i]['right'], df.loc[i - 1]['right']) - min(df.loc[i]['left'],
                #                                                                           df.loc[i - 1]['right'])
                df.at[i, 'width'] = df['right'][i - 1] - df['left'][i]
                df.at[i, 'conf'] = (df.loc[i]['conf'] + df.loc[i - 1]['conf']) / 2
                rows_to_drop.append(i - 1)

    if 'level' in df.columns:
        df.drop(columns=['level'], inplace=True)
    if 'level_0' in df.columns:
        df.drop(columns=['level_0'], inplace=True)
    consolidated_df = df.drop(index=rows_to_drop).reset_index()

    return consolidated_df


def merge_df_rows_by_line_num(df):
    """

    :param df:
    :return:
    """
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']


    if df['text'].eq("X").any():
        df['next_top'] = df['top'].shift(-1)
        df.loc[(df['text'] == "X") & (df['bottom'] < df['next_top']), 'line_num'] = 0

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


def extract_text_from_image(img, cmd_config='', remove_chars='[^a-zA-Z0-9+é]+', initial_scan=False):
    """

    :param img:
    :param cmd_config:
    :param remove_chars:
    :param initial_scan:
    :return:
    """
    def ensure_text_is_string(value):
        try:
            # Try converting the value to a float
            number = float(value)
            # If successful, return the whole number part as a string
            return str(int(number))
        except (ValueError, OverflowError):
            # ValueError in case 'value' is not a number
            # OverflowError in case 'value' is the word 'Infinity' (in which case, 'number' is now inf)
            return str(value)

    # img = cv2.GaussianBlur(img, (7,7), 0)  # Might help finding the large totals, not sure how it affects headings
    if len(cmd_config) > 0:
        df_words = pytesseract.image_to_data(img, output_type='data.frame', config=cmd_config)
    else:
        df_words = pytesseract.image_to_data(img, output_type='data.frame')
    df_words = df_words.replace({remove_chars: ''}, regex=True)
    df_words = df_words.replace({r'é': 'e'}, regex=True)  # For app name "Pokémon GO", etc.
    df_words = df_words.replace({r'^[xX]+\s?[xX]*$': 'X'}, regex=True)
    # df_words = df_words[~((df_words['text'] == 'X') & (df_words['left'] < int(0.15 * img.shape[1])))]
    df_words = df_words[df_words['conf'] > 0]
    df_words = df_words.fillna('')
    df_words = df_words[(df_words['text'] != '') & (df_words['text'] != ' ')]
    df_words['text'] = (df_words['text'].apply(ensure_text_is_string))
    df_words = df_words[~df_words['text'].str.contains('^[aemu]+$')] if df_words.shape[0] > 0 else df_words

    # Sometimes tesseract misreads (Italian) "Foto" as "mele"/"melee"
    df_words['text'] = df_words['text'].replace({r'^melee$|^mele$': 'Foto'}, regex=True)
    df_words.reset_index(drop=True, inplace=True)

    # To avoid multiple instances of the app name "X" getting merged into one, remove any duplicated row with "X" text
    x_rows_to_drop = []
    for idx in df_words.index:
        if idx == 0:
            continue
        else:
            if df_words['text'][idx] == "X":
                if df_words['text'][idx - 1] == "X":
                    if df_words['left'][idx - 1] < df_words['left'][idx]:
                        x_rows_to_drop.append(idx - 1)
                    elif df_words['left'][idx] < df_words['left'][idx - 1]:
                        x_rows_to_drop.append(idx)
                else:
                    if initial_scan and df_words['left'][idx] < int(0.2 * img.shape[1]):
                        df_words.loc[idx, 'left'] += int(2.5 * df_words['width'][idx])
                        df_words.loc[idx, 'top'] -= int(0.33 * df_words['height'][idx])
                        df_words.loc[idx, 'height'] = int(0.75 * df_words.loc[idx, 'height'])
                        # df_words.loc[idx, 'width'] = int(0.66 * df_words.loc[idx, 'width'])

    df_words.drop(index=x_rows_to_drop, inplace=True)
    df_words.reset_index(drop=True, inplace=True)

    df_lines = merge_df_rows_by_line_num(df_words)

    # ...or misreads ".AI" as ".Al"
    df_lines['text'] = df_lines['text'].replace({'.Al': '.AI'})
    df_lines['text'] = df_lines['text'].replace({'openal.com': 'OpenAI.com'})

    # def add_one_to_hr(_s):
    #     if bool(re.search(r"^hr\s", _s)):
    #         return "1 " + _s
    #     return _s
    # df_lines['text'] = df_lines['text'].apply(add_one_to_hr)

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
        m = int(6 / scale)  # A small margin to expand the borders of the boxes away from the text
        for i in df.index:
            start_point = (df['left'][i] - m, df['top'][i] - m)
            end_point = (df['left'][i] + df['width'][i] + m,
                         df['top'][i] + df['height'][i] + m)

            conf_p = df['conf'][i] / 100  # confidence value for each text item, expressed as a proportion from 0 to 1
            box_color_to_paint = (int(255 * conf_p), int(127 * conf_p), 255 - int(255 * conf_p))

            thick = int(4 / scale) if df['conf'][i] >= conf_limit else int(2 / scale)
            cv2.rectangle(img, start_point, end_point, box_color_to_paint, thickness=thick)
            # Border colour indicates confidence level. More blue = more confident; more red = less confident.

    print("\nText found in image:")
    print(df[['left', 'top', 'width', 'height', 'conf', 'text']])

    scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Text found in image", scaled_img)
    # if cv2.waitKey(1000) & 0xFF == ord('q'):
    #     pass
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def determine_language_of_image(participant, df):
    """

    :param participant:
    :param df:
    :return:
    """
    backup_lang = participant.language if participant.language is not None else default_language
    backup_lang_msg = f"Setting image language to {'study' if participant.language is None else 'user'} default ({backup_lang})."
    user_lang_exists = True if participant.language is not None else False

    if df.shape[0] <= 1:
        print(f"No text detected. {backup_lang_msg}")
        return backup_lang, False
    for key, _list in LANGUAGE_KEYWORDS.items():
        for val in _list:
            if df['text'].str.contains(val).any():
                print(f"Language keyword detected: \"{val}\". Setting language to {key}.")
                return key, True

    print(f"No language keywords detected. {backup_lang_msg}")
    return backup_lang, user_lang_exists


def get_best_language(screenshot):
    """
    Determines the language to use when looking up keywords in language dictionaries. Returns (in descending order
    of availability) the screenshot language, the participant language, or the study default language.

    :param screenshot: The screenshot to find the best language for

    :return: (String) The language to use as the key for dictionaries
    """
    if screenshot.language is not None:
        return screenshot.language
    elif screenshot.participant.language is not None:
        return screenshot.participant.language
    else:
        return default_language


def get_date_in_screenshot(screenshot):
    """
    Checks if any text found in the given screenshot matches the appropriate date pattern based on the language of the
    image. If there's a match, return it.
    :param screenshot: The screenshot to search for a date.
    :return: The date-format value of the date found in the screenshot text (if any), otherwise 'None'.
    """
    empty_df = pd.DataFrame
    df = screenshot.text
    lang = get_best_language(screenshot)
    # Different languages display dates in different formats. Create the correct regex pattern for the date.
    date_pattern = get_date_regex(lang)
    week_pattern = get_date_regex(lang, fmt=DATE_RANGE_FORMAT)

    try:
        # Pull out the first row of df where the text contains the date regex
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dates_df = df[df['text'].str.contains(date_pattern, regex=True, case=False)]
            date_row_text = dates_df['text'].iloc[0]
            if bool(re.search(week_pattern, date_row_text)):
                print("Screenshot contains week info.")
                return None, empty_df

        # Extract the date, month, and day from that row of text, as strings
        date_detected = re.search(date_pattern, date_row_text, flags=re.IGNORECASE).group()
        month_detected = re.search(r'[a-zA-Z]+', date_detected).group().lower()
        day_detected = re.search(r'\d+', date_detected).group()

        # Create a translation dictionary to replace non-English month names with English ones.
        months_to_replace = MONTH_ABBREVIATIONS[lang]
        for i, abbr in enumerate(months_to_replace):
            month_detected = month_detected.replace(abbr, english_months[i])
        month_detected = month_detected[0:3]  # datetime.strptime (used below) requires month to be 3 characters

        try:
            # Convert the string date to a date object
            date_object = datetime.strptime(f"{day_detected} {month_detected} {datetime.now().year}", "%d %b %Y")
            # Note: the 'year' part of the date object will be replaced with the year that the screenshot was submitted

            # Get the numeric month value from the mapping
            month_numeric = month_mapping.get(month_detected)
            if month_numeric:
                # Construct the complete date with the year
                complete_date = date_object.replace(year=int(screenshot.date_submitted.year), month=month_numeric)
                if (screenshot.date_submitted - complete_date.date()).days < 0:
                    # In case a screenshot of a previous year is submitted after the new year, correct the year.
                    complete_date = date_object.replace(year=(int(screenshot.date_submitted.year) - 1))
                print(f"Date text detected: \"{date_row_text}\".  Setting date to {complete_date.date()}.")
                return complete_date.date(), dates_df
            else:
                print("Invalid month abbreviation.")
        except ValueError:
            print("Invalid date format.")
    except:
        print("No date text detected.")

    return None, empty_df


def get_day_type_in_screenshot(screenshot):
    """
    Determines whether the given screenshot contains daily data (today, yesterday, weekday) or weekly data.
    :param screenshot: The screenshot to find the date (range) for
    :returns: tuple: (A day identifier, and the row of text in the screenshot that contains the day identifier)
    """
    lang = get_best_language(screenshot)
    df = screenshot.text.copy()
    date_pattern = get_date_regex(lang)
    device_os = screenshot.device_os_detected

    moe_yesterday = round(np.log(max((len(key) for key in KEYWORDS_FOR_YESTERDAY[lang]))))
    moe_today = round(np.log(max((len(key) for key in KEYWORDS_FOR_TODAY[lang]))))
    moe_weekday = round(np.log(max((len(key) for key in KEYWORDS_FOR_WEEKDAY_NAMES[lang]))))
    moe_week_keyword = round(np.log(max((len(key) for key in KEYWORDS_FOR_WEEK[lang]))))
    moe_day_before_yesterday = round(np.log(max((len(key) for key in KEYWORDS_FOR_DAY_BEFORE_YESTERDAY[lang]))))
    # moe = 'margin of error'
    # Set how close a spelling can be to a keyword in order for that spelling to be considered the (misread) keyword.

    df['next_text'] = df['text'].shift(-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        rows_with_today = df[(df['text'].apply(
            # Row text is close to the word 'today', and
            #   (1) also contains date, or
            #   (2) the next row contains a date, or
            #   (3) is Android; and
            # Row text does not come close to 'rest of the day' (or its variants)
            lambda x: min(levenshtein_distance(row_word[:len(key)], key)
                          for row_word in str.split(x)[:2]
                          for key in KEYWORDS_FOR_TODAY[lang])) <= moe_today) &
                              ((device_os == ANDROID) |
                               (df['text'].str.contains(date_pattern, case=False)) |
                               (df['next_text'].str.contains(date_pattern, case=False))) & (
                                 df['text'].apply(lambda x: min(levenshtein_distance(x, key) for key in Android.KEYWORDS_FOR_REST_OF_THE_DAY[lang])) > moe_today)]

        rows_with_yesterday = df[(df['text'].apply(
            # Row contains yesterday, and (1) also contains date or (2) the next row contains a date, or (3) is Android
            # (In Android, rows with day names are not guaranteed to be followed by a date.)
            lambda x: min(levenshtein_distance(row_word[:len(key)], key)
                          for row_word in str.split(x)[0:1]
                          for key in KEYWORDS_FOR_YESTERDAY[lang])) <= moe_yesterday) &
                                 ((device_os == ANDROID) |
                                  (df['text'].str.contains(date_pattern, case=False)) |
                                  (df['next_text'].str.contains(date_pattern, case=False)))]

        rows_with_day_before_yesterday = df[(df['text'].apply(
            # Row contains 'day before yesterday'
            lambda x: min(levenshtein_distance(x, key)
                          for key in KEYWORDS_FOR_DAY_BEFORE_YESTERDAY[lang])) <= moe_day_before_yesterday)]
        # 'Day before yesterday' text may only appear in Android screenshots

        rows_with_weekday = df[(df['text'].apply(
            # Row contains name of weekday, and (1) also contains date, or (2) the next row contains a date
            lambda x: min(levenshtein_distance(str.split(x)[0], key)
                          for key in KEYWORDS_FOR_WEEKDAY_NAMES[lang])) <= moe_weekday) &
                               ((df['text'].str.contains(date_pattern, case=False)) |
                                (df['next_text'].str.contains(date_pattern, case=False)))]
        # Full (un-abbreviated) 'weekday' text may only appear in iOS screenshots'

        rows_with_week_keyword = df[(df['text'].apply(
            # Row contains one of the keywords for a week-format screenshot (e.g. Daily Average) (iOS only)
            lambda x: min(levenshtein_distance(x, key) for key in KEYWORDS_FOR_WEEK[lang])) <= moe_week_keyword)]

    if rows_with_yesterday.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_yesterday.iloc[0]['text']}\". "
              f"Setting image type to '{YESTERDAY}'.")
        return YESTERDAY, rows_with_yesterday
    elif rows_with_today.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_today.iloc[0]['text']}\". "
              f"Setting image type to '{TODAY}'.")
        return TODAY, rows_with_today
    elif rows_with_day_before_yesterday.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_day_before_yesterday.iloc[0]['text']}\". "
              f"Setting image type to '{DAY_BEFORE_YESTERDAY}.")
        return DAY_BEFORE_YESTERDAY, rows_with_day_before_yesterday
    elif rows_with_weekday.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_weekday.iloc[0]['text']}.\" "
              f"Setting image type to '{DAY_OF_THE_WEEK}'.")
        return DAY_OF_THE_WEEK, rows_with_weekday
    elif rows_with_week_keyword.shape[0] > 0:
        print(f"Week text detected: \"{rows_with_week_keyword.iloc[0]['text']}\". "
              f"Setting image type to '{WEEK}'.")
        return WEEK, rows_with_week_keyword
    else:
        print("No day/week text detected.")
        return None, None


def get_or_create_participant(users, user_id, dev_id):
    """

    :param users:
    :param user_id:
    :param dev_id:
    :return:
    """
    for u in users:
        if u.user_id == user_id:
            print(f"Found existing user: {u}")
            return u

    # If no matching participant is found, create a new one
    new_user = Participant(user_id=user_id, device_id=dev_id, device_os=get_os(dev_id))
    users.append(new_user)
    print(f"New user created: {new_user}")
    return new_user


def get_os(dev_id, screenshot=None):
    """ Returns the (almost certain) device OS based on the Device ID.

    :param dev_id: The Device ID of the device used to submit an image.
    :param screenshot: The current screenshot (available in case a flag needs to be added for the Device ID)
    :returns: The operating system of the device (iOS or Android); if unsure, returns 'Unknown'.

    As of February 7, 2025, all iPhone Device IDs in Avicenna CSVs appear as 32-digit hexadecimal numbers, and
    all Android device IDs in Avicenna CSVs appear as 16-digit hexadecimal (or sometimes alphanumeric) numbers.
    If a user submits a survey response via a web browser instead, the Device ID will reflect that browser:

    W_MACOSX_SAFARI             - macOS (MacBook or iMac)
    W_IOS_MOBILESAFARI          - iOS Safari browser
    W_ANDROID_CHROMEMOBILE      - Android Chrome browser
    W_ANDROID_FIREFOXMOBILE     - Android Firefox browser
    """
    if not bool(re.compile(r'^[0-9a-fA-F]+$').match(dev_id)) and screenshot is not None:
        screenshot.add_error(ERR_DEVICE_ID)
    if dev_id is None:
        return None
    elif len(dev_id) == 16 or ANDROID.upper() in dev_id:
        return ANDROID
    elif len(dev_id) == 32 or IOS.upper() in dev_id:
        return IOS
    else:
        # If the Device ID is a hexadecimal string that is not 16- or 32-digits, we can't guarantee the phone OS.
        if screenshot is not None:
            screenshot.add_error(ERR_DEVICE_ID)
        return UNKNOWN


def choose_between_two_values(text1, conf1, text2, conf2, value_is_number=False, val_fmt=None):
    """

    :param text1:
    :param conf1:
    :param text2:
    :param conf2:
    :param value_is_number:
    :param val_fmt:
    :return:
    """
    str_text1 = str(text1)
    str_text2 = str(text2)
    t1 = f"'{str_text1}'" if conf1 != NO_CONF else "N/A"
    t2 = f"'{str_text2}'" if conf2 != NO_CONF else "N/A"
    c1 = f"(conf = {conf1})" if conf1 != NO_CONF else ""
    c2 = f"(conf = {conf2})" if conf2 != NO_CONF else ""

    if val_fmt is None:
        val_fmt = misread_number_format_iOS if value_is_number else misread_time_format_iOS

    format_name = 'number' if value_is_number else 'time'

    print(f"Comparing scan 1: {t1} {c1}\n       vs scan 2: {t2} {c2}  ——  ", end='')
    if conf1 != NO_CONF and conf2 != NO_CONF:
        if bool(re.search(val_fmt, str_text1)) and bool(re.search(val_fmt, str_text2)) and str_text1 != str_text2:
            if str_text1 in str_text2:
                print(f"{t2} contains {t1}. Using {t2}.")
                return text2, conf2
            elif str_text2 in str_text1:
                print(f"{t1} contains {t2}. Keeping {t1}.")
                return text1, conf1
        if bool(re.search(val_fmt, str_text1)) and not bool(re.search(val_fmt, str_text2)):
            print(f"Only {t1} matches a proper {format_name} format. Keeping {t1}.")
            return text1, conf1
        elif not bool(re.search(val_fmt, str_text1)) and bool(re.search(val_fmt, str_text2)):
            print(f"Only {t2} matches a proper {format_name} format. Using {t2}.")
            return text2, conf2
        elif len(str_text1) > len(str_text2) and value_is_number:
            print(f"{t1} has more characters than {t2}. Keeping {t1}.")
            return text1, conf1
        elif len(str_text1) < len(str_text2) and value_is_number:
            print(f"{t2} has more characters than {t1}. Using {t2}.")
            return text2, conf2
        else:
            if conf1 > conf2:
                print("1st scan has higher confidence. Keeping 1st scan.")
                return text1, conf1
            elif conf1 < conf2:
                print("2nd scan has higher confidence. Using 2nd scan.")
                return text2, conf2
            else:
                print("Confidence values are equal. Keeping 1st scan.")
                return text2, conf2
    elif conf1 != NO_CONF:
        if value_is_number and not bool(re.search(val_fmt, str_text1)):
            print(f"Improper number format found in {t1}; no text found in {t2}.")
            return NO_NUMBER, NO_CONF
        print(f"No text found in 2nd scan. Keeping {t1}.")
        return text1, conf1
    elif conf2 != NO_CONF:
        if value_is_number and not bool(re.search(val_fmt, str_text2)):
            print(f"No text found in {t1}; improper number format found in {t2}.")
            return NO_NUMBER, NO_CONF
        print(f"No text found on 1st scan. Using {t2}.")
        return text2, conf2
    else:
        print("No text found on 1st or 2nd scan.")
        return NO_NUMBER, NO_CONF


def extract_app_info(screenshot, image, coordinates, scale):
    """

    :param screenshot:
    :param image:
    :param coordinates:
    :param scale:
    :return:
    """
    text = screenshot.text
    empty_text = text.iloc[0:0]
    bg_colour = WHITE if is_light_mode else BLACK
    lang = get_best_language(screenshot)
    crop_top, crop_left, crop_bottom, crop_right = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    remove_chars = r"[^a-zA-Z0-9+é:.!,()'&\-]+" if screenshot.device_os_detected == IOS else r"[^a-zA-Z0-9+é.!()'&\-]+"  # r"[^a-zA-Z0-9+()'\-.\<é]+"
    # Android needs characters like commas (,) removed because they appear in screentime values

    _, app_info_scan_1 = extract_text_from_image(image, remove_chars=remove_chars)

    # If the first text found in the cropped image is too far down/right in the image to be app info, then consider the
    # cropped region to NOT contain app info.
    # if screenshot.device_os_detected == IOS and not app_info_scan_1.empty and \
    #         (app_info_scan_1['top'][0] > 5 * app_info_scan_1['height'][0] or
    #          not re.fullmatch(misread_time_or_number_format, app_info_scan_1['text'][0]) and (
    #                  app_info_scan_1['left'][0] > int(0.25 * image.shape[1]) or
    #                  abs(app_info_scan_1['left'][0] + app_info_scan_1['width'][0]) < int(0.02 * image.shape[1]))):
    #     # First found row of text is more than 5x its own height from the top of the cropped image; or
    #     # First found row of text is not a time/number and either:
    #     #     it starts too far from the left edge of the cropped image, or
    #     #     it's too close to the right edge of the cropped image
    #
    #     print("First text found in cropped region is too far down/right. App-level data not found.")
    #     screenshot.add_error(ERR_APP_DATA)
    #     return empty_text

    """Sometimes the cropped rescan misses app numbers that were found on the initial scan.
                    Merge these app numbers from the initial scan into the rescan."""
    # Select only numbers from the initial scan that have high confidence (above conf_limit)
    # and that lie in the 'app info' cropped region
    # text.loc[text['text'].str.match(r'^[xX]+\s?[xX]*$'), 'text'] = 'X'

    truncated_text_df = text[(text['conf'] > 70) | (text['text'] == 'X')]
    truncated_text_df = truncated_text_df[(truncated_text_df['left'] > crop_left) &
                                          (truncated_text_df['top'] > crop_top) &
                                          (truncated_text_df['left'] < crop_right) &
                                          (truncated_text_df['top'] < crop_bottom) &
                                          ((truncated_text_df['text'].str.isdigit()) |
                                           (truncated_text_df['text'].str.fullmatch(time_format_long)) |
                                           (truncated_text_df['text'].str.fullmatch(misread_time_format_iOS))|
                                           (truncated_text_df['text'] == 'X'))]
    truncated_text_df.loc[truncated_text_df.index, 'left'] = truncated_text_df['left'] - crop_left
    truncated_text_df.loc[truncated_text_df.index, 'top'] = truncated_text_df['top'] - crop_top

    # if screenshot.device_os_detected == ANDROID and screenshot.android_version == GOOGLE:
    #     truncated_text_df = truncated_text_df[truncated_text_df['left'] + crop_left < int(0.5 * screenshot.width)]
    # truncated_text_df = OCRScript_v3.merge_df_rows_by_line_num(truncated_text_df)
    # Keep only the rows that contain only digits (a.k.a. notification counts or pickup counts) or 'X' (Twitter)
    # truncated_text_df = truncated_text_df[(truncated_text_df['text'].str.isdigit()) | (truncated_text_df['text'].str.match(r'[xX]{1,2}'))]

    print(f"\nApp numbers from initial scan, where conf > 0.5, plus any instances of X (Twitter):")
    print(truncated_text_df[['left', 'top', 'width', 'height', 'conf', 'text']])

    if app_info_scan_1['text'].eq("X").any() and truncated_text_df['text'].eq("X").any():
        # If both the initial scan and the first cropped scan found the app name 'X', only use the one in the cropped scan
        truncated_text_df = truncated_text_df[~(truncated_text_df['text'] == "X")]

    cols_to_scale = ['left', 'top', 'width', 'height']
    truncated_text_df[cols_to_scale] = truncated_text_df[cols_to_scale].apply(lambda x: x * scale).astype(int)
    overlapped_text = pd.concat([app_info_scan_1, truncated_text_df], ignore_index=True)
    app_info_scan_1 = iOS.consolidate_overlapping_text(overlapped_text) if screenshot.device_os_detected == IOS else (
        Android.consolidate_overlapping_text(overlapped_text, time_format_eol))
    app_info_scan_1 = app_info_scan_1.sort_values(by=['top', 'left']).reset_index(drop=True)

    index_of_day_axis = next((idx for idx, row in app_info_scan_1.iterrows() if
                              len(set(row['text'].split()).intersection(DAY_ABBREVIATIONS[lang])) >= 3), None)
    if index_of_day_axis is not None:
        # If the initial scan failed to find the 'DAY AXIS' row but it was found on the cropped region,
        # Erase the area above this DAY AXIS and remove any text above it.
        app_info_scan_1 = app_info_scan_1.iloc[index_of_day_axis + 1: ]
        image = cv2.rectangle(image, (0, 0), (screenshot.width, app_info_scan_1.iloc[0]['top']), bg_colour, -1)

    if show_images:
        show_image(app_info_scan_1, image)

    if screenshot.device_os_detected == ANDROID:
        if bg_colour == WHITE:
            bw_threshold = 210
            max_value = 255
        elif bg_colour == BLACK:
            bw_threshold = 75
            max_value = 180
        else:
            bw_threshold = 140
            max_value = 255

        cropped_grey_image = screenshot.grey_image[crop_top:crop_bottom, crop_left:crop_right]
        _, image_missed_text = cv2.threshold(cropped_grey_image, bw_threshold, max_value, cv2.THRESH_BINARY)  # Initialize
        image_missed_text = cv2.GaussianBlur(image_missed_text, ksize=ksize, sigmaX=0)
    else:
        image_missed_text = image.copy()

    # If there's confident text found in the cropped image, we 'erase' it to give pytesseract a better chance of reading
    # any text that it missed (or whose confidence wasn't high enough).
    if not app_info_scan_1.empty:
        if screenshot.device_os_detected == ANDROID:
            start_row = app_info_scan_1.iloc[0]['top']
            for r in range(start_row, 0, -1):
                if np.all(image[r, :] == image[r, 0]) and np.all(image[r, 0] != bg_colour):
                    image_missed_text = cv2.rectangle(image_missed_text, (0, 0), (image.shape[1], r), bg_colour, -1)
                    print("Horizontal area of solid colour found above first row of text; erasing that area.")
                    break

        for _i in app_info_scan_1.index:
            if app_info_scan_1['conf'][_i] < conf_limit:
                continue
            upper_left_corner = (app_info_scan_1['left'][_i] - 1, app_info_scan_1['top'][_i] - 1)
            bottom_right_corner = (app_info_scan_1['left'][_i] + app_info_scan_1['width'][_i] + 1,
                                   app_info_scan_1['top'][_i] + app_info_scan_1['height'][_i] + 1)
            cv2.rectangle(image_missed_text, upper_left_corner, bottom_right_corner, bg_colour, -1)

        app_info_names_only = app_info_scan_1[~app_info_scan_1['text'].str.fullmatch(misread_time_or_number_format)]

        if not app_info_names_only.empty:
            image_missed_text = cv2.rectangle(image_missed_text, (0, 0),
                                              (min(app_info_names_only['left']), image_missed_text.shape[0]), bg_colour, -1)

    _, app_info_scan_2 = extract_text_from_image(image_missed_text, remove_chars=remove_chars)

    if app_info_scan_2['text'].eq("X").any() and app_info_scan_1['text'].eq("X").any():
        # If both the initial scan and the first cropped scan found the app name 'X', only use the one in the cropped scan
        app_info_scan_1 = app_info_scan_1[~(app_info_scan_1['text'] == "X")]

    if show_images:
        show_image(app_info_scan_2, image_missed_text)

    app_info = pd.concat([app_info_scan_1, app_info_scan_2], ignore_index=True)
    app_info = iOS.consolidate_overlapping_text(app_info) if screenshot.device_os_detected == IOS else (
        Android.consolidate_overlapping_text(app_info, time_format_eol))

    # Sometimes the text for 'rest of the day' isn't found on the initial scan, but it gets found in the app-info scan.
    if screenshot.device_os_detected == ANDROID and screenshot.android_version == GOOGLE:
        rows_with_rest_of_the_day = app_info[app_info['text'].apply(lambda x: min(levenshtein_distance(x, key) for key in
                                                                    Android.KEYWORDS_FOR_REST_OF_THE_DAY[lang])) <= 2]
        if not rows_with_rest_of_the_day.empty:
            app_info = app_info[app_info.index > rows_with_rest_of_the_day.index[0]]

    # Sometimes the hours row isn't found on the initial scan, but it gets found in the app-info scan.
    if screenshot.device_os_detected == IOS:
        hours_axis_pattern = '|'.join(iOS.KEYWORDS_FOR_HOURS_AXIS)
        rows_with_hours_axis = app_info[app_info['text'].str.contains(hours_axis_pattern, regex=True)]
        if not rows_with_hours_axis.empty:
            app_info = app_info[app_info.index > rows_with_hours_axis.index[0]]

        app_info = app_info[app_info['left'] < int(0.95*image.shape[1])]

    app_info = app_info.reset_index(drop=True)

    return app_info


def get_dashboard_category(screenshot):
    """

    :param screenshot:
    :return:
    """
    dev_os = screenshot.device_os_detected
    heads_df = screenshot.headings_df
    # Get the category of data that is visible in the screenshot (Screen time, pickups, or notifications)
    if study_category is not None:
        # The category is normally 'None' at this point, but if the current study only requested one category
        # of screenshot, then we don't want to look for data from other categories.
        print(f"{study_to_analyze['Name']} study only requested screenshots of {study_category} data.  "
              f"Category set to '{study_category}'.")
        category_detected = True
        category = study_category
    elif not heads_df.empty:
        category = iOS.get_dashboard_category(screenshot) if dev_os == IOS else \
            Android.get_dashboard_category(screenshot)
        if category is None:
            category_detected = False
        else:
            category_detected = True

    else:
        current_screenshot.set_android_version(None)

        category_detected = False
        category = current_screenshot.category_submitted

    return category, category_detected


def update_eta(ss_start_time, idx):

    def convert_seconds_to_hms(sec):
        _hr = int(sec / 3600)
        _min = int((sec / 60) % 60)
        _sec = int(sec % 60)
        if _hr == 0:
            str_hr = ""
        else:
            str_hr = str(_hr) + ":"

        if _min == 0:
            str_min = f"{"00:" if _hr > 0 else "0:"}"
        elif 0 < _min < 10 and _hr > 0:
            str_min = "0" + str(_min) + ":"
        else:
            str_min = str(_min) + ":"

        if 0 <= _sec < 10:
            str_sec = "0" + str(_sec)
        else:
            str_sec = str(_sec)

        str_time = str_hr + str_min + str_sec
        return str_time

    current_time = time.time()
    elapsed_time_in_seconds = current_time - start_time
    ss_time = current_time - ss_start_time

    print(f"\n\nElapsed time:  {convert_seconds_to_hms(elapsed_time_in_seconds)}")

    all_times.loc[idx, 'time'] = ss_time
    all_times.loc[idx, 'elapsed_time'] = elapsed_time_in_seconds
    if not all_times.empty:
        average_time_per_screenshot = np.median(all_times['time'])
        estimated_time_remaining = (average_time_per_screenshot*0.8) * (min([test_upper_bound, num_urls]) - index - 1)
        all_times.loc[idx, 'eta'] = estimated_time_remaining
        if estimated_time_remaining > 0:
            print(f"Estimated time remaining:  {convert_seconds_to_hms(estimated_time_remaining)}")

    return


def add_screenshot_info_to_master_df(screenshot, idx):
    """

    :param screenshot:
    :param idx:
    :return:
    """
    same_or_other_user = None  # Initialize
    if screenshot.daily_total_conf == NO_CONF and \
            screenshot.app_data['name_conf'].eq(NO_CONF).all() and \
            screenshot.app_data['number_conf'].eq(NO_CONF).all():
        # Do not hash screenshots that contain no daily total or app-level info.
        current_screenshot_hash = None
    else:
        # Create a hash for the screenshot, using its detected data.
        string_to_hash = (str(screenshot.language) +
                          str(screenshot.device_os_detected) +
                          str(screenshot.android_version) +
                          str(screenshot.date_detected) +
                          str(screenshot.time_period) +
                          str(screenshot.category_detected) +
                          str(screenshot.daily_total))
        for n in range(1, max_apps_per_category + 1):
            string_to_hash += screenshot.app_data['name'][n] + str(screenshot.app_data['number'][n])
        current_screenshot_hash = hashlib.md5(string_to_hash.encode()).hexdigest()
        # Find all other screenshots with the same hash
        matching_screenshots = all_screenshots_df[(all_screenshots_df['hashed'] == current_screenshot_hash) &
                                                  (all_screenshots_df['hashed'] is not None)]
        if not matching_screenshots.empty:
            # There are other screenshots with the same data
            screenshot.add_error(ERR_DUPLICATE_DATA)
            if not (ERR_DUPLICATE_DATA in all_screenshots_df.columns):
                # Make sure the ERR_DUPLICATE_DATA column exists
                all_screenshots_df[ERR_DUPLICATE_DATA] = None
            if not (matching_screenshots['participant_id'].eq(screenshot.user_id).all()):
                same_or_other_user = "MULTIPLE USERS"
            else:
                same_or_other_user = "SAME USER"
            for n in matching_screenshots.index:
                if all_screenshots_df.loc[n, ERR_DUPLICATE_DATA] is None:
                    all_screenshots_df.loc[n, 'num_review_reasons'] += 1
            all_screenshots_df.loc[all_screenshots_df['hashed'] == current_screenshot_hash, ERR_DUPLICATE_DATA] = same_or_other_user

    all_screenshots_df.loc[idx, 'image_url'] = screenshot.url
    all_screenshots_df.loc[idx, 'participant_id'] = screenshot.user_id
    all_screenshots_df.loc[idx, 'device_id'] = screenshot.device_id
    all_screenshots_df.loc[idx, 'language'] = screenshot.language
    all_screenshots_df.loc[idx, 'device_os'] = screenshot.device_os_detected
    all_screenshots_df.loc[idx, 'android_version'] = screenshot.android_version
    all_screenshots_df.loc[idx, 'date_submitted'] = screenshot.date_submitted
    all_screenshots_df.loc[idx, 'date_detected'] = screenshot.date_detected
    all_screenshots_df.loc[idx, 'day_type'] = screenshot.time_period
    all_screenshots_df.loc[idx, 'category_submitted'] = screenshot.category_submitted
    all_screenshots_df.loc[idx, 'category_detected'] = PICKUPS if (
            screenshot.category_detected == UNLOCKS) else screenshot.category_detected
    all_screenshots_df.loc[idx, 'daily_total'] = screenshot.daily_total
    for n in range(1, max_apps_per_category + 1):
        all_screenshots_df.loc[idx, f'app_{n}_name'] = screenshot.app_data['name'][n]
        all_screenshots_df.loc[idx, f'app_{n}_number'] = screenshot.app_data['number'][n]

    all_screenshots_df.loc[idx, 'hashed'] = current_screenshot_hash
    all_screenshots_df.loc[idx, 'num_review_reasons'] = len(screenshot.errors)
    for col in screenshot.data_row.columns:
        if col == ERR_CONFIDENCE:
            all_screenshots_df.loc[idx, col] = screenshot.num_values_low_conf
        elif col == ERR_MISSING_VALUE:
            all_screenshots_df.loc[idx, col] = screenshot.num_missed_values
        elif col == ERR_DUPLICATE_DATA:
            all_screenshots_df.loc[idx, col] = same_or_other_user
        elif col.startswith("ERR"):
            all_screenshots_df.loc[idx, col] = True
        else:
            pass

    return


if __name__ == '__main__':
    # Read in the list of URLs for the appropriate Study (as specified in RuntimeValues.py)
    url_list = pd.DataFrame()
    for survey in survey_list:
        print(f"Compiling URLs from {survey[CSV_FILE]}...", end='')
        survey_csv = pd.read_csv(survey[CSV_FILE])
        current_list = compile_list_of_urls(survey_csv, survey[URL_COLUMNS],
                                            date_col=date_record_col_name,
                                            id_col=user_id_col_name,
                                            device_id_col=device_id_col_name)
        print(f"Done.\n{current_list.shape[0]} URLs found.")
        url_list = pd.concat([url_list, current_list], ignore_index=True)
        if len(url_list) >= test_upper_bound:
            print(f"URL list now contains at least {test_upper_bound} images. No further URLs needed.")
            break

    num_urls = url_list.shape[0]
    print(f'Total URLs from all surveys: {num_urls}')

    study_category = study_to_analyze[CATEGORIES][0] if study_to_analyze[CATEGORIES].__len__() == 1 else None
    # If the study we're analyzing only asked for one category of screenshot,
    # then we can ignore looking for the other categories.

    # Initialize an empty dataframe of app data
    empty_app_data = pd.DataFrame({
        'name': [NO_TEXT] * max_apps_per_category,
        'name_conf': [NO_CONF] * max_apps_per_category,
        'number': [NO_TEXT if study_category == SCREENTIME else NO_NUMBER] * max_apps_per_category,
        'number_conf': [NO_CONF] * max_apps_per_category,
        'minutes': [NO_TEXT] * max_apps_per_category
    })
    empty_app_data.index = pd.Index([idx + 1 for idx in empty_app_data.index])
    # All app_data should have indexes that start at 1 instead of starting at 0

    # Time the data extraction process
    start_time = time.time()
    all_times = pd.DataFrame(columns=['time', 'elapsed_time', 'eta'])

    all_screenshots_df = ScreenshotClass.initialize_data_row()

    participants = []
    test_upper_bound = test_lower_bound if test_upper_bound < test_lower_bound else test_upper_bound
    for index in url_list.index:
        if not (test_lower_bound <= index+1 <= test_upper_bound):
            # Only extract data from the images within the bounds specified in RuntimeValues.py
            continue

        min_url_index = min(num_urls, test_upper_bound)
        print(f"\n\nFile {index + 1} of {min_url_index}: {url_list[IMG_URL][index]}")

        screenshot_time_start = time.time()
        device_id = url_list[DEVICE_ID][index]
        # Load the participant for the current screenshot if they already exist, or create a new participant if not
        current_participant = get_or_create_participant(users=participants,
                                                        user_id=url_list[PARTICIPANT_ID][index],
                                                        dev_id=device_id)

        current_screenshot = Screenshot(participant=current_participant,
                                        url=url_list[IMG_URL][index],
                                        device_id=device_id,
                                        date=url_list[RESPONSE_DATE][index],
                                        category=url_list[IMG_RESPONSE_TYPE][index])
        device_os = get_os(device_id, current_screenshot)
        current_screenshot.set_device_os(device_os)

        print(current_screenshot)

        """ FOR ANDROID TESTING: SKIP iOS IMAGES"""
        # if current_screenshot.device_os == IOS:
        #     continue

        # Download the image (if not using local images) or open the local image
        grey_image, bw_image = load_and_process_image(current_screenshot, white_threshold=220)  # 226

        # If screenshot cannot be downloaded, set data for its submitted category to N/A
        if grey_image.size == 0:
            category_submitted = url_list[IMG_RESPONSE_TYPE][index]
            print(f"Error downloading file from URL. Setting {category_submitted} values to N/A.")
            current_screenshot.add_error(ERR_FILE_NOT_FOUND)
            current_screenshot.set_daily_total(NO_TEXT)
            if current_screenshot.category_submitted == SCREENTIME:
                current_screenshot.set_daily_total_minutes(NO_NUMBER)
            current_screenshot.set_app_data(empty_app_data)
            current_participant.add_screenshot_data(current_screenshot)

            add_screenshot_info_to_master_df(current_screenshot, index)
            update_eta(screenshot_time_start, index)  # Update the ETA without adding the current screenshot's time to the list
            continue

        is_light_mode = True if np.mean(grey_image) > 170 else False
        current_screenshot.set_is_light_mode(is_light_mode)
        # Light-mode images have an average pixel brightness above 170 (scale 0 to 255).

        # pytesseract does a better job of extracting text from images if the text isn't too big.
        if grey_image.shape[1] >= 2500:
            screenshot_scale_factor = 1 / 3
            ksize = (5, 5)
        elif grey_image.shape[1] >= 2000:
            screenshot_scale_factor = 1 / 2
            ksize = (3, 3)
        elif grey_image.shape[1] >= 1500:
            screenshot_scale_factor = 2 / 3
            ksize = (3, 3)
        elif grey_image.shape[1] >= 1000:
            screenshot_scale_factor = 1
            ksize = (1, 1)
        else:
            screenshot_scale_factor = 2
            ksize = (3, 3)

        grey_image_scaled = cv2.resize(grey_image,
                                       dsize=None,
                                       fx=screenshot_scale_factor,
                                       fy=screenshot_scale_factor,
                                       interpolation=cv2.INTER_AREA)
        bw_image_scaled = cv2.resize(bw_image,
                                     dsize=None,
                                     fx=screenshot_scale_factor,
                                     fy=screenshot_scale_factor,
                                     interpolation=cv2.INTER_AREA)
        if screenshot_scale_factor == 1 and ksize is not None:
            bw_image_scaled = cv2.GaussianBlur(bw_image_scaled, ksize=ksize, sigmaX=0)

        current_screenshot.set_scale_factor(screenshot_scale_factor)
        current_screenshot.set_image(grey_image_scaled)
        current_screenshot.set_dimensions(grey_image_scaled.shape)

        # Extract the text (if any) that can be found in the image.
        text_df_single_words, text_df = extract_text_from_image(bw_image_scaled, initial_scan=True)

        if show_images:
            show_image(text_df, bw_image_scaled, draw_boxes=True)

        if text_df.shape[0] == 0:
            print(f"No text found.  Setting {current_screenshot.category_submitted} values to N/A.")
            current_screenshot.add_error(ERR_NO_TEXT)
            current_screenshot.set_daily_total(NO_TEXT)
            if current_screenshot.category_submitted == SCREENTIME:
                current_screenshot.set_daily_total_minutes(NO_NUMBER)
            current_screenshot.set_app_data(empty_app_data)
            current_participant.add_screenshot_data(current_screenshot)

            add_screenshot_info_to_master_df(current_screenshot, index)
            update_eta(screenshot_time_start, index)
            continue

        # If there was text found, we can keep going
        print("Text found in initial scan:")
        print(text_df)
        current_screenshot.set_text(text_df)
        current_screenshot.set_words_df(text_df_single_words)

        # Get the language of the image, and assign that language to the screenshot & user (if a language was detected)
        image_language, language_was_detected = determine_language_of_image(current_participant, text_df)
        current_screenshot.set_language(image_language)
        current_screenshot.set_date_format(get_date_regex(image_language))
        if language_was_detected:
            current_participant.set_language(image_language)
        else:
            current_screenshot.add_error(ERR_LANGUAGE)

        time_formats = Android.get_time_formats_in_lang(current_screenshot.language)
        time_format_short, time_format_long, time_format_eol = time_formats[0], time_formats[1], time_formats[2]

        # Determine the date in the screenshot
        date_in_screenshot, rows_with_date = get_date_in_screenshot(current_screenshot)
        current_screenshot.set_date_detected(date_in_screenshot)
        current_screenshot.set_rows_with_date(rows_with_date)

        # Determine if the screenshot contains 'daily' data ('today', 'yesterday', etc.) or 'weekly' data
        day_type, rows_with_day_type = get_day_type_in_screenshot(current_screenshot)
        if day_type is not None:
            current_screenshot.set_time_period(day_type)
            current_screenshot.set_rows_with_day_type(rows_with_day_type)
        else:
            current_screenshot.add_error(ERR_DAY_TEXT)

        # Sometimes, the Device ID extracted from the Metadata is 16 hexadecimal digits long (which correlates with
        # Android images), but the screenshot is iOS, and iPhones have 32-digit hexadecimal Device IDs.
        # There are ways to catch if a screenshot is iOS though; one of the easiest ways is if the screenshot contains
        # the text "Updated today at", or "iPhone".
        # Other ways to tell are if the screenshot contains iOS-style headings but not Android-style headings.

        android_short_time_format, _, _ = Android.get_time_formats_in_lang(image_language)
        iOS_headings_df = iOS.get_headings(current_screenshot)
        android_headings_df = Android.get_headings(current_screenshot, android_short_time_format)

        num_iOS_headings = iOS_headings_df[HEADING_COLUMN].str.contains(IOS, na=False).sum()
        num_Android_headings = android_headings_df[HEADING_COLUMN].str.contains(ANDROID, na=False).sum()
        if current_screenshot.device_os_submitted == ANDROID and num_iOS_headings > num_Android_headings:
            print("Screenshot has Android-style Device ID but contains iOS headings. "
                  f"Setting device OS to '{IOS}'.")
            current_screenshot.set_device_os_detected(IOS)
            current_screenshot.add_error(ERR_DEVICE_OS)
        elif current_screenshot.device_os_submitted == IOS and num_Android_headings > num_iOS_headings:
            print("Screenshot has iOS-style Device ID but contains Android headings. "
                  f"Setting device OS to '{ANDROID}'.")
            current_screenshot.set_device_os_detected(ANDROID)
            current_screenshot.add_error(ERR_DEVICE_OS)

        if current_screenshot.device_os_detected == ANDROID:
            current_screenshot.set_time_formats(time_formats)

        """
            Here, the phone OS determines which branch of code we run to extract the daily total and app-level data.
        """

        if current_screenshot.device_os_detected == ANDROID:
            """
            
            ANDROID  -  Execute the procedure for extracting data from an Android screenshot  
            
            """

            # Determine if the screenshot contains data mis-considered relevant by participant -- if so, skip it
            if Android.screenshot_contains_unrelated_data(current_screenshot):
                current_screenshot.add_error(ERR_UNREADABLE_DATA)
                current_screenshot.set_daily_total(NO_TEXT if study_category == SCREENTIME else NO_NUMBER)
                if study_category == SCREENTIME:
                    current_screenshot.set_daily_total_minutes(NO_NUMBER)
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)
                update_eta(screenshot_time_start, index)  # Update the ETA w/o adding the current screenshot's time to the list
                continue

            if day_type is None and date_in_screenshot is not None:
                print(f"Determining day type based on number of days between submitted date "
                      f"({current_screenshot.date_submitted}) and date in screenshot ({date_in_screenshot}):")
                if current_screenshot.date_submitted == date_in_screenshot:
                    day_type = TODAY
                elif current_screenshot.date_submitted == date_in_screenshot + timedelta(days=1):
                    day_type = YESTERDAY
                elif current_screenshot.date_submitted == date_in_screenshot + timedelta(days=2):
                    day_type = DAY_BEFORE_YESTERDAY
                elif current_screenshot.date_submitted > date_in_screenshot:
                    day_type = DAY_OF_THE_WEEK
                print(f"Screenshot set to '{day_type}'.")
                current_screenshot.set_time_period(day_type)

            # Get headings from screenshot text
            headings_df = android_headings_df
            if headings_df.shape[0] > 0:
                print("\nHeadings found:")
                print(headings_df[['text', 'heading']])
                print()
            else:
                print("\nNo headings found.\n")

            current_screenshot.set_headings(headings_df)

            # Get which version of Android the screenshot is
            android_version = Android.get_android_version(current_screenshot)
            current_screenshot.set_android_version(android_version)

            dashboard_category, dashboard_category_detected = get_dashboard_category(current_screenshot)
            if dashboard_category_detected:
                current_screenshot.set_category_detected(dashboard_category)
            else:
                current_screenshot.add_error(ERR_CATEGORY)
                dashboard_category = current_screenshot.category_submitted

            daily_total, daily_total_conf = Android.get_daily_total_and_confidence(screenshot=current_screenshot,
                                                                                   image=bw_image_scaled,
                                                                                   heading=dashboard_category)
            if dashboard_category != SCREENTIME:
                try:
                    daily_total = str(int(daily_total))
                except ValueError:
                    print(f"Daily total {dashboard_category} '{daily_total}' is not a number. Resetting to N/A (confidence = {NO_CONF}).")
                    current_screenshot.add_error(ERR_NOT_A_NUMBER)
                    daily_total = NO_TEXT
                    daily_total_conf = NO_CONF

            current_screenshot.set_daily_total(daily_total, daily_total_conf)
            if daily_total_conf == NO_CONF:
                current_screenshot.add_error(ERR_DAILY_TOTAL)
                dt = "N/A"
            else:
                dt = daily_total

            dtm = ''  # Initialize
            if dashboard_category == SCREENTIME:
                daily_total_minutes = Android.convert_string_time_to_minutes(str_time=daily_total,
                                                                             screenshot=current_screenshot)
                dtm = (" (" + str(daily_total_minutes) + " minutes)") if daily_total_conf != NO_CONF else ""
                current_screenshot.set_daily_total_minutes(daily_total_minutes)
                print(f"Daily total {dashboard_category}: {dt}{dtm}")
            else:
                print(f"Daily total {dashboard_category}: {dt}")

            # For Samsung_2021 and 2018 versions of the dashboard, the Screentime heading and Notifications heading
            # both have sub-headings ('most used' and 'most notifications', respectively).
            if dashboard_category == SCREENTIME:
                heading_above_apps = MOST_USED_APPS_HEADING
                # Determine whether the row of text immediately above the app area is found
                # (used in ParticipantClass for comparing two screenshots from the same person & day & category)
                current_screenshot.set_screentime_subheading_found(False)
            elif dashboard_category == NOTIFICATIONS:
                heading_above_apps = MOST_NOTIFICATIONS_HEADING
                current_screenshot.set_notifications_subheading_found(False)
            else:  # dashboard_category == UNLOCKS, or no dashboard category
                heading_above_apps = None
                current_screenshot.set_pickups_subheading_found(False)

            # if the daily total is 0 (and not GOOGLE unlocks version), then there will be no app-level data to extract.
            if daily_total[0] in ['0', 'o', 'O'] and not (android_version == GOOGLE and dashboard_category == UNLOCKS):
                print(
                    f"No app-level data for {android_version} dashboard when daily total {dashboard_category} = 0. "
                    f"Skipping search for app-level data.")
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)

                screenshot_time = time.time() - screenshot_time_start

                update_eta(screenshot_time_start, index)
                continue

            if dashboard_category == UNLOCKS and android_version in [SAMSUNG_2024, SAMSUNG_2021, VERSION_2018]:
                print(f"{android_version} Dashboard does not contain app-level {dashboard_category} data. "
                      f"Skipping search for app data.")
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)

                screenshot_time = time.time() - screenshot_time_start

                update_eta(screenshot_time_start, index)
                continue

            # Crop the image to the app-specific region
            cropped_image, crop_coordinates = (
                Android.crop_image_to_app_area(image=bw_image_scaled,
                                               heading_above_apps=heading_above_apps,
                                               screenshot=current_screenshot,
                                               time_format_short=time_format_short))
            (app_area_crop_top, app_area_crop_left,
             app_area_crop_bottom, app_area_crop_right) = (crop_coordinates[0], crop_coordinates[1],
                                                           crop_coordinates[2], crop_coordinates[3])

            daily_total_heading_row = headings_df[headings_df[HEADING_COLUMN].str.fullmatch(f"total " + dashboard_category)]
            if all(crops is None for crops in crop_coordinates) or (not daily_total_heading_row.empty and app_area_crop_top < daily_total_heading_row.iloc[0]['top']):
                print(f"Crop region not found or includes daily total. Setting all app-specific data to N/A.")
                current_screenshot.add_error(ERR_APP_DATA)
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)

                screenshot_time = time.time() - screenshot_time_start

                update_eta(screenshot_time_start, index)
                continue

            app_area_crop_width = app_area_crop_right - app_area_crop_left

            cropped_image = cv2.GaussianBlur(cropped_image, ksize=(3, 3), sigmaX=0)
            scaled_cropped_image = cv2.resize(cropped_image,
                                              dsize=None,
                                              fx=app_area_scale_factor,
                                              fy=app_area_scale_factor,
                                              interpolation=cv2.INTER_AREA)

            # Extract app info from cropped image
            app_area_df = extract_app_info(current_screenshot, scaled_cropped_image, crop_coordinates, app_area_scale_factor)
            if android_version == GOOGLE:
                app_area_df = app_area_df[app_area_df['left'] < int(0.7 * app_area_crop_width)]

            if show_images:
                show_image(app_area_df, scaled_cropped_image)
            # app_area_df['text'] = app_area_df['text'].apply(lambda x: 'X' if re.match(r'[xX]{2}', x) else x)

            print("\nText found in app-area:")
            print(app_area_df[['left', 'top', 'width', 'height', 'conf', 'text']])
            # if dashboard_category is None and current_screenshot.category_detected is not None:
            #     # Sometimes there is screentime data in an image but the category is not detected.
            #     # If the cropped df contains enough rows that match a (misread) time format, set the dashboard category
            #     # to 'screentime'.
            #     dashboard_category = current_screenshot.category_detected

            # Sort the app-specific data into app names and app usage numbers
            app_data = Android.get_app_names_and_numbers(screenshot=current_screenshot,
                                                         df=app_area_df,
                                                         category=dashboard_category,
                                                         max_apps=max_apps_per_category,
                                                         time_formats=time_formats,
                                                         coordinates=crop_coordinates)

            dt = current_screenshot.daily_total if str(current_screenshot.daily_total) != NO_TEXT else "N/A"
            if dashboard_category == SCREENTIME:
                for i in range(1, max_apps_per_category + 1):
                    if i in app_data.index:  # Make sure the index exists
                        app_data.loc[i, 'minutes'] = Android.convert_string_time_to_minutes(
                            str_time=app_data.loc[i, 'number'],
                            screenshot=current_screenshot)
                app_data['minutes'] = app_data['minutes'].astype(int)

                print("\nApp data found:")
                print(app_data[['name', 'number', 'minutes']])
                print(f"Daily total {dashboard_category}: {dt} {dtm}")
                if current_screenshot.daily_total_minutes is not None and current_screenshot.daily_total_minutes != -1:
                    sum_app_minutes = app_data[app_data['minutes'] != NO_CONF]['minutes'].astype(int).sum()
                    if int(current_screenshot.daily_total_minutes) < sum_app_minutes:
                        current_screenshot.add_error(ERR_TOTAL_BELOW_APP_SUM)

            else:
                print("\nApp data found:")
                print(app_data[['name', 'number']])
                print(f"Daily total {dashboard_category}: {dt}")
                if current_screenshot.daily_total is not None and \
                        current_screenshot.daily_total != -1 and \
                        current_screenshot.category_detected != UNLOCKS:
                    # Android does not calculate daily unlocks as the sum of the times each app was opened.
                    # Apps can be opened more than once per unlock.
                    sum_app_numbers = app_data[app_data['number'] != NO_CONF]['number'].astype(int).sum()
                    if int(current_screenshot.daily_total) < sum_app_numbers:
                        print(current_screenshot.category_detected)
                        current_screenshot.add_error(ERR_TOTAL_BELOW_APP_SUM)

            current_screenshot.set_app_data(app_data)
            current_participant.add_screenshot_data(current_screenshot)

            # Collect some review-oriented statistics on the screenshot
            # Put the data from the screenshot into the master CSV for all screenshots

        elif current_screenshot.device_os_detected == IOS:
            """

            iOS  -  Execute the procedure for extracting data from an iOS screenshot  

            """
            # Find the rows in the screenshot that contain headings ("SCREEN TIME", "MOST USED", "PICKUPS", etc.)
            headings_df = iOS_headings_df
            if headings_df.shape[0] > 0:
                print("\nHeadings found:")
                print(headings_df[['text', 'heading']])
                print()
            else:
                print("\nNo headings found.\n")

            current_screenshot.set_headings(headings_df)

            dashboard_category, dashboard_category_detected = get_dashboard_category(current_screenshot)
            if dashboard_category_detected:
                current_screenshot.set_category_detected(dashboard_category)
            else:
                current_screenshot.add_error(ERR_CATEGORY)

            # for category in 'categories to search for' loop with 'category' as the 3rd input to function
            # if screentime is in the categories to search for ??
            daily_total, daily_total_conf = iOS.get_daily_total_and_confidence(current_screenshot,
                                                                               bw_image_scaled,
                                                                               dashboard_category)
            current_screenshot.set_daily_total(daily_total, daily_total_conf)
            if daily_total_conf == NO_CONF:
                dt = "N/A"
                if current_screenshot.total_heading_found and dashboard_category != PICKUPS:
                    current_screenshot.add_error(ERR_DAILY_TOTAL_MISSED)
            else:
                dt = daily_total
            dtm = ''

            if dashboard_category == SCREENTIME:
                # Get the daily total usage (if it's present in the screenshot)
                daily_total_minutes = iOS.convert_text_time_to_minutes(daily_total, current_screenshot)

                dtm = (" (" + str(daily_total_minutes) + " minutes)") if daily_total_conf != NO_CONF else ""
                print(f"Daily total {dashboard_category}: {dt}{dtm}")

                current_screenshot.set_daily_total_minutes(daily_total_minutes)
                headings_above_applist = [MOST_USED_HEADING]
                heading_below_applist = PICKUPS_HEADING

            elif dashboard_category == PICKUPS:

                daily_total_2nd_loc, daily_total_2nd_loc_conf = iOS.get_total_pickups_2nd_location(current_screenshot,
                                                                                                   bw_image_scaled)
                print("Comparing both locations for total pickups:")
                daily_total, daily_total_conf = choose_between_two_values(daily_total, daily_total_conf,
                                                                          daily_total_2nd_loc, daily_total_2nd_loc_conf,
                                                                          val_fmt=misread_number_format_iOS)
                current_screenshot.set_daily_total(daily_total, daily_total_conf)
                if daily_total_conf == NO_CONF:
                    if current_screenshot.total_heading_found:
                        current_screenshot.add_error(ERR_DAILY_TOTAL)
                    dt = "N/A"
                else:
                    dt = daily_total
                print(f"Daily total {dashboard_category}: {dt}")

                headings_above_applist = [FIRST_USED_AFTER_PICKUP_HEADING, FIRST_PICKUP_HEADING]
                heading_below_applist = NOTIFICATIONS_HEADING

            elif dashboard_category == NOTIFICATIONS:
                current_screenshot.set_daily_total(daily_total, daily_total_conf)
                print(f"Daily total {dashboard_category}: {dt}")

                headings_above_applist = [HOURS_AXIS_HEADING]
                heading_below_applist = ''

            else:
                headings_above_applist = ['']
                heading_below_applist = ''
                daily_total = NO_TEXT
                daily_total_conf = NO_CONF

            if str(daily_total) in ["0", "0s"]:
                print(f"No app-level data available when daily total {dashboard_category} is {daily_total}.")
                print(f"Setting all app-specific data to N/A.")
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)

                screenshot_time = time.time() - screenshot_time_start

                update_eta(screenshot_time_start, index)
                continue
            # Crop image to app region
            cropped_image, crop_coordinates = iOS.crop_image_to_app_area(current_screenshot, headings_above_applist, heading_below_applist)
            if all(crops is None for crops in crop_coordinates):
                print(f"Suitable crop region not detected. Setting all app-specific data to N/A.")
                current_screenshot.add_error(ERR_APP_DATA)
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)

                screenshot_time = time.time() - screenshot_time_start

                update_eta(screenshot_time_start, index)
                continue

            (app_area_crop_top, app_area_crop_left,
             app_area_crop_bottom, app_area_crop_right) = (crop_coordinates[0], crop_coordinates[1],
                                                           crop_coordinates[2], crop_coordinates[3])

            cropped_image = cv2.GaussianBlur(cropped_image, ksize=(3, 3), sigmaX=0)
            scaled_cropped_image = cv2.resize(cropped_image,
                                              dsize=None,
                                              fx=app_area_scale_factor,
                                              fy=app_area_scale_factor,
                                              interpolation=cv2.INTER_AREA)

            # Perform pre-scan to remove value bars below app names and fragments of app icons left of app names
            cropped_prescan_words, cropped_prescan_df = extract_text_from_image(cropped_image,
                                                                                remove_chars="[^a-zA-Z0-9+é:.!,()'&-]+")
            cropped_prescan_words['text'] = cropped_prescan_words['text'].astype(str)
            cropped_prescan_df['text'] = cropped_prescan_df['text'].astype(str)
            cropped_prescan_words = cropped_prescan_words[cropped_prescan_words['text'].str.fullmatch(r'[a-zA-Z0-9]+', na=False)]
            cropped_prescan_words = cropped_prescan_words[~((cropped_prescan_words['text'] == '2') &
                                                            ((cropped_prescan_words['left'] < int(0.1 * cropped_image.shape[1])) |
                                                             (cropped_prescan_words['width'] > cropped_prescan_words['height'])))]
            cropped_prescan_words = cropped_prescan_words.reset_index(drop=True)

            cropped_filtered_image = iOS.erase_value_bars_and_icons(screenshot=current_screenshot,
                                                                    df=cropped_prescan_words,
                                                                    image=cropped_image)

            scaled_cropped_filtered_image = cv2.resize(cropped_filtered_image,
                                                       dsize=None,
                                                       fx=app_area_scale_factor,
                                                       fy=app_area_scale_factor,
                                                       interpolation=cv2.INTER_AREA)

            # Extract app info from cropped image
            app_area_df = extract_app_info(current_screenshot, scaled_cropped_filtered_image, crop_coordinates, app_area_scale_factor)
            if ERR_APP_DATA in current_screenshot.errors:
                current_screenshot.set_app_data(empty_app_data)
                current_participant.add_screenshot_data(current_screenshot)

                add_screenshot_info_to_master_df(current_screenshot, index)

                update_eta(screenshot_time_start, index)
                continue

            value_format = misread_time_format_iOS if dashboard_category == SCREENTIME else misread_number_format_iOS
            confident_text_from_prescan = \
                cropped_prescan_df[(cropped_prescan_df['right'] > 0.05 * scaled_cropped_filtered_image.shape[1]) &
                                   ((cropped_prescan_df['conf'] > 80) |
                                    ((cropped_prescan_df['text'].str.fullmatch(value_format)) & (cropped_prescan_df['conf'] > 50))) |
                                   (cropped_prescan_df['text'].str.fullmatch('X'))]

            if app_area_df['text'].eq("X").any() and confident_text_from_prescan['text'].eq("X").any():
                # If both the initial scan and the first cropped scan found the app name 'X',
                # then only use the one in the cropped scan
                app_area_df = app_area_df[~(app_area_df['text'] == "X")]

            columns_to_scale = ['left', 'top', 'width', 'height']
            confident_text_from_prescan.loc[:, columns_to_scale] = \
                confident_text_from_prescan.loc[:, columns_to_scale].apply(lambda x: x * app_area_scale_factor).astype(int)
            app_area_2_df = iOS.consolidate_overlapping_text(
                pd.concat([app_area_df, confident_text_from_prescan], ignore_index=True))
            # Divide the extracted app info into app names and their numbers
            if show_images:
                show_image(app_area_2_df, scaled_cropped_image)

            # app_area_2_df['text'] = app_area_2_df['text'].apply(lambda x: 'X' if re.match(r'[xX]{2}', x) else x)

            print("\nText found in app-area:")
            print(app_area_2_df[['left', 'top', 'width', 'height', 'conf', 'text']])

            # if dashboard_category is None and current_screenshot.category_detected is not None:
            #     # Sometimes there is screentime data in an image but the category is not detected.
            #     # If the cropped df contains enough rows that match a (misread) time format, set the dashboard category
            #     # to 'screentime'.
            #     dashboard_category = current_screenshot.category_detected

            app_data = iOS.get_app_names_and_numbers(screenshot=current_screenshot,
                                                     crop_img=scaled_cropped_filtered_image,
                                                     df=app_area_2_df,
                                                     category=dashboard_category,
                                                     max_apps=max_apps_per_category)
            dt = current_screenshot.daily_total if current_screenshot.daily_total_conf != NO_CONF else "N/A"
            dtm = f"{' (' + str(current_screenshot.daily_total_minutes) + " minutes)" if current_screenshot.daily_total_conf != NO_CONF else ''}"
            if dashboard_category == SCREENTIME:
                for i in range(1, max_apps_per_category + 1):
                    if i in app_data.index:  # Make sure the index exists
                        app_data.loc[i, 'minutes'] = Android.convert_string_time_to_minutes(
                            str_time=app_data.loc[i, 'number'],
                            screenshot=current_screenshot)
                app_data['minutes'] = app_data['minutes'].astype(int)
                print("\nApp data found:")
                print(app_data[['name', 'number', 'minutes']])
                print(f"Daily total {dashboard_category}: {dt}{dtm}")

                # iOS Daily screentime can exceed the sum of the app times. Do not flag iOS screentime images.

            else:
                print("\nApp data found:")
                print(app_data[['name', 'number']])
                print(f"Daily total {dashboard_category}: {dt}")
                if current_screenshot.daily_total is not None and current_screenshot.daily_total != -1:
                    sum_app_numbers = app_data[app_data['number'] != NO_CONF]['number'].astype(int).sum()
                    if int(current_screenshot.daily_total) < sum_app_numbers:
                        current_screenshot.add_error(ERR_TOTAL_BELOW_APP_SUM)

            current_screenshot.set_app_data(app_data)
            current_participant.add_screenshot_data(current_screenshot)
            # And also give it to the Participant object, checking to see if data already exists for that day & category
            #   (if it does, run the function (within Participant?) to determine how to merge the two sets of data together)

        else:
            print("Operating System not detected.")
            current_screenshot.add_error(ERR_OS_NOT_FOUND)

            current_screenshot.set_daily_total(NO_TEXT)
            current_screenshot.set_app_data(empty_app_data)

            add_screenshot_info_to_master_df(current_screenshot, index)

            update_eta(screenshot_time_start, index)
            continue

        # Count the number of top-n apps/numbers/times whose confidence is below the confidence threshold
        count_below_conf_limit = app_data[['name_conf', 'number_conf']].map(
            lambda x: 0 < x < conf_limit).sum().sum() + (1 if daily_total_conf < conf_limit else 0)

        if count_below_conf_limit > 0:
            current_screenshot.add_error(ERR_CONFIDENCE, num=count_below_conf_limit)

        add_screenshot_info_to_master_df(current_screenshot, index)

        screenshot_time = time.time() - screenshot_time_start

        update_eta(screenshot_time_start, index)

        """ End of the for-loop of all URLs """

    total_elapsed_time = time.time() - start_time

    all_screenshots_df.index += 1  # So that the index lines up with the file number

    print("\nCompiling participants' temporal data...", end='')
    all_usage_dataframes = []  # Initialize
    for p in participants:
        all_usage_dataframes.append(p.usage_data)

    all_participants_df = pd.concat(all_usage_dataframes, ignore_index=True)
    all_participants_df = all_participants_df.sort_values(by=['participant_id', 'date']).reset_index(drop=True)

    print("Done.")

    if ERR_DUPLICATE_DATA in all_screenshots_df.columns:
        duplicate_screenshots_df = all_screenshots_df[all_screenshots_df[ERR_DUPLICATE_DATA].notna()]
        # Count occurrences of each participant_id
        if not duplicate_screenshots_df.empty:
            print("Compiling duplicate screenshot info...", end='')
            counts = duplicate_screenshots_df['participant_id'].value_counts()

            # Convert the result to a DataFrame
            counts_df = counts.reset_index()
            counts_df.columns = ['participant_id', 'count']

            counts_df.to_csv(f"{study_to_analyze['Name']}_duplicate_screenshot_info.csv")
            print("Done.")

    print("Exporting CSVs...", end='')
    all_screenshots_df.drop(columns=['hashed'], inplace=True)
    all_ios_screenshots_df = all_screenshots_df[all_screenshots_df['device_os'] == IOS]
    all_android_screenshots_df = all_screenshots_df[all_screenshots_df['device_os'] == ANDROID]

    all_screentime_screenshots_df = all_screenshots_df[all_screenshots_df['category_detected'] == SCREENTIME]
    all_pickups_screenshots_df = all_screenshots_df[(all_screenshots_df['category_detected'] == PICKUPS) |
                                                    (all_screenshots_df['category_detected'] == UNLOCKS)]
    all_notifications_screenshots_df = all_screenshots_df[all_screenshots_df['category_detected'] == NOTIFICATIONS]

    all_participants_df.to_csv(f"{study_to_analyze['Name']}_all_participants_temporal_data.csv")
    all_screenshots_df.to_csv(f"{study_to_analyze['Name']}_all_screenshots.csv")

    all_ios_screenshots_df.to_csv(f"{study_to_analyze['Name']}_all_ios_data.csv")
    all_android_screenshots_df.to_csv(f"{study_to_analyze['Name']}_all_android_data.csv")

    all_screentime_screenshots_df.to_csv(f"{study_to_analyze['Name']}_all_screentime_data.csv")
    all_pickups_screenshots_df.to_csv(f"{study_to_analyze['Name']}_all_pickups_data.csv")
    all_notifications_screenshots_df.to_csv(f"{study_to_analyze['Name']}_all_notifications_data.csv")

    all_times['actual_time_remaining'] = total_elapsed_time - all_times['elapsed_time']
    all_times.to_csv(f"{study_to_analyze['Name']}_all_ETAs.csv")  # Mostly for interest's sake
    print("Done.")
