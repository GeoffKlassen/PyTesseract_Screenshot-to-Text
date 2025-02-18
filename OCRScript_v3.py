import AndroidFunctions as Android
import iOSFunctions as iOS
from RuntimeValues import *
from ScreenshotClass import Screenshot
from ParticipantClass import Participant
from ConvenienceVariables import *
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
                    print(f"Image saved to '{dir_for_downloaded_images}\\{screenshot.device_os}\\{screenshot.filename}'.")
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
        if not os.path.exists(f"{dir_for_downloaded_images}\\{screenshot.device_os}"):
            os.makedirs(f"{dir_for_downloaded_images}\\{screenshot.device_os}")
    img_local_path = os.path.join(dir_for_downloaded_images, screenshot.device_os, screenshot.filename)
    img_temp_path = os.path.join(dir_for_downloaded_images, screenshot.device_os, "temp.jpg")

    if use_downloaded_images:
        if os.path.exists(img_local_path):
            print(f"Opening local image '{dir_for_downloaded_images}\\{screenshot.device_os}\\{screenshot.filename}'...")
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

    # img = cv2.GaussianBlur(img, (7,7), 0)  # Might help finding the large totals, not sure how it affects headings
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
    # if cv2.waitKey(1000) & 0xFF == ord('q'):
    #     pass
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def determine_language_of_image(participant, df):
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
    """ Returns the (almost certain) device OS based on the Device ID.

    :param dev_id: The Device ID of the device used to submit an image.

    :returns: The operating system of the device (iOS or Android); if unsure, returns 'Unknown'.

    As of February 7, 2025, all iPhone Device IDs in Avicenna CSVs appear as 32-digit hexadecimal numbers, and
    all Android device IDs in Avicenna CSVs appear as 16-digit hexadecimal numbers.
    If a user submits a survey response via a web browser instead, the Device ID will reflect that browser:

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


def choose_between_two_values(text1, conf1, text2, conf2, value_is_number=False):
    t1 = f"'{str(text1)}'" if conf1 != NO_CONF else "N/A"
    t2 = f"'{str(text2)}'" if conf2 != NO_CONF else "N/A"
    c1 = f"(conf = {conf1})" if conf1 != NO_CONF else ""
    c2 = f"(conf = {conf2})" if conf2 != NO_CONF else ""

    value_format = misread_number_format if value_is_number else misread_time_format
    format_name = 'number' if value_is_number else 'time'

    print(f"Comparing scan 1: {t1} {c1}\n       vs scan 2: {t2} {c2}  ——  ", end='')
    if conf1 != NO_CONF and conf2 != NO_CONF:
        if bool(re.search(value_format, text1)) and bool(re.search(value_format, text2)) and text1 != text2:
            if text1 in text2:
                print(f"{t2} contains {t1}. Using {t2}.")
                return text2, conf2
            elif text2 in text1:
                print(f"{t1} contains {t2}. Keeping {t1}.")
                return text1, conf1
        if bool(re.search(value_format, text1)) and not bool(re.search(value_format, text2)):
            print(f"Only {t1} matches a proper {format_name} format. Keeping {t1}.")
            return text1, conf1
        elif not bool(re.search(value_format, text1)) and bool(re.search(value_format, text2)):
            print(f"Only {t2} matches a proper {format_name} format. Using {t2}.")
            return text2, conf2
        elif len(text1) > len(text2) and value_is_number:
            print(f"{t1} has more characters than {t2}. Keeping {t1}.")
            return text1, conf1
        elif len(text1) < len(text2) and value_is_number:
            print(f"{t2} has more characters than {t1}. Using {t2}.")
            return text2, conf2
        else:
            if conf1 > conf2:
                print("1st scan has higher confidence. Keeping 1st scan.")
                return text1, conf1
            else:
                print("2nd scan has higher confidence. Using 2nd scan.")
                return text2, conf2
    elif conf1 != NO_CONF:
        if value_is_number and not bool(re.search(value_format, text1)):
            print(f"Improper number format found in {t1}; no text found in {t2}.")
            return NO_NUMBER, NO_CONF
        print(f"No text found in 2nd scan. Keeping {t1}.")
        return text1, conf1
    elif conf2 != NO_CONF:
        if value_is_number and not bool(re.search(value_format, text2)):
            print(f"No text found in {t1}; improper number format found in {t2}.")
            return NO_NUMBER, NO_CONF
        print(f"No text found on 1st scan. Using {t2}.")
        return text2, conf2
    else:
        print("No text found on 1st or 2nd scan.")
        return NO_NUMBER, NO_CONF


def extract_app_info(screenshot, image, scale):
    text = screenshot.text

    _, app_info_scan_1 = extract_text_from_image(image, remove_chars="[^a-zA-Z0-9+é:.!,()'&-]+")

    # paste the truncated text df stuff here
    """Sometimes the cropped rescan misses app numbers that were found on the initial scan.
                    Merge these app numbers from the initial scan into the rescan."""
    # Select only numbers from the initial scan that have high confidence (above conf_limit)
    # and that lie in the 'app info' cropped region
    truncated_text_df = text[text['conf'] > 0.5]
    truncated_text_df.loc[truncated_text_df.index, 'left'] = truncated_text_df['left'] - crop_left
    truncated_text_df.loc[truncated_text_df.index, 'top'] = truncated_text_df['top'] - crop_top
    truncated_text_df = truncated_text_df[(truncated_text_df['left'] > 0) &
                                          (truncated_text_df['top'] > 0) &
                                          (truncated_text_df['left'] + truncated_text_df[
                                              'width'] < crop_right - crop_left) &
                                          (truncated_text_df['top'] + truncated_text_df[
                                              'height'] < crop_bottom - crop_top)]
    # truncated_text_df = OCRScript_v3.merge_df_rows_by_line_num(truncated_text_df)
    # Keep only the rows that contain only digits (a.k.a. notification counts or pickup counts)
    truncated_text_df = truncated_text_df[truncated_text_df['text'].str.isdigit()]

    print(f"\nApp numbers from initial scan, where conf > 0.5:")
    print(truncated_text_df[['left', 'top', 'width', 'height', 'conf', 'text']])

    columns_to_scale = ['left', 'top', 'width', 'height']
    truncated_text_df[columns_to_scale] = truncated_text_df[columns_to_scale].apply(lambda x: x * scale).astype(int)
    app_info_scan_1 = iOS.consolidate_overlapping_text(pd.concat([app_info_scan_1, truncated_text_df], ignore_index=True))
    app_info_scan_1 = app_info_scan_1.sort_values(by=['top', 'left']).reset_index(drop=True)

    image_missed_text = image.copy()
    for i in app_info_scan_1.index:
        if app_info_scan_1['conf'][i] < conf_limit:
            continue
        upper_left_corner = (app_info_scan_1['left'][i], app_info_scan_1['top'][i])
        bottom_right_corner = (app_info_scan_1['left'][i] + app_info_scan_1['width'][i],
                               app_info_scan_1['top'][i] + app_info_scan_1['height'][i])
        bg_colour = (255, 255, 255) if is_light_mode else (0, 0, 0)
        cv2.rectangle(image_missed_text, upper_left_corner, bottom_right_corner, bg_colour, -1)

    _, app_info_scan_2 = extract_text_from_image(image_missed_text, remove_chars="[^a-zA-Z0-9+é:.!,()'&-]+")

    app_info = pd.concat([app_info_scan_1, app_info_scan_2]).sort_values(by = ['top', 'left']).reset_index(drop=True)
    app_info = iOS.consolidate_overlapping_text(app_info)

    return app_info


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

        current_screenshot = Screenshot(participant=current_participant,
                                        url=url_list[IMG_URL][index],
                                        device_os=get_os(url_list[DEVICE_ID][index]),
                                        date=url_list[RESPONSE_DATE][index],
                                        category=url_list[IMG_RESPONSE_TYPE][index])

        # Add the current screenshot to the list of all screenshots
        screenshots.append(current_screenshot)
        # Download the image (if not using local images) or open the local image
        grey_image, bw_image = load_and_process_image(current_screenshot, white_threshold=220)  # 226
        is_light_mode = True if np.mean(grey_image) > 170 else False
        current_screenshot.set_is_light_mode(is_light_mode)
        # Light-mode images have an average pixel brightness above 170 (scale 0 to 255).

        # pytesseract does a better job of extracting text from images if the text isn't too big.
        if grey_image.shape[1] >= 2500:
            screenshot_scale_factor = 1 / 3
        elif grey_image.shape[1] >= 2000:
            screenshot_scale_factor = 1 / 2
        elif grey_image.shape[1] >= 1500:
            screenshot_scale_factor = 2 / 3
        else:
            screenshot_scale_factor = 1

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

        current_screenshot.set_scale_factor(screenshot_scale_factor)
        current_screenshot.set_image(grey_image_scaled)
        current_screenshot.set_dimensions(grey_image_scaled.shape)

        # Extract the text (if any) that can be found in the image.
        text_df_single_words, text_df = extract_text_from_image(bw_image_scaled)

        if show_images:
            show_image(text_df, bw_image_scaled, draw_boxes=True)

        if text_df.shape[0] == 0:
            print(f"No text found.  Setting {current_screenshot.category_submitted} values to {NO_NUMBER}.")
            # update_all_columns_for_empty_screenshot(reason='No text found')
            continue

        current_screenshot.set_text(text_df)

        # Get the language of the image, and assign that language to the screenshot & user (if a language was detected)
        image_language, language_was_detected = determine_language_of_image(current_participant, text_df)
        if language_was_detected:
            current_screenshot.set_language(image_language)
            current_participant.set_language(image_language)
            current_screenshot.set_date_format(iOS.get_date_regex(image_language))

        if study_to_analyze[CATEGORIES].__len__() == 1:
            # If the study we're analyzing only asked for one category of screenshot,
            # then we can ignore looking for the other categories.
            dashboard_category = study_to_analyze[CATEGORIES][0]
        else:
            dashboard_category = None

        """ Here, the phone OS determines which branch of code we run to extract daily total and app-level data """

        if current_screenshot.device_os == ANDROID:
            Android.main()
            app_data = None  # Temporary until the android code is in place
            # use functions from AndroidFunctions.py
            # End up with a dataframe app_data[['app', 'number']]

        elif current_screenshot.device_os == IOS:
            """Execute the procedure for extracting data from an iOS screenshot"""

            # Determine the date in the screenshot
            date_in_screenshot = iOS.get_date_in_screenshot(current_screenshot)
            current_screenshot.set_date_detected(date_in_screenshot)

            # Determine if the screenshot contains 'daily' data ('today', 'yesterday', etc.) or 'weekly' data
            day_type, rows_with_day_type = iOS.get_day_type_in_screenshot(current_screenshot)
            current_screenshot.set_time_period(day_type)
            current_screenshot.set_rows_with_day_type(rows_with_day_type)

            # Find the rows in the screenshot that contain headings ("SCREEN TIME", "MOST USED", "PICKUPS", etc.)
            headings_df = iOS.get_headings(current_screenshot)
            current_screenshot.set_headings(headings_df)

            # Get the category of data that is visible in the screenshot (Screen time, pickups, or notifications)
            if dashboard_category is not None:
                print(f"{study_to_analyze['Name']} study only requested screenshots of {dashboard_category} data.  "
                      f"Category already set to '{dashboard_category}'.")
                dashboard_category_not_detected = False
            elif not headings_df.empty:
                dashboard_category = iOS.get_dashboard_category(current_screenshot)
                if dashboard_category is None:
                    dashboard_category_not_detected = True
                    dashboard_category = current_screenshot.category_submitted
                else:
                    dashboard_category_not_detected = False
            else:
                dashboard_category_not_detected = True
                dashboard_category = current_screenshot.category_submitted

            current_screenshot.set_category_detected(dashboard_category)

            # for category in 'categories to search for' loop with 'category' as the 3rd input to function
            # if screentime is in the categories to search for ??
            daily_total, daily_total_conf = iOS.get_daily_total_and_confidence(current_screenshot,
                                                                               bw_image_scaled,
                                                                               dashboard_category)
            current_screenshot.set_daily_total(daily_total, daily_total_conf)
            dt = "N/A" if daily_total_conf == NO_CONF else daily_total

            if dashboard_category == SCREENTIME:
                # Get the daily total usage (if it's present in the screenshot)
                daily_total_minutes = iOS.convert_text_time_to_minutes(daily_total)

                dtm = (" (" + str(daily_total_minutes) + " minutes)") if daily_total_conf != NO_CONF else ""
                print(f"Daily total {dashboard_category}: {dt}{dtm}")

                current_screenshot.set_daily_total_minutes(daily_total_minutes)
                heading_above_applist = MOST_USED_HEADING
                heading_below_applist = PICKUPS_HEADING

            elif dashboard_category == PICKUPS:

                daily_total_2nd_loc, daily_total_2nd_loc_conf = iOS.get_total_pickups_2nd_location(current_screenshot,
                                                                                                   bw_image_scaled)
                print("Comparing both locations for total pickups:")
                daily_total, daily_total_conf = choose_between_two_values(daily_total, daily_total_conf,
                                                                          daily_total_2nd_loc, daily_total_2nd_loc_conf)
                current_screenshot.set_daily_total(daily_total, daily_total_conf)
                print(f"Daily total {dashboard_category}: {dt}")

                heading_above_applist = FIRST_USED_AFTER_PICKUP_HEADING
                heading_below_applist = NOTIFICATIONS_HEADING

            elif dashboard_category == NOTIFICATIONS:
                current_screenshot.set_daily_total(daily_total, daily_total_conf)
                print(f"Daily total {dashboard_category}: {dt}")

                heading_above_applist = HOURS_AXIS_HEADING
                heading_below_applist = ''

            else:
                heading_above_applist = ''
                heading_below_applist = ''
                daily_total = NO_TEXT
                daily_total_conf = NO_CONF

            # Crop image to app region
            cropped_image, crop_top, crop_left, crop_bottom, crop_right = iOS.crop_image_to_app_area(current_screenshot, heading_above_applist, heading_below_applist)
            if all(crops is None for crops in (crop_top, crop_left, crop_bottom, crop_right)):
                print(f"Setting all app-specific data to N/A.")
                app_data = pd.DataFrame({
                    'app': [NO_TEXT] * max_apps_per_category,
                    'app_conf': [NO_CONF] * max_apps_per_category,
                    'number': [NO_TEXT if dashboard_category == SCREENTIME else NO_NUMBER] * max_apps_per_category,
                    'number_conf': [NO_CONF] * max_apps_per_category
                })
                #  here you also need to store the data in the screenshot object and participant object
            else:

                cropped_image = cv2.GaussianBlur(cropped_image, ksize=(3, 3), sigmaX=0)

                # Perform pre-scan to remove bars below app names
                cropped_prescan_words, cropped_prescan_df = extract_text_from_image(cropped_image)
                cropped_prescan_words = cropped_prescan_words.reset_index(drop=True)
                cropped_image_no_bars = iOS.erase_bars_below_app_names(screenshot=current_screenshot,
                                                                       df=cropped_prescan_words,
                                                                       image=cropped_image)
                scaled_cropped_image = cv2.resize(cropped_image,
                                                  dsize=None,
                                                  fx=app_area_scale_factor,
                                                  fy=app_area_scale_factor,
                                                  interpolation=cv2.INTER_AREA)

                # Extract app info from cropped image
                app_area_df = extract_app_info(current_screenshot, scaled_cropped_image, app_area_scale_factor)
                if show_images:
                    show_image(app_area_df, scaled_cropped_image)

                # Divide the extracted app info into app names and their numbers
                app_data = iOS.get_app_names_and_numbers(screenshot=current_screenshot,
                                                         df=app_area_df,
                                                         category=dashboard_category,
                                                         max_apps=max_apps_per_category)
                print("\nApp data found:")
                print(app_data[['name', 'number']])
                print(f"Daily total {dashboard_category}: {daily_total}")
        else:
            print("Operating System not detected.")
            app_data = None
            # For both android and iOS screenshots, we can now store the app-level data in the Screenshot object.
        current_screenshot.set_app_data(app_data)
        current_participant.add_screenshot(current_screenshot)
        # And also give it to the Participant object, checking to see if data already exists for that day & category
        #   (if it does, run the function (within Participant?) to determine how to merge the two sets of data together)
