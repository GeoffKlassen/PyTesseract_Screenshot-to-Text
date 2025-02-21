"""This file contains Android-specific dictionaries, functions, and variables."""
import cv2
import numpy as np
import re

import pandas as pd

import OCRScript_v3
from RuntimeValues import *

"""
    Android-Specific dictionaries

    Some keyword dictionaries are Android-specific. For example, Android has a dictionary for HOURS, but iOS doesn't need
    one because iOS uses 'h' for hours in all languages. Conversely, iOS has a LIMITATIONS heading while Android doesn't.
"""

DATE_RANGE_FORMAT = {ITA: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-d{1,2}\s?MMM'],
                     ENG: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'MMM\s?\d{1,2}-\d{1,2}',
                           r'MMM*\s?\d{1,2}-MMM*\s?\d{1,2}'],
                     GER: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-\d{1,2}\s?MMM'],
                     FRA: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-\d{1,2}\s?MMM']}

# hhh/HHH stands for the short/long format for hours; mmm/MMM stands for the short/long format for minutes.
# The list of abbreviations for the necessary language will be subbed in as needed to create the full regex.
TIME_FORMATS = [r'^[01ilLT]?[0-9aAilLStT]\s?hhh\s?[0-5aAilLT]?[0-9aAilLStT]\s?mmm$',  # Format for ## hr ## min
                r'^[01ilLT]?[0-9aAilLStT]\s?HHH$',                                    # Format for ## hours
                r'^[0-5aAilT]?[0-9AlLOStT]\s?MMM$']                                   # Format for ## minutes
# Sometimes pytesseract mistakes digits for A, I, L, S, or T (e.g.  A = 4,   I/L/T = 1,   S = 5)
# Including these letters in the regex ensures that times with a misread digit still match a time format.

KEYWORDS_FOR_HOURS = {ITA: ['ore', 'ora'],
                      ENG: ['hours', 'hour'],
                      GER: ['Stunden', 'Stunde'],
                      FRA: ['heures', 'heure']}
KEYWORDS_FOR_MINUTES = {ITA: ['minuti', 'minuto'],
                        ENG: ['minutes', 'minute'],
                        GER: ['Minuten', 'Minute'],
                        FRA: ['minutes', 'minute']}
# Long format for time words

KEYWORD_FOR_HR = {ITA: 'h e',
                  ENG: 'hr',
                  GER: 'Std',
                  FRA: 'h et'}
KEYWORD_FOR_MIN = {ITA: 'min',
                   ENG: 'min',
                   GER: 'Min',
                   FRA: 'min'}
# Short format for time words

H = 'h'
MIN = '(mi?n?)'

KEYWORDS_FOR_2018_SCREENTIME = {ITA: ['DURATA SCHERMO', 'DURATA SCHERMO Altro'],
                                ENG: ['TODO FILL THIS IN'],
                                GER: ['TODO FILL THIS IN'],
                                FRA: ['TODO FILL THIS IN']}
KEYWORDS_FOR_2018_MOST_USED = {ITA: ['UTILIZZO APP'],
                               ENG: ['TODO FILL THIS IN'],
                               GER: ['TODO FILL THIS IN'],
                               FRA: ['TODO FILL THIS IN']}
KEYWORDS_FOR_2018_UNLOCKS = {ITA: ['SBLOCCHI'],
                             ENG: ['TODO FILL THIS IN'],
                             GER: ['TODO FILL THIS IN'],
                             FRA: ['TODO FILL THIS IN']}

KEYWORDS_FOR_SCREEN_TIME = {ITA: ['Tempo di utilizzo', 'Tempo di utilizzo dello schermo',
                                    'DURATA SCHERMO', 'DURATA SCHERMO Altro'],
                            ENG: ['Screen time'],
                            GER: ['TODO: FILL THIS IN'],  # TODO Fill this in
                            FRA: ['Temps dutilisation des écrans', 'Temps decran']}
# Actual phrases are "Temps d'utilisation des ecrans" and "Temps d'ecran"
KEYWORDS_FOR_MOST_USED_APPS = {ITA: ['Applicazioni piu utilizzate', 'UTILIZZO APP', 'Applicazioni utilizzate'],
                               ENG: ['Most used apps'],
                               GER: ['Geratenutzungsdauer'],
                               FRA: ['Applications les plus', 'Applications les plus utilisees']}
KEYWORDS_FOR_NOTIFICATIONS_RECEIVED = {ITA: ['Notifiche ricevute'],
                                       ENG: ['Notifications received'],
                                       GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                                       FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
KEYWORDS_FOR_MOST_NOTIFICATIONS = {ITA: ['Piu notifiche'],
                                   ENG: ['Most notifications'],
                                   GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                                   FRA: ['Notifications les plus nombreuses', 'Notifications les', 'plus nombreuses']}
KEYWORDS_FOR_TIMES_OPENED = {ITA: ['Numero di aperture', 'Sblocchi', 'SBLOCCHI'],
                             ENG: ['Times opened', 'Unlocks'],
                             GER: ['Wie oft geoffnet'],  # TODO Fill this in
                             FRA: ['Nombre douvertures']}  # Actual phrase is Nombre d'ouvertures
KEYWORDS_FOR_VIEW_MORE = {ITA: ['Visualizza altro'],
                          ENG: ['View more', 'View all'],
                          GER: ['TODO FILL THIS IN'],
                          FRA: ['Afficher plus']}

GOOGLE_SCREENTIME_FORMATS = {ITA: ['# ora', '# h e # min', '# minuti', '1 minuto', 'Meno di 1 minuto'],
                             ENG: ['# hours', '# hr # min', '# minutes', '1 minute', 'Less than 1 minute'],
                             GER: ['# Stunde', '# Std # Min', '# Minuten', '1 Minute', 'Weniger als 1 Minute'],
                             FRA: ['# heures', '# h et # min', '# minutes', '1 minute', 'Moins de 1 minute']}
GOOGLE_NOTIFICATIONS_FORMATS = {ITA: ['# notifiche'],
                                ENG: ['# notifications', '# notification'],
                                GER: ['# Benachrichtigungen'],
                                FRA: ['# notifications', '# notification']}
GOOGLE_UNLOCKS_FORMATS = {ITA: ['# sblocchi', '# aperture'],
                          ENG: ['# unlocks', 'Opened # times'],
                          GER: ['# Entsperrungen', '# Mal geoffnet'],
                          FRA: ['Déverrouillé # fois', 'Ouverte # fois']}
SAMSUNG_NOTIFICATIONS_FORMATS = {ITA: ['# notifiche ricevute', "# ricevute"],
                                 ENG: ['# notifications', '# notification', '# received'],
                                 # TODO Should this include '# notifications received'?
                                 GER: ['# Benachrichtigungen'],  # TODO Make sure this is correct
                                 FRA: ['# notifications', '# notification']}  # TODO Fill this in
SAMSUNG_UNLOCKS_FORMAT = {ITA: ['# volte', '# in totale'],
                          ENG: ['# times'],
                          GER: [''],  # TODO Fill this in
                          FRA: ['']}  # TODO Fill this in

# YOU_CAN_SET_DAILY_TIMERS = {ITA: 'Imposta i timer per le app',
#                             ENG: 'You can set daily timers',
#                             GER: 'Timer fur Apps einrichten',
#                             FRA: ''}  # TODO Fill this in
# "You can set daily timers" is a tip box that appears in the Google version of Dashboard, until a user clears it.

REST_OF_THE_DAY = {ITA: 'giornata',  # full phrase is 'resto della giornata' but 'giornata' is sometimes its own line
                   ENG: 'rest of the day',
                   GER: 'Rest des Tages pausiert',
                   FRA: ''}
# "rest of the day" is the last text in the dialogue box for "You can set daily timers".

SHOW_SITES_YOU_VISIT = {ITA: 'Mostra i siti visitati',
                        ENG: 'Show sites that you visit',  # TODO: used to be 'Show sites you visit' for HappyB - check if this is the correct phrase
                        GER: 'Besuchte Websites anzeigen',
                        FRA: ''}  # TODO Fill this in
# "Show sites you visit" can appear in the Google version of Dashboard, under the Chrome app (if it's in the app list).
# Thus, it can be mistaken for an app name, so we need to ignore it.

KEYWORDS_FOR_UNRELATED_SCREENSHOTS = {ITA: ['USO BATTERIA', 'Benessere digitale'],
                                      ENG: ['BATTERY USE', 'BATTERY USAGE', 'Digital wellbeing'],
                                      GER: ['TODO FILL THIS IN'],
                                      FRA: ['TODO FILL THIS IN']}
# Some screenshots show only Battery Usage info; these screenshots do not contain any of the requested info.


def screenshot_contains_unrelated_data(ss):
    text_df = ss.text
    lang = ss.language
    moe = int(np.log(min(len(k) for k in KEYWORDS_FOR_UNRELATED_SCREENSHOTS[lang]))) + 1
    # margin of error for text (number of characters two strings can differ by and still be considered the same text)
    
    if any(text_df['text'].apply(lambda x: min(OCRScript_v3.levenshtein_distance(x[0:len(key)], key)
                                               for key in KEYWORDS_FOR_UNRELATED_SCREENSHOTS[lang])) < moe):
        print("Did not detect data relevant to the study.")
        # One of the rows of text_df starts with one of the keywords for unrelated screenshots
        return True

    return False


def get_time_formats_in_lang(lang):
    short_format = re.sub('HHH|hhh', H, re.sub('MMM|mmm', MIN, '|'.join(TIME_FORMATS)))

    long_format = re.sub('hhh', KEYWORD_FOR_HR[lang], '|'.join(TIME_FORMATS))
    long_format = re.sub('mmm', KEYWORD_FOR_MIN[lang], long_format)
    long_format = re.sub('HHH', "(" + '|'.join(KEYWORDS_FOR_HOURS[lang]) + ")", long_format)
    long_format = re.sub('MMM', "(" + '|'.join(KEYWORDS_FOR_MINUTES[lang]) + ")", long_format)
    long_format = long_format.replace(" ", r"\s?")
    long_format = "(" + long_format + ")"

    return short_format, long_format


def get_headings(screenshot, time_fmt_short):
    """
    Finds the rows of text within the given screenshot that contain key headings ("SCREENTIME", "MOST USED", etc.)
    :param screenshot: The screenshot to search for headings
    :param time_fmt_short: A regex of the abbreviated time format for the relevant language (e.g. 1h 23m)
    :return: A df that contains only the rows with headings from the text within the screenshot (if none, an empty df)
    """
    df = screenshot.text
    lang = screenshot.language
    df[HEADING_COLUMN] = None
    dates_df = screenshot.rows_with_date

    if lang is None:
        print("Language not detected; cannot get headings from screenshot.")
        return df

    # Compile a list of all the 'day' keywords for the current language ('Yesterday', 'Today', etc.)
    day_types = [day for _dict in [KEYWORDS_FOR_TODAY, KEYWORDS_FOR_YESTERDAY, KEYWORDS_FOR_DAY_BEFORE_YESTERDAY]
                 for day in _dict.get(lang, [])]

    for i in df.index:
        row_text = df['text'][i]

        # Replace numbers with '#' symbol for matching with total screentime/notifications/unlocks formats
        row_text_filtered = re.sub(r'\d+', '#', row_text).replace(' ', '')

        row_text_contains_digits = bool(re.search(r'\d+', row_text))
        centre_of_row = int(df['left'][i] + 0.5 * df['width'][i])
        moe = round(np.log(len(str(row_text))))
        # margin of error (number of characters two strings can differ by and still be considered the same text)

        if min(OCRScript_v3.levenshtein_distance(row_text, day) for day in day_types) <= moe:
            # Row contains a day name
            df.loc[i, HEADING_COLUMN] = DAY_NAME_HEADING
        elif re.search(screenshot.date_format, row_text, re.IGNORECASE):
            # Row contains date text
            df.loc[i, HEADING_COLUMN] = DATE_HEADING
        # elif dates_df is not None and i in dates_df.index:  # If this isn't needed, you can get rid of dates_df
        #     df.loc[i, HEADING_COLUMN] = DATE_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key)
                 for key in KEYWORDS_FOR_SCREEN_TIME[lang]) <= moe:
            # Row contains 'Screen time'
            df.loc[i, HEADING_COLUMN] = SCREENTIME_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key)
            # Row contains 'Most used'
                 for key in KEYWORDS_FOR_MOST_USED_APPS[lang]) <= moe:
            df.loc[i, HEADING_COLUMN] = MOST_USED_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key) for key in
                 KEYWORDS_FOR_NOTIFICATIONS_RECEIVED[lang]) <= moe and not row_text_contains_digits:
            # Row contains 'Notifications'
            df.loc[i, HEADING_COLUMN] = NOTIFICATIONS_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key) for key in
                 KEYWORDS_FOR_MOST_NOTIFICATIONS[lang]) <= moe and not row_text_contains_digits:
            # Row contains 'Most Notifications'
            df.loc[i, HEADING_COLUMN] = MOST_NOTIFICATIONS_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key) for key in
                 KEYWORDS_FOR_TIMES_OPENED[lang]) <= moe and not row_text_contains_digits:
            # Row contains 'Times Opened'
            df.loc[i, HEADING_COLUMN] = UNLOCKS_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key) for key in KEYWORDS_FOR_VIEW_MORE[lang]) <= moe:
            # Row contains 'View More'
            df.loc[i, HEADING_COLUMN] = VIEW_MORE_HEADING

        elif min(OCRScript_v3.levenshtein_distance(row_text, key) for key in KEYWORDS_FOR_2018_SCREENTIME[lang]) <= moe:
            # Row contains
            df.loc[i, HEADING_COLUMN] = OLD_SCREENTIME_HEADING
        elif OCRScript_v3.levenshtein_distance(row_text, KEYWORDS_FOR_2018_MOST_USED[lang]) <= moe:
            df.loc[i, HEADING_COLUMN] = OLD_MOST_USED_HEADING
        elif OCRScript_v3.levenshtein_distance(row_text, KEYWORDS_FOR_2018_UNLOCKS[lang]) <= moe:
            df.loc[i, HEADING_COLUMN] = OLD_UNLOCKS_HEADING

        elif bool(re.match(time_fmt_short, row_text)) and df['left'][i] < 0.15 * screenshot.width or \
                min(OCRScript_v3.levenshtein_distance(row_text_filtered, key.replace(' ', ''))
                    for key in GOOGLE_SCREENTIME_FORMATS[lang]) <= moe and \
                abs(centre_of_row - (0.5 * screenshot.width)) < (0.1 * screenshot.width) and \
                not df[HEADING_COLUMN].str.contains(TOTAL_SCREENTIME).any():
            # Row text starts with a short-format time length (e.g. 1h5m) and is left-aligned (Samsung style), or
            # Row text matches a long-format time length (e.g. 1 hr 5 min) and is centred (Google style)
            df.loc[i, HEADING_COLUMN] = TOTAL_SCREENTIME
        elif (min(OCRScript_v3.levenshtein_distance(row_text_filtered, key.replace(' ', '')) for key in
                  (GOOGLE_NOTIFICATIONS_FORMATS[lang] + SAMSUNG_NOTIFICATIONS_FORMATS[lang])) <= moe and
              (df['left'][i] < 0.15 * screenshot.width or
               abs(centre_of_row - (0.5 * screenshot.width)) < (0.1 * screenshot.width))) and \
                not df[HEADING_COLUMN].str.contains(TOTAL_NOTIFICATIONS).any():
            # Row text matches a 'total notifications' format and is either left-aligned (Samsung) or centred (Google)
            df.loc[i, HEADING_COLUMN] = TOTAL_NOTIFICATIONS
        elif ((min(OCRScript_v3.levenshtein_distance(row_text_filtered, key.replace(' ', '')) for key in
                   (GOOGLE_UNLOCKS_FORMATS[lang] + SAMSUNG_UNLOCKS_FORMAT[lang]))) <= moe and
              (df['left'][i] < 0.15 * screenshot.width or abs(centre_of_row - (0.5 * screenshot.width)) < (
                      0.1 * screenshot.width))) and \
                not df[HEADING_COLUMN].str.contains(TOTAL_UNLOCKS).any():
            # Row text matches a 'total unlocks' format and is either left-aligned (Samsung) or centred (Google)
            df.loc[i, HEADING_COLUMN] = TOTAL_UNLOCKS
        else:
            df = df.drop(i)
    print("\nHeadings found:")
    print(df[['text', 'heading']])
    print()

    return df


def get_android_version(screenshot):
    """
    As of 2024, there are 4 distinct versions of the Android Dashboard, depending on phone brand and OS version.

    In version 1 (Samsung version), all headings appear on one scrollable page, in this order:
        Screen time,   Most used apps,   Notifications received,   Most notifications,   Unlocks
    In version 2 (NEW Samsung version), all 3 categories are available, but there are no category headings.

    In version 3 (Google version), each category is its own page, with the heading selected from a drop-down menu. The headings to choose from are:
        Screen time,   Notifications received,   Times opened
    In version 4 (2018 version), the user can choose to view an activity summary of either 'Today' or 'Last 7 days'.
    Thus, the data cannot be for one complete 24-hour day, and are thus not usable for the HappyB study.
    However, the data from these screenshots will still be extracted and saved. The headings are:
        SCREEN TIME,   APPS USED,   UNLOCKS   (no 'Notifications' data)
    :param screenshot: The screenshot to determine the android version for (contains headings_df)
    :return: The version of Android
    """
    img_lang = screenshot.language
    heads_df = screenshot.headings_df
    samsung_2021_headings = [SCREENTIME_HEADING, MOST_USED_HEADING,
                             NOTIFICATIONS_HEADING, MOST_NOTIFICATIONS_HEADING,
                             UNLOCKS_HEADING]
    _2018_headings_in_img_lang = (KEYWORDS_FOR_2018_SCREENTIME[img_lang] +
                                  KEYWORDS_FOR_2018_MOST_USED[img_lang] +
                                  KEYWORDS_FOR_2018_UNLOCKS[img_lang])
    error_margin = 2  # For finding headings from the 2018 Dashboard version

    if heads_df.empty:
        return None, None
    elif not heads_df[heads_df['text'].str.isupper()].empty:
        android_ver = VERSION_2018  # TODO: not sure on the year
    elif max(abs(heads_df['left'] + 0.5 * heads_df['width'] - (0.5 * screenshot.width))) < (0.1 * screenshot.width):
        # All the headings found are centred
        android_ver = GOOGLE
    elif not (heads_df['heading'].isin(samsung_2021_headings)).any():
        # None of the Samsung 2021 headings are found
        android_ver = SAMSUNG_2024
    elif (heads_df['heading'][1:].isin(samsung_2021_headings)).any() or \
        (not heads_df[~heads_df[HEADING_COLUMN].str.contains('total')].empty and
         min(heads_df['left'][~heads_df[HEADING_COLUMN].str.contains('total')]) < 0.15 * screenshot.width):
        # There's at least one non-'total' heading, and at least one of them is left-aligned
        android_ver = SAMSUNG_2021
    else:
        android_ver = None

    if android_ver is not None:
        print(f"Android version detected: {android_ver}")
    else:
        print("Android version not detected.")

    return android_ver


def get_dashboard_category(screenshot):
    heads_df = screenshot.headings_df
    text_df = screenshot.text
    category_submitted = screenshot.category_submitted
    categories_found = []
    
    if any(heads_df[HEADING_COLUMN].eq(SCREENTIME_HEADING)) or \
            any(heads_df[HEADING_COLUMN].eq(TOTAL_SCREENTIME)) or \
            any(heads_df[HEADING_COLUMN].eq(MOST_USED_HEADING)) and \
            (text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].index[0] + 1 or
             heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].iloc[-1]['top'] < 0.9 * screenshot.height):
        # Found screentime heading; or
        # Found total screentime; or
        # Found 'most used' heading and either:
        #     text_df has more data below the 'most used' heading, or
        #     the 'most used' heading is not too close to the bottom of the screenshot
        categories_found.append(SCREENTIME)

    if any(heads_df[HEADING_COLUMN].eq(NOTIFICATIONS_HEADING)) and \
            (text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == NOTIFICATIONS_HEADING].index[0] + 1 or
             heads_df[heads_df[HEADING_COLUMN] == NOTIFICATIONS_HEADING].iloc[-1]['top'] < 0.9 * screenshot.height) or \
            any(heads_df[HEADING_COLUMN].eq(TOTAL_NOTIFICATIONS)) or \
            any(heads_df[HEADING_COLUMN].eq(MOST_NOTIFICATIONS_HEADING)) and \
            (text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_NOTIFICATIONS_HEADING].index[0] + 1 or
             heads_df[heads_df[HEADING_COLUMN] == MOST_NOTIFICATIONS_HEADING].iloc[-1][
                 'top'] < 0.9 * screenshot.height):
        # Found notifications heading, and either;
        #     text_df has more data below the notifications heading, or
        #     the notifications heading is not too close to the bottom of the screenshot; or
        # Found total notifications; or
        # Found 'most notifications' heading and either:
        #     text_df has more data below the 'most notifications' heading, or
        #     the 'most notifications' heading is not too close to the bottom of the screenshot
        categories_found.append(NOTIFICATIONS)

    if any(heads_df[HEADING_COLUMN].eq(UNLOCKS_HEADING)) and \
            (text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == UNLOCKS_HEADING].index[0] + 1 or
             heads_df[heads_df[HEADING_COLUMN] == UNLOCKS_HEADING].iloc[-1]['top'] < 0.9 * screenshot.height) or \
            heads_df[HEADING_COLUMN].str.contains(TOTAL_UNLOCKS).any():
        # Found unlocks heading, and either:
        #     text_df has more data below the unlocks heading, or
        #     the unlocks heading is not too close to the bottom of the screenshot; or
        # Found total unlocks
        categories_found.append(UNLOCKS)

    if not categories_found:
        print(f"No categories detected.")  # Defaulting to submitted category: {backup_category}")
        # Returning None to flag the screenshot for manual review;
        # screenshot category is set to backup_category after this function is called
        return None
    else:
        print(f"Category submitted: {category_submitted}    Categories detected: {categories_found}")
        if category_submitted in categories_found:
            category_found = category_submitted
        else:
            category_found = categories_found[0]
            print(f"Screenshot submitted under '{category_submitted}' category, but found '{category_found}' instead.")

    print(f"Setting category to '{category_found}'.")

    return category_found


def filter_time_text(text, conf, hr_f, min_f):

    if str(text) == NO_TEXT:
        return NO_TEXT, NO_CONF

    def replace_misread_digit(misread, actual, s):
        # Replaces a 'misread' digit with the 'actual' digit, but only if it is followed by a time word/character
        # (hours or minutes) in the relevant language
        pattern = re.compile(''.join([misread, r"(?=[0-9tails]?\s?(", hr_f, "|", min_f, "))"]), flags=re.IGNORECASE)
        filtered_str = re.sub(pattern, actual, s)

        return filtered_str

    # Replace common misread characters (e.g. pytesseract sometimes misreads '1h' as 'Th'/'th').
    text = replace_misread_digit('(t|T)', '1', text)
    text = replace_misread_digit('(a|A)', '4', text)
    text = replace_misread_digit('(i|I)', '1', text)
    text = replace_misread_digit('(l|L)', '1', text)
    text = replace_misread_digit('(s|S)', '5', text)
    text = replace_misread_digit('(o|O)', '0', text)

    return text, conf


def get_daily_total_and_confidence(screenshot, image, heading):
    df = screenshot.text
    headings_df = screenshot.headings_df
    android_version = screenshot.android_version
    img_lang = screenshot.language
    total_heading = "total " + heading if heading is not None else "total " + screenshot.category_submitted
    day_rows = screenshot.rows_with_day_type
    date_rows = screenshot.rows_with_date

    def rescan_cropped_area(img, c_top, c_bottom, c_left, c_right):
        cropped_image = img[c_top:c_bottom, c_left:c_right]
        scale = 0.5  # Pytesseract sometimes fails to read very large text; scale down the cropped region
        scaled_cropped_image = cv2.resize(cropped_image, None,
                                          fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, rescan_df = OCRScript_v3.extract_text_from_image(scaled_cropped_image)

        if show_images:
            OCRScript_v3.show_image(rescan_df, scaled_cropped_image)

        return rescan_df

    if any(headings_df[HEADING_COLUMN] == total_heading):
        total_value_row = headings_df[headings_df[HEADING_COLUMN] == total_heading].iloc[0]
        total_value_1st_scan = total_value_row['text']
        total_value_1st_scan_conf = round(total_value_row['conf'], 4)
        print(f"Initial scan: {total_value_1st_scan} (conf = {total_value_1st_scan_conf})")

        crop_top = max(0, total_value_row['top'] - int(0.05 * screenshot.width))
        crop_bottom = min(total_value_row['top'] + total_value_row['height'] + int(0.025 * screenshot.width), screenshot.height)
        crop_left = max(0, total_value_row['left'] - int(0.05 * screenshot.width))
        crop_right = min(total_value_row['left'] + total_value_row['width'] + int(0.05 * screenshot.width), screenshot.width)

        cropped_scan = rescan_cropped_area(image, crop_top, crop_bottom, crop_left, crop_right)
    elif android_version == GOOGLE and not (day_rows.empty or date_rows.empty):
        total_value_1st_scan = NO_TEXT
        total_value_1st_scan_conf = NO_CONF

        rows_below_total = pd.concat([day_rows, date_rows], ignore_index=True)
        rows_below_total = rows_below_total.sort_index()
        row_below_total = rows_below_total.iloc[0]
        crop_top = row_below_total['top'] - (4 * row_below_total['height'])
        crop_bottom = row_below_total['top']
        crop_left = 0
        crop_right = screenshot.width

        cropped_scan = rescan_cropped_area(image, crop_top, crop_bottom, crop_left, crop_right)

    elif android_version == SAMSUNG_2021 and not headings_df[headings_df[HEADING_COLUMN] == heading].empty:
        total_value_1st_scan = NO_TEXT
        total_value_1st_scan_conf = NO_CONF

        row_above_total = headings_df[headings_df[HEADING_COLUMN] == heading].iloc[0]
        crop_top = row_above_total['top'] + (3 * row_above_total['height'])
        crop_bottom = crop_top + (6 * row_above_total['height'])
        crop_left = 0
        crop_right = screenshot.width

        cropped_scan = rescan_cropped_area(image, crop_top, crop_bottom, crop_left, crop_right)

    elif android_version == VERSION_2018 and not headings_df[headings_df[HEADING_COLUMN] == heading].empty:
        total_value_1st_scan = NO_TEXT
        total_value_1st_scan_conf = NO_CONF

        row_above_total = headings_df[headings_df[HEADING_COLUMN] == heading].iloc[0]
        crop_top = row_above_total['top'] + (2 * row_above_total['height'])
        crop_bottom = crop_top + (4 * row_above_total['height'])
        crop_left = 0
        crop_right = int(0.5 * screenshot.width)

        cropped_scan = rescan_cropped_area(image, crop_top, crop_bottom, crop_left, crop_right)
    else:
        # Since Samsung 2024 version has no headings, it is not possible to select a suitable crop area
        # to rescan for totals when the desired 'total' row is not found
        total_value_1st_scan = NO_TEXT
        total_value_1st_scan_conf = NO_CONF
        cropped_scan = df.drop(df.index)

    if not cropped_scan.empty:
        total_value_2nd_scan = cropped_scan.iloc[0]['text']
        total_value_2nd_scan_conf = round(cropped_scan.iloc[0]['conf'], 4)
        print(f"Cropped scan: {total_value_2nd_scan} (conf = {total_value_2nd_scan_conf})")
    else:
        print(f"Total {heading} not found on 2nd scan.")
        total_value_2nd_scan = NO_TEXT
        total_value_2nd_scan_conf = NO_CONF

    total_value, total_conf = OCRScript_v3.choose_between_two_values(total_value_1st_scan, total_value_1st_scan_conf,
                                                                     total_value_2nd_scan, total_value_2nd_scan_conf)
    total_value = str(total_value)

    if screenshot.category_detected != SCREENTIME:
        try:
            total_value_filtered = re.findall(r'-?\d+', total_value)[0]  # TODO why the - symbol?
        except IndexError:
            total_value_filtered = str.split(total_value)[0]
    else:
        if android_version == GOOGLE:
            hours_format = '|'.join([('|'.join(KEYWORDS_FOR_HOURS[img_lang])), KEYWORD_FOR_HR[img_lang]]).replace(" ",
                                                                                                                  r"\s?")
            minutes_format = '|'.join([('|'.join(KEYWORDS_FOR_MINUTES[img_lang])), KEYWORD_FOR_MIN[img_lang]])
        else:
            hours_format = H
            minutes_format = MIN
        total_value_filtered, total_conf = filter_time_text(total_value, total_conf,
                                                            hours_format, minutes_format)

    return total_value_filtered, total_conf


def convert_string_time_to_minutes(screenshot):
    """

    :param screenshot:
    :return:
    """
    lang = OCRScript_v3.get_best_language(screenshot)
    s = screenshot.daily_total
    android_version = screenshot.android_version

    if str(s) == NO_TEXT:
        return NO_TEXT

    def split_time(_s, _format):
        split = re.split(_format, _s)
        if split[0] != s:
            # Time did start with 'format' (either '# hours' or '# minutes'); extract the number before the format
            time_str = split[0].replace(" ", "")
            time_str = re.findall(r'\d+', time_str)
            time_int = int(time_str[0]) if len(time_str) > 0 else 0
            leftover_str = "".join(split[1:])
        else:
            time_int = 0
            leftover_str = _s

        return time_int, leftover_str


    if android_version == GOOGLE:
        hours_format = '|'.join([('|'.join(KEYWORDS_FOR_HOURS[lang])), KEYWORD_FOR_HR[lang]]).replace(" ", r"\s?")
        minutes_format = '|'.join([('|'.join(KEYWORDS_FOR_MINUTES[lang])), KEYWORD_FOR_MIN[lang]])
    else:
        hours_format = H
        minutes_format = MIN

    hours, str_after_hours = split_time(s, hours_format)
    minutes, _ = split_time(str_after_hours, minutes_format)

    time_in_min = (hours * 60) + minutes

    return time_in_min


def crop_image_to_app_area(image, heading_above_apps, screenshot, time_format_short):
    android_version = screenshot.android_version
    date_rows = screenshot.rows_with_date
    text_df = screenshot.text
    headings_df = screenshot.headings_df
    dashboard_category = screenshot.category_detected

    if android_version == GOOGLE:
        crop_left = int(0.15 * screenshot.width)  # Crop out the app icon area (first 15% of screenshot_width)
        crop_right = int(0.79 * screenshot.width)  # Crop out the hourglass area (last 79% of screenshot_width)
        if not date_rows.empty:
            crop_top = date_rows.iloc[-1]['top'] + (2 * date_rows.iloc[-1]['height'])
            crop_bottom = screenshot.height
        else:
            # TODO Leaving this as a catch-all for now -- debug later if this condition is used
            crop_top = 0
            crop_bottom = screenshot.height

        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]

    elif android_version in [SAMSUNG_2024, SAMSUNG_2021, VERSION_2018]:
        crop_left = int(0.1 * screenshot.width) if android_version == SAMSUNG_2024 else int(
            0.17 * screenshot.width)  # Crop out the app icon area (first 17% of screenshot_width)
        # Crop out the > arrows to the right of each app (these appear in the Samsung 2018 version of Dashboard)
        crop_right = int(0.9 * screenshot.width) if android_version == VERSION_2018 else screenshot.width

        if any(headings_df[HEADING_COLUMN].eq(heading_above_apps)):
            row_above_apps = headings_df[headings_df[HEADING_COLUMN] == heading_above_apps].iloc[0]
            crop_top = row_above_apps['top'] + row_above_apps['height']
            headings_below_apps_df = headings_df[headings_df.index > row_above_apps.name]
            if not headings_below_apps_df.empty:
                row_below_apps = headings_below_apps_df.iloc[0]
                crop_bottom = row_below_apps['top']
            else:
                crop_bottom = screenshot.height
        else:
            print("Sub-heading above app rows not found. Searching for app rows directly.")
            crop_top = 0
            format_for_time_or_number_eol = '|'.join([time_format_short.replace("^", "\\s"), "\\s\\d+"])

            headings_above_apps_df = headings_df[headings_df[HEADING_COLUMN].str.contains(dashboard_category)]
            index_of_closest_heading = headings_above_apps_df.index[-1] if not headings_above_apps_df.empty else 0
            index_of_first_app = -1

            for i in range(index_of_closest_heading, text_df.shape[0]):
                if text_df['left'][i] < 0.25 * screenshot.width and \
                        text_df['left'][i] + text_df['width'][i] > 0.85 * screenshot.width and \
                        re.search(format_for_time_or_number_eol, text_df['text'][i]):
                    # Row text spans most of the width of the screenshot, and also ends with a time/number
                    print(f"App row found: {text_df['text'][i]}.  "
                          f"Setting top of crop region to top of this app row.")
                    crop_top = int(text_df['top'][i] - 0.01 * screenshot.height)
                    index_of_first_app = i
                    break

            headings_below_apps_df = headings_df[headings_df.index > max(index_of_first_app, index_of_closest_heading)]
            if not headings_below_apps_df.empty:
                row_below_apps = headings_below_apps_df.iloc[0]
                crop_bottom = row_below_apps['top']
            else:
                crop_bottom = screenshot.height

        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
    else:
        print(f"Android version not detected; image will not be cropped.")
        return image

    return cropped_image


def main():
    print("I am now in AndroidFunctions.py")
