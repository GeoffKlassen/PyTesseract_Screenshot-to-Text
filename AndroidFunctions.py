"""This file contains Android-specific dictionaries, functions, and variables."""
from collections import namedtuple

import cv2
import numpy as np
import re

import pandas as pd

import OCRScript_v3
from OCRScript_v3 import get_best_language, levenshtein_distance
from RuntimeValues import *
from iOSFunctions import filter_time_or_number_text

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
TIME_FORMATS = [r'^[0-9ilLT]?[0-9aAilLStT]\s?hhh\s?[0-9aAilLT]?[0-9aAilLoStT]\s?mmm$',  # Format for ## hr ## min
                r'^[0-9ilLT]?[0-9aAilLStT]\s?HHH\s?[0-9aAilLT]?[0-9aAilLoStT]\s?MMM$',  # Format for ## hours ## minutes
                r'^[01ilLT]?[0-9aAilLStT]\s?HHH$',                                     # Format for ## hours
                r'^[0-9aAilT]?[0-9AlLOStT]\s?MMM$']                                    # Format for ## minutes
# Sometimes pytesseract mistakes digits for letters (e.g.  A = 4,   I/L/T = 1,   S = 5)
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

KEYWORDS_FOR_HR = {ITA: ['h e'],
                   ENG: ['hrs', 'hr', 'h'],
                   GER: ['Std'],
                   FRA: ['h et']}
KEYWORDS_FOR_MIN = {ITA: ['min'],
                    ENG: ['mins', 'min', 'm'],
                    GER: ['Min'],
                    FRA: ['min']}
# Short format for time words

H = 'hr?s?'
MIN = 'mi?n?s?'

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
                             ENG: ['# hours', '# hour',
                                   '# hours # minutes', '# hours # minute', '# hour # minutes', '# hour # minute',
                                   '# hrs # mins', '# hr # mins', '# hrs # min', '# hr # min', '#h #m',
                                   '# minutes', '1 minute', 'Less than 1 minute', '< 1 minute'],
                             GER: ['# Stunde', '# Std # Min', '# Minuten', '1 Minute', 'Weniger als 1 Minute'],
                             FRA: ['# heures', '# h et # min', '# minutes', '1 minute', 'Moins de 1 minute']}
GOOGLE_LESS_THAN_1_MINUTE = {ITA: ['Meno di 1 minuto'],
                             ENG: ['Less than 1 minute'],
                             GER: ['Weniger als 1 Minute'],
                             FRA: ['Moins de 1 minute']}
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
                                 FRA: ['# notifications', '# notification']}
SAMSUNG_UNLOCKS_FORMAT = {ITA: ['# volte', '# in totale'],
                          ENG: ['# times'],
                          GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                          FRA: ['TODO FILL THIS IN']}  # TODO Fill this in

SAMSUNG_SEARCH_APPS = {ITA: ['TODO FILL THIS IN'],  # TODO Fill this in
                       ENG: ['Search apps', 'Search for categories'],
                       GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                       FRA: ['TODO FILL THIS IN']}  # TODO Fill this in

# YOU_CAN_SET_DAILY_TIMERS = {ITA: 'Imposta i timer per le app',
#                             ENG: 'You can set daily timers',
#                             GER: 'Timer fur Apps einrichten',
#                             FRA: ''}  # TODO Fill this in
# "You can set daily timers" is a tip box that appears in the Google version of Dashboard, until a user clears it.

KEYWORDS_FOR_REST_OF_THE_DAY = {ITA: ['giornata'],  # full phrase is 'resto della giornata' but 'giornata' is sometimes its own line
                                ENG: ['rest of the day', 'rest of the', 'of the day', 'the day.'],
                                GER: ['Rest des Tages pausiert'],
                                FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
# "rest of the day" is the last text in the dialogue box for "You can set daily timers".

KEYWORDS_FOR_SHOW_SITES_YOU_VISIT = {ITA: ['Mostra i siti visitati'],
                                     ENG: ['Show sites you visit', 'Show sites that you visit',
                                           '( Show sites you visit )', '( Show sites that you visit )'],
                                     # Sometimes the oval around this text gets read as parentheses ()
                                     GER: ['Besuchte Websites anzeigen'],
                                     FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
# "Show sites you visit" can appear in the Google version of Dashboard, under the Chrome app (if it's in the app list).
# Thus, it can be mistaken for an app name, so we need to ignore it.

KEYWORDS_FOR_UNRELATED_SCREENSHOTS = {ITA: ['USO BATTERIA', 'Benessere digitale'],
                                      ENG: ['BATTERY USE', 'BATTERY USAGE',
                                            'Digital wellbeing', 'Digital Wellbeing & parental',
                                            'No limit', 'Manage notifications', 'Weekly report'],
                                      GER: ['TODO FILL THIS IN'],
                                      FRA: ['TODO FILL THIS IN']}
# Some screenshots show only Battery Usage info; these screenshots do not contain any of the requested info.


def screenshot_contains_unrelated_data(ss):
    """

    :param ss:
    :return:
    """
    text_df = ss.text
    lang = ss.language
    moe = int(np.log(min(len(k) for k in KEYWORDS_FOR_UNRELATED_SCREENSHOTS[lang]))) + 1
    # margin of error for text (number of characters two strings can differ by and still be considered the same text)
    
    if any(text_df['text'].apply(lambda x: min(OCRScript_v3.levenshtein_distance(w, key)
                                               for key in KEYWORDS_FOR_UNRELATED_SCREENSHOTS[lang]
                                               for w in [x[0:len(key)], x[-len(key):]])) < moe):
        print("Detected data irrelevant to the study.")
        # One of the rows of text_df starts with one of the keywords for unrelated screenshots
        return True

    return False


def get_time_formats_in_lang(lang):
    """

    :param lang:
    :return:
    """
    short_format = re.sub('HHH|hhh', H, re.sub('MMM|mmm', MIN, '|'.join(TIME_FORMATS)))

    long_format = re.sub('hhh', "(" + '|'.join(KEYWORDS_FOR_HR[lang]) + ")", '|'.join(TIME_FORMATS))
    long_format = re.sub('mmm', "(" + '|'.join(KEYWORDS_FOR_MIN[lang]) + ")", long_format)
    long_format = re.sub('HHH', "(" + '|'.join(KEYWORDS_FOR_HOURS[lang]) + ")", long_format)
    long_format = re.sub('MMM', "(" + '|'.join(KEYWORDS_FOR_MINUTES[lang]) + ")", long_format)
    long_format = long_format.replace(" ", r"\s?")
    long_format = "(" + long_format + ")"

    end_of_line_format = '|'.join([short_format, long_format]).replace("^", "\\s")  # eol = 'end of line'

    return short_format, long_format, end_of_line_format


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

    if lang is None:
        print("Language not detected; cannot get headings from screenshot.")
        return df

    # Compile a list of all the 'day' keywords for the current language ('Yesterday', 'Today', etc.)
    day_types = [day for _dict in [KEYWORDS_FOR_TODAY, KEYWORDS_FOR_YESTERDAY, KEYWORDS_FOR_DAY_BEFORE_YESTERDAY]
                 for day in _dict.get(lang, [])]

    for i in df.index:
        row_text = df['text'][i]
        row_words = set(df['text'][i].split())

        row_text = re.sub(r'(?<=\d)n', 'h', row_text) # Occasionally 'h' gets read as 'n'
        # Replace numbers with '#' symbol for matching with total screentime/notifications/unlocks formats
        row_text_filtered = re.sub(r'\d+', '#', row_text).replace(' ', '')
        # debug
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
        elif len(row_words.intersection(DAY_ABBREVIATIONS[lang])) >= 3:
            df.loc[i, HEADING_COLUMN] = DAYS_AXIS_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text, key)
                 for key in KEYWORDS_FOR_SCREEN_TIME[lang]) <= moe:
            # Row contains 'Screen time'
            df.loc[i, HEADING_COLUMN] = SCREENTIME_HEADING
        elif min(OCRScript_v3.levenshtein_distance(row_text[:len(key)], key)
            # Row contains 'Most used'
                 for key in KEYWORDS_FOR_MOST_USED_APPS[lang]) <= moe:
            df.loc[i, HEADING_COLUMN] = MOST_USED_APPS_HEADING
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

        elif (bool(re.match(time_fmt_short, row_text)) and df['left'][i] < 0.15 * screenshot.width or
                (not re.fullmatch(r'#+', row_text_filtered) and
                 min(OCRScript_v3.levenshtein_distance(row_text_filtered, key.replace(' ', ''))
                     for key in GOOGLE_SCREENTIME_FORMATS[lang]) <= moe and
                 abs(centre_of_row - (0.5 * screenshot.width)) < (0.1 * screenshot.width))) and \
                 not df[HEADING_COLUMN].str.contains(TOTAL_SCREENTIME).any():
            # Row text starts with a short-format time length (e.g. 1h5m) and is left-aligned (Samsung style), or
            # Row text matches a long-format time length (e.g. 1 hr 5 min) and is centred (Google style)
            df.loc[i, HEADING_COLUMN] = TOTAL_SCREENTIME
        elif (min(OCRScript_v3.levenshtein_distance(row_text_filtered, key.replace(' ', '')) for key in
                  (GOOGLE_NOTIFICATIONS_FORMATS[lang] + SAMSUNG_NOTIFICATIONS_FORMATS[lang])) <= moe and
              (df['left'][i] < 0.15 * screenshot.width or
               abs(centre_of_row - (0.5 * screenshot.width)) < (0.1 * screenshot.width))) and \
                not row_text_filtered.startswith(('N', 'n')) and \
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

        elif (min(OCRScript_v3.levenshtein_distance(row_text[-len(key):], key) for key in KEYWORDS_FOR_REST_OF_THE_DAY[lang])) < moe:
            df.loc[i, HEADING_COLUMN] = REST_OF_THE_DAY

        elif (min(OCRScript_v3.levenshtein_distance(row_text[-len(key):], key) for key in SAMSUNG_SEARCH_APPS[lang])) < moe:
            df.loc[i, HEADING_COLUMN] = SEARCH_APPS
        else:
            df = df.drop(i)

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
    heads_df = screenshot.headings_df[~(screenshot.headings_df[HEADING_COLUMN] == REST_OF_THE_DAY) &
                                      ~(screenshot.headings_df[HEADING_COLUMN] == DAYS_AXIS_HEADING)]
    samsung_2021_headings = [SCREENTIME_HEADING, MOST_USED_HEADING,
                             NOTIFICATIONS_HEADING, MOST_NOTIFICATIONS_HEADING,
                             UNLOCKS_HEADING]
    _2018_headings_in_img_lang = (KEYWORDS_FOR_2018_SCREENTIME[img_lang] +
                                  KEYWORDS_FOR_2018_MOST_USED[img_lang] +
                                  KEYWORDS_FOR_2018_UNLOCKS[img_lang])

    if heads_df.empty:
        return None
    elif not heads_df[heads_df['text'].str.isupper()].empty:
        android_ver = VERSION_2018  # TODO: not sure on the year
    elif max(abs(heads_df['left'] + 0.5 * heads_df['width'] - (0.5 * screenshot.width))) < (0.11 * screenshot.width) and \
            not heads_df[HEADING_COLUMN].eq(VIEW_MORE_HEADING).any():
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
    """

    :param screenshot:
    :return:
    """

    def count_matching_rows(df, format_list):
        count = 0
        for text in df['text']:
            for f in format_list:
                if text[-1] != "#" and OCRScript_v3.levenshtein_distance(text, f) < 4:
                    # If the last character in the row isn't a digit, and the whole row text is close to a time format
                    count += 1
                    break  # Move to the next text after finding a match
        return count

    lang = get_best_language(screenshot)
    heads_df = screenshot.headings_df
    text_df = screenshot.text
    text_df_hashes = text_df.copy()
    text_df_hashes['text'] = text_df_hashes['text'].str.replace(misread_number_format_iOS, '#', regex=True)

    category_submitted = screenshot.category_submitted
    categories_found = []
    
    if any(heads_df[HEADING_COLUMN].eq(SCREENTIME_HEADING)) or \
            any(heads_df[HEADING_COLUMN].eq(TOTAL_SCREENTIME)) or \
            any(heads_df[HEADING_COLUMN].eq(MOST_USED_HEADING)) and \
            (text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].index[0] + 1 or
             heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].iloc[-1]['top'] < 0.9 * screenshot.height) or (
            count_matching_rows(text_df_hashes, GOOGLE_SCREENTIME_FORMATS[lang]) >= 2):
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
                 'top'] < 0.9 * screenshot.height) or (
        count_matching_rows(text_df_hashes, GOOGLE_NOTIFICATIONS_FORMATS[lang]) >= 2):
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
            heads_df[HEADING_COLUMN].str.contains(TOTAL_UNLOCKS).any() or (
        count_matching_rows(text_df_hashes, GOOGLE_UNLOCKS_FORMATS[lang]) >= 2):
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
        elif category_submitted == PICKUPS and UNLOCKS in categories_found:
            category_found = UNLOCKS
        else:
            category_found = categories_found[0]
            print(f"Screenshot submitted under '{category_submitted}' category, but found '{category_found}' instead.")

    print(f"Setting category to '{category_found}'.")

    return category_found


def filter_time_text(text, conf, hr_f, min_f):
    """

    :param text:
    :param conf:
    :param hr_f:
    :param min_f:
    :return:
    """

    if str(text) == NO_TEXT:
        return NO_TEXT, NO_CONF

    def replace_misread_digit(misread, actual, s):
        """

        :param misread:
        :param actual:
        :param s:
        :return:
        """
        # Replaces a 'misread' digit with the 'actual' digit, but only if it is followed by a time word/character
        # (hours or minutes) in the relevant language
        pattern = re.compile(''.join([r"(?<![^0-9tailsh\s\b])", misread, r"(?=\s?[0-9tails]{0,2}\s?(", hr_f, "|", min_f, "))"]), flags=re.IGNORECASE)
        # Note: Don't look behind for r if searching to replace an s, otherwise 'hrs' will become 'hr5'
        filtered_str = re.sub(pattern, actual, s)
        return filtered_str

    text2 = re.sub(r"bre|bra", "hrs", text)
    text2 = re.sub(r"Zhe|zhe", "2hr", text2)
    text2 = re.sub(r"br|Ar", "hr", text2)
    text2 = re.sub(r"(?<=[\d\s])ming$", "mins", text2)
    text2 = re.sub(r"ii", "11", text2, re.IGNORECASE)
    text2 = re.sub(r"((?<=\d\s)tr)|((?<=\d)tr)", "hr", text2)
    text2 = re.sub(r"((?<=\d\s)hy)|((?<=\d)hy)", "hr", text2)
    text2 = re.sub(r"((?<=\d\s)ty)|((?<=\d)ty)", "hr", text2)
    text2 = re.sub(r"((?<=\d\s)he)|((?<=\d)he)", "hr", text2)
    text2 = re.sub(r"((?<=\d\s)by)|((?<=\d)by)", "hr", text2)
    text2 = re.sub(r"((?<=\d\s)kr)|((?<=\d)kr)", "hr", text2)
    text2 = re.sub(r"(7(?=\d))", "1", text2)
    text2 = re.sub(r"(8(?=\d))", "3", text2)  # Might be better to replace 8's with 5's? Have to investigate
    text2 = text2.replace('hr.', 'hr')

    # Replace common misread characters (e.g. pytesseract sometimes misreads '1h' as 'Th'/'th').
    text2 = replace_misread_digit('The', '1hr', text2)
    text2 = replace_misread_digit('(t|T)', '1', text2)
    text2 = replace_misread_digit('(a|A)', '4', text2)
    text2 = replace_misread_digit('(i|I)', '1', text2)
    text2 = replace_misread_digit('(l|L)', '1', text2)
    text2 = replace_misread_digit('(s|S)', '5', text2)
    text2 = replace_misread_digit('(o|O)', '0', text2)
    text2 = replace_misread_digit('(b)', '6', text2)

    text2 = re.sub(r"[^0-9a-zA-Z\s]", '', text2)

    return text2, conf


def filter_number_text(text):
    text2 = text.replace("A", "4")
    text2 = text2.replace("O", "0")
    text2 = text2.replace("I", "1")
    text2 = re.sub(r"L|l", "1", text2)
    text2 = text2.replace("S", "5")
    text2 = text2.replace("T", "1")
    text2 = text2.replace("K", "3")
    if text2 != text:
        print(f"Replaced '{text}' with '{text2}'.")

    return text2


def get_daily_total_and_confidence(screenshot, image, heading):
    """

    :param screenshot:
    :param image:
    :param heading:
    :return:
    """
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

        if show_images_at_runtime:
            OCRScript_v3.show_image(rescan_df, scaled_cropped_image)

        return rescan_df

    if headings_df[HEADING_COLUMN].eq(total_heading).any():
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
        crop_top = max([0, row_below_total['top'] - (4 * row_below_total['height'])])
        crop_bottom = row_below_total['top']
        crop_left = 0
        crop_right = screenshot.width

        cropped_scan = rescan_cropped_area(image, crop_top, crop_bottom, crop_left, crop_right)

    elif android_version == SAMSUNG_2021 and not headings_df[headings_df[HEADING_COLUMN] == heading].empty:
        total_value_1st_scan = NO_TEXT
        total_value_1st_scan_conf = NO_CONF

        row_above_total = headings_df[headings_df[HEADING_COLUMN] == heading].iloc[0]
        crop_top = min([screenshot.height, row_above_total['top'] + (3 * row_above_total['height'])])
        crop_bottom = max([screenshot.height, crop_top + (6 * row_above_total['height'])])
        crop_left = 0
        crop_right = screenshot.width

        cropped_scan = rescan_cropped_area(image, crop_top, crop_bottom, crop_left, crop_right)

    elif android_version == VERSION_2018 and not headings_df[headings_df[HEADING_COLUMN] == heading].empty:
        total_value_1st_scan = NO_TEXT
        total_value_1st_scan_conf = NO_CONF

        row_above_total = headings_df[headings_df[HEADING_COLUMN] == heading].iloc[0]
        crop_top = min([screenshot.height, row_above_total['top'] + (2 * row_above_total['height'])])
        crop_bottom = max([screenshot.height, crop_top + (4 * row_above_total['height'])])
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

    if screenshot.category_detected is not None:
        is_number = False if screenshot.category_detected == SCREENTIME else True
    else:
        is_number = False if screenshot.category_submitted == SCREENTIME else True

    total_value, total_conf = OCRScript_v3.choose_between_two_values(total_value_1st_scan, total_value_1st_scan_conf,
                                                                     total_value_2nd_scan, total_value_2nd_scan_conf,
                                                                     value_is_number=is_number,
                                                                     val_fmt=screenshot.time_format_long)
    total_value = str(total_value)

    if screenshot.category_detected != SCREENTIME:
        try:
            total_value_filtered = re.findall(r'-?\d+', total_value)[0]  # TODO why the - symbol?
        except IndexError:
            total_value_filtered = str.split(total_value)[0]
            total_value_filtered = filter_number_text(total_value_filtered)
    else:
        if android_version == GOOGLE:
            hours_format = '|'.join([('|'.join(KEYWORDS_FOR_HOURS[img_lang])), '|'.join(KEYWORDS_FOR_HR[img_lang])]).replace(" ",
                                                                                                                  r"\s?")
            minutes_format = '|'.join([('|'.join(KEYWORDS_FOR_MINUTES[img_lang])), '|'.join(KEYWORDS_FOR_MIN[img_lang])])
        else:
            hours_format = H
            minutes_format = MIN
        total_value_filtered, total_conf = filter_time_text(total_value, total_conf,
                                                            hours_format, minutes_format)

    moe = int(np.log(len(total_value))) + 1
    if min(OCRScript_v3.levenshtein_distance(total_value[-len(key):], key) <= int(np.log(len(key)))
           for key in KEYWORDS_FOR_YESTERDAY[img_lang]):
        print("Daily total found ends in 'yesterday'. Could not find daily total.")
        return NO_TEXT, NO_CONF

    elif min(OCRScript_v3.levenshtein_distance(total_value, key) for key in SAMSUNG_SEARCH_APPS[img_lang]) <= moe:
        print("Daily total matches heading 'Search apps' (or 'Search categories'). Could not find daily total.")
        return NO_TEXT, NO_CONF

    elif min(OCRScript_v3.levenshtein_distance(total_value, key) for key in GOOGLE_LESS_THAN_1_MINUTE[img_lang]) < moe:
            return '0 ' + KEYWORDS_FOR_MINUTES[img_lang][0], total_conf

    if total_heading == "total " + SCREENTIME and total_value != total_value_filtered:
        print(f"Filtered total: replaced '{total_value}' with '{total_value_filtered}'.")

    return total_value_filtered, total_conf


def convert_string_time_to_minutes(str_time, screenshot):
    """

    :param str_time:
    :param screenshot:
    :return:
    """
    lang = OCRScript_v3.get_best_language(screenshot)
    android_version = screenshot.android_version

    if str(str_time) == NO_TEXT:
        return NO_NUMBER

    def split_time(_s, _format):
        """

        :param _s:
        :param _format:
        :return:
        """
        split = re.split(_format, _s)
        if split[0] != str_time:
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
        hours_format = '|'.join(
            [('|'.join(KEYWORDS_FOR_HOURS[lang])), '|'.join(KEYWORDS_FOR_HR[lang])]).replace(" ",r"\s?")
        minutes_format = '|'.join([('|'.join(KEYWORDS_FOR_MINUTES[lang])), '|'.join(KEYWORDS_FOR_MIN[lang])])
    else:
        hours_format = H
        minutes_format = MIN

    hours, str_after_hours = split_time(str_time, hours_format)
    minutes, _ = split_time(str_after_hours, minutes_format)

    if minutes >= 60 or hours >= 24:
        print(f"'{str_time}' is not a proper time value. Value will be accepted, but screenshot will be flagged.")
        screenshot.add_error(ERR_MISREAD_TIME)

    time_in_min = (hours * 60) + minutes

    return time_in_min


def crop_image_to_app_area(image, headings_above_apps, screenshot, time_format_short):
    """

    :param image:
    :param headings_above_apps:
    :param screenshot:
    :param time_format_short:
    :return:
    """
    lang = get_best_language(screenshot)
    android_version = screenshot.android_version
    date_rows = screenshot.rows_with_date
    text_df = screenshot.text
    headings_df = screenshot.headings_df
    dashboard_category = screenshot.category_detected if screenshot.category_detected is not None else (
        screenshot.category_submitted)

    if android_version == GOOGLE:
        value_formats = (GOOGLE_SCREENTIME_FORMATS[lang] +
                         GOOGLE_NOTIFICATIONS_FORMATS[lang] +
                         GOOGLE_UNLOCKS_FORMATS[lang])
        last_index_headings = headings_df.index[-1]
        moe = 3
        text_df['filtered_text'] = text_df['text'].str.replace(r'\d+', '#', regex=True)

        def matches_any_pattern(text, patterns, m):
            for pattern in patterns:
                if levenshtein_distance(text, pattern) < m and ('#' in text):
                    return True
            return False

        filtered_text_df = text_df[(text_df.index > last_index_headings) &
                                   (text_df['filtered_text'].apply(lambda x: matches_any_pattern(x, value_formats, moe))) &
                                   (text_df['left'] > int(0.1 * screenshot.width))]
        if not filtered_text_df.empty:
            crop_left = max(0, min(filtered_text_df['left']) - int(0.02 * screenshot.width))
            crop_right = max(0, screenshot.width - crop_left - int(0.04 * screenshot.width))
        else:
            # Default values for crop_left and crop_right
            crop_left = int(0.15 * screenshot.width)  # Crop out the app icon area (first 15% of screenshot.width)
            crop_right = int(0.79 * screenshot.width)  # Crop out the hourglass area (last 79% of screenshot.width)

            if not date_rows.empty:
                # If we can find text rows below the date, then use the median 'left' value of that region
                text_df_below_date = text_df[(text_df.index > date_rows.index[0]) &
                                             (text_df.index != text_df.index[-1]) &
                                             (text_df['left'] > int(0.05 * screenshot.width))]
                if not text_df_below_date.empty:
                    crop_left = max(0, int(np.median(text_df_below_date['left']) - 0.02 * screenshot.width))
                    crop_right = max(0, screenshot.width - crop_left - int(0.04 * screenshot.width))

        if REST_OF_THE_DAY in headings_df[HEADING_COLUMN].values:
            row_above_apps = headings_df[headings_df[HEADING_COLUMN] == REST_OF_THE_DAY].iloc[-1]
            crop_top = min([screenshot.height, row_above_apps['top'] + int(2.5 * row_above_apps['height'])])
            crop_bottom = screenshot.height
        elif not date_rows.empty:
            crop_top = min([screenshot.height, date_rows.iloc[-1]['top'] + (2 * date_rows.iloc[-1]['height'])])
            crop_bottom = screenshot.height
        elif not filtered_text_df.empty:
            crop_top = max([0, filtered_text_df.iloc[0]['top'] - 3*int(filtered_text_df.iloc[0]['height'])])
            crop_bottom = screenshot.height
        else:
            # TODO Leaving this as a catch-all for now -- debug later if this condition is used
            crop_top = 0
            crop_bottom = screenshot.height

        if crop_top == 0 and crop_bottom == screenshot.height:
            print("Could not find suitable values for top/bottom of app region.")
            screenshot.add_error(ERR_APP_AREA)

        text_df.drop(columns=['filtered_text'], inplace=True)
        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]

    elif android_version in [SAMSUNG_2024, SAMSUNG_2021, VERSION_2018]:
        crop_left = int(0.08 * screenshot.width) if android_version == SAMSUNG_2024 else int(0.17 * screenshot.width)
        # Crop out the app icon area (first 8% or 17% of screenshot.width)
        crop_right = int(0.9 * screenshot.width) if android_version == VERSION_2018 else screenshot.width
        # Crop out the > arrows to the right of each app (these appear in the Samsung 2018 version of Dashboard)

        if headings_df[HEADING_COLUMN].isin(headings_above_apps + [DAYS_AXIS_HEADING]).any():
            rows_above_apps = headings_df[headings_df[HEADING_COLUMN].isin(headings_above_apps + [DAYS_AXIS_HEADING])]
            if any(heading in rows_above_apps[HEADING_COLUMN].values for heading in headings_above_apps):
                row_above_apps = rows_above_apps[rows_above_apps[HEADING_COLUMN].isin(headings_above_apps)].iloc[0]
            else:
                row_above_apps = rows_above_apps[rows_above_apps[HEADING_COLUMN] == DAYS_AXIS_HEADING].iloc[0]
            crop_top = row_above_apps['top'] + row_above_apps['height']
            headings_below_apps_df = headings_df[headings_df.index > row_above_apps.name]

            text_df_app_area = text_df[(text_df.index > row_above_apps.name) &
                                       (text_df.index != text_df.index[-1]) &
                                       (text_df['left'] > int(0.05 * screenshot.width))]

            if not headings_below_apps_df.empty:
                row_below_apps = headings_below_apps_df.iloc[0]
                text_df_app_area = text_df_app_area[text_df_app_area.index < row_below_apps.name]
                crop_bottom = row_below_apps['top']
            else:
                crop_bottom = screenshot.height

            if not text_df_app_area.empty and android_version != SAMSUNG_2024:
                for i, _idx in enumerate(text_df_app_area.index):
                    if i == 0 or crop_left < text_df_app_area['left'][_idx] < int(0.2 * screenshot.width):
                        crop_left = text_df_app_area['left'][_idx] - int(0.02 * screenshot.width)
                if crop_left < int(0.1 * screenshot.width):
                    crop_left = int(0.15 * screenshot.width)

                # new_crop_left = max(0, int(np.median(text_df_below_heading['left']) - 0.02 * screenshot.width))
                # crop_left = new_crop_left + int(0.1 * screenshot.width if (
                #         android_version != SAMSUNG_2024 and new_crop_left < int(0.15 * screenshot.width)) else 0)

        else:
            print("Sub-heading above app rows not found. Searching for app rows directly.")
            # If android version is SAMSUNG_2024, then search row by row for a row that matches an app w/ number format.
            crop_top = 0
            format_for_time_or_number_eol = '|'.join([time_format_short.replace("^", "\\s"), "\\s\\d+"])

            headings_above_apps_df = headings_df[headings_df[HEADING_COLUMN].str.contains(dashboard_category)]
            index_of_closest_heading = headings_above_apps_df.index[-1] if not headings_above_apps_df.empty else 0
            index_of_first_app = -1

            for i in range(index_of_closest_heading, text_df.shape[0]):
                if text_df['left'][i] < 0.25 * screenshot.width and \
                        text_df['left'][i] + text_df['width'][i] > 0.85 * screenshot.width and \
                        re.search(format_for_time_or_number_eol, text_df['text'][i]) and \
                        text_df['height'][i] < 0.08 * screenshot.width:  # 0.08 settled on through trial and error
                    # Row text spans most of the width of the screenshot, and also ends with a time/number, and also
                    # isn't too tall (sometimes a line with a graph bar + a graph axis number is interpreted as text)
                    print(f"App row found: '{text_df['text'][i]}'. ", end='')
                    if crop_top == 0:
                        print(f"Setting top-left of crop region to the top-left of this app row.")
                        crop_top = max([0, int(text_df['top'][i] - 0.01 * screenshot.width)])
                        crop_left = max([0, int(text_df['left'][i] - 0.02 * screenshot.width)])
                        index_of_first_app = i
                    else:
                        if crop_left < int(0.05 * screenshot.width) and text_df['left'][i] < int(0.25 * screenshot.width):
                            # Occasionally, the app icon is read as part of the app name, which shouldn't be in the crop.
                            # Look for more app rows, and if one is found that starts further from the left edge of the
                            # screenshot, use that as the crop_left bound.
                            print(f"Changing left of crop region to the left of this app row.")
                            crop_left = max([0, int(text_df['left'][i] - 0.02 * screenshot.width)])
                        break

            headings_below_apps_df = headings_df[headings_df.index > max(index_of_first_app, index_of_closest_heading)]
            if not headings_below_apps_df.empty:
                row_below_apps = headings_below_apps_df.iloc[0]
                crop_bottom = row_below_apps['top']
            else:
                crop_bottom = screenshot.height

        if crop_top == 0 and crop_bottom == screenshot.height:
            print("Could not determine suitable crop area; image will not be cropped.")
            return image, [None, None, None, None]

        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
    else:
        print(f"Android version not detected; image will not be cropped.")
        return image, [None, None, None, None]

    if crop_top >= crop_bottom or crop_left >= crop_right:
        # In case the crop region is invalid
        return image, [None, None, None, None]
    else:
        # if screenshot.is_light_mode:
        #     _, cropped_filtered_image = cv2.threshold(cropped_grey_image, 210, 255, cv2.THRESH_BINARY)
        # else:
        #     _, cropped_filtered_image = cv2.threshold(cropped_grey_image, 50, 180, cv2.THRESH_BINARY)

        return cropped_image, [crop_top, crop_left, crop_bottom, crop_right]


def consolidate_overlapping_text(df, time_format_eol):
    # For calculating overlap of two text boxes
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    def calculate_overlap(rect_a, rect_b):
        # Find the overlap in the x-axis
        dx = min(rect_a.xmax, rect_b.xmax) - max(rect_a.xmin, rect_b.xmin)
        # Find the overlap in the y-axis
        dy = min(rect_a.ymax, rect_b.ymax) - max(rect_a.ymin, rect_b.ymin)
        #
        smallest_rect_area = min((rect_a.xmax - rect_a.xmin) * (rect_a.ymax - rect_a.ymin),
                                 (rect_b.xmax - rect_b.xmin) * (rect_b.ymax - rect_b.ymin))

        # If there is an overlap in both x and y, calculate the area
        if (dx >= 0) and (dy >= 0):
            return dx * dy / smallest_rect_area
        else:
            return 0  # No overlap

    # This section checks if two text values physically overlap;
    # if there is enough overlap, it decides which word to consider the 'correct' word.
    rows_to_drop = []
    df = df.sort_values(by=['top', 'left']).reset_index(drop=True)
    for i in df.index:
        if i == 0:
            continue
        current_left = df['left'][i]
        current_right = df['left'][i] + df['width'][i]
        current_top = df['top'][i]
        current_bottom = df['top'][i] + df['height'][i]
        prev_left = df['left'][i - 1]
        prev_right = df['left'][i - 1] + df['width'][i - 1]
        prev_top = df['top'][i - 1]
        prev_bottom = df['top'][i - 1] + df['height'][i - 1]

        current_text = df['text'][i]
        previous_text = df['text'][i - 1]
        current_textbox = Rectangle(current_left, current_top, current_right, current_bottom)
        prev_textbox = Rectangle(prev_left, prev_top, prev_right, prev_bottom)

        current_num_digits = len(re.findall(r'\d', current_text))
        prev_num_digits = len(re.findall(r'\d', previous_text))

        if calculate_overlap(current_textbox, prev_textbox) > 0.5:  # Used to be 0.3; revert if it causes issues.
            # If two text boxes overlap by at least 50%, consider them to be two readings of the same text.
            if (re.search(time_format_eol, df.loc[i, 'text']) and
                    not re.search(time_format_eol, df.loc[i - 1, 'text'])):
                rows_to_drop.append(i - 1)
            elif (not re.search(time_format_eol, df.loc[i, 'text']) and
                  re.search(time_format_eol, df.loc[i - 1, 'text'])):
                rows_to_drop.append(i)
            elif current_text == "X" and previous_text != "X":
                rows_to_drop.append(i - 1)
            elif current_text != "X" and previous_text == "X":
                rows_to_drop.append(i)
            elif current_num_digits > prev_num_digits:
                rows_to_drop.append(i - 1)
            elif current_num_digits < prev_num_digits:
                rows_to_drop.append(i)
            elif len(previous_text) > len(current_text):
                rows_to_drop.append(i)
            elif len(current_text) > len(previous_text):
                rows_to_drop.append(i - 1)
            elif df['conf'][i - 1] > df['conf'][i]:
                if previous_text == 'min':
                    rows_to_drop.append(i - 1)
                else:
                    rows_to_drop.append(i)
            else:
                if current_text == 'min':
                    rows_to_drop.append(i)
                else:
                    rows_to_drop.append(i - 1)

    if 'level_0' in df.columns:
        df.drop(columns=['level_0'], inplace=True)

    merged_df = df.drop(index=rows_to_drop).reset_index()
    consolidated_df = OCRScript_v3.merge_df_rows_by_height(merged_df)

    if not merged_df.equals(consolidated_df):
        print("Some rows were consolidated.")

    return consolidated_df


def get_app_names_and_numbers(screenshot, df, category, max_apps, time_formats, coordinates):

    num_missed_app_values = 0
    android_version = screenshot.android_version
    time_short, time_long, time_eol = time_formats[0], time_formats[1], time_formats[2]
    crop_top, crop_left, crop_bottom, crop_right = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    crop_width = crop_right - crop_left
    img_lang = get_best_language(screenshot)
    empty_name_row = pd.DataFrame({NAME: [NO_TEXT], NAME_CONF: [NO_CONF]})
    empty_number_row = pd.DataFrame({NUMBER: [NO_TEXT], NUMBER_CONF: [NO_CONF]}) if category == SCREENTIME else (
                       pd.DataFrame({NUMBER: [NO_NUMBER], NUMBER_CONF: [NO_CONF]}))
    app_names = empty_name_row.copy()
    app_numbers = empty_number_row.copy()

    def build_app_and_number_dfs(app, num):
        nonlocal previous_text
        nonlocal app_names
        nonlocal app_numbers
        nonlocal num_missed_app_values

        if previous_text == NUMBER:
            if app != '' and num != '':  # App and its number in same row)
                new_name = pd.DataFrame({NAME: [app], NAME_CONF: [row_conf]})
                new_number = pd.DataFrame({NUMBER: [num], NUMBER_CONF: [row_conf]})
                app_names = new_name if app_names.empty else pd.concat([app_names, new_name], ignore_index=True)
                app_numbers = new_number if app_numbers.empty else pd.concat([app_numbers, new_number], ignore_index=True)

            elif app == '':  # Only number
                if len(app_names) - 1 < max_apps:
                    num_missed_app_values += 1
                app_names = empty_name_row if app_names.empty else pd.concat([app_names, empty_name_row], ignore_index=True)
                new_number = pd.DataFrame({NUMBER: [num], NUMBER_CONF: [row_conf]})
                app_numbers = new_number if app_numbers.empty else pd.concat([app_numbers, new_number], ignore_index=True)

            elif num == '':  # Only app
                new_name = pd.DataFrame({NAME: [app], NAME_CONF: [row_conf]})
                app_names = new_name if app_names.empty else pd.concat([app_names, new_name], ignore_index=True)
                previous_text = NAME

        elif previous_text == NAME:
            if app != '' and num != '':  # App and its number in same row
                if len(app_names) < max_apps:
                    num_missed_app_values += 1
                app_numbers = empty_number_row if app_numbers.empty else pd.concat([app_numbers, empty_number_row], ignore_index=True)

                new_name = pd.DataFrame({NAME: [app], NAME_CONF: [row_conf]})
                new_number = pd.DataFrame({NUMBER: [num], NUMBER_CONF: [row_conf]})
                app_names = new_name if app_names.empty else pd.concat([app_names, new_name], ignore_index=True)
                app_numbers = new_number if app_numbers.empty else pd.concat([app_numbers, new_number], ignore_index=True)
                previous_text = NUMBER

            elif app == '':  # Only number
                new_number = pd.DataFrame({NUMBER: [num], NUMBER_CONF: [row_conf]})
                app_numbers = new_number if app_numbers.empty else pd.concat([app_numbers, new_number], ignore_index=True)
                previous_text = NUMBER

            elif num == '':  # Only app name
                if len(app_names) - 1 <= max_apps:
                    num_missed_app_values += 1
                app_numbers = empty_number_row if app_numbers.empty else pd.concat([app_numbers, empty_number_row], ignore_index=True)
                new_name = pd.DataFrame({NAME: [app], NAME_CONF: [row_conf]})
                app_names = new_name if app_names.empty else pd.concat([app_names, new_name], ignore_index=True)

        if android_version != GOOGLE and len(app_names) == max_apps and num == '':
            num_missed_app_values += 1

    def split_app_name_and_screen_time(s):
        def replace_misread_time_words(_s, dict_row, _moe):
            # If we take the last word in the string 's' and it's very close to a time keyword, then replace that
            # last word with the corresponding keyword (i.e. 'hours', 'minutes').
            s_words = _s.split()
            s_last_word = s_words[-1]
            for val in dict_row:
                if s_last_word == val:
                    return _s

            for val in dict_row:
                if OCRScript_v3.levenshtein_distance(s_last_word, val) <= _moe:
                    s_words[-1] = val
                    _s = ' '.join(s_words)
                    break

            return _s

        hours_format = '|'.join([H,
                                 '|'.join(KEYWORDS_FOR_HR[img_lang]),
                                 ('|'.join(KEYWORDS_FOR_HOURS[img_lang]))]).replace(" ",r"\s?")
        minutes_format = '|'.join([MIN, '|'.join(KEYWORDS_FOR_MIN[img_lang]), ('|'.join(KEYWORDS_FOR_MINUTES[img_lang]))])

        if screenshot.android_version == GOOGLE:
            _moe = round(np.log(len(s))) + 1 if len(s) >= 1 else 1
            if min(levenshtein_distance(s, item) for item in GOOGLE_LESS_THAN_1_MINUTE[img_lang]) < _moe:
                return '', '0 ' + KEYWORDS_FOR_MINUTES[img_lang][0]
            # Sometimes words like 'minutes' can be misread as something like 'minuies'. These are still time values,
            # so we still want to process them as time values.
            s = replace_misread_time_words(s, KEYWORDS_FOR_MINUTES[img_lang], 2)
            s = replace_misread_time_words(s, KEYWORDS_FOR_HOURS[img_lang], 1)

        filtered_s, _ = filter_time_text(s, NO_CONF, hours_format, minutes_format)
        if re.match('|'.join([time_short, time_long]), filtered_s):
            # If the entire string matches a time format, then it must be a time only (no app name)
            name = ''
            time = filtered_s
            if time != s:
                print(f"Filtering time text: Replaced '{s}' with '{time}'.")
        else:
            split_text = re.split(time_eol, filtered_s)
            if len(split_text) == 1:
                # If the string does not end in a time format, then it must be an app name only (no time)
                return s, ''
            name = s[:len(split_text[0])]  # split_text[0].strip()
            s_time_only = s.replace(name, "").strip()
            time, _ = filter_time_text(s_time_only, NO_CONF, hours_format, minutes_format)
            if time != s_time_only:
                print(f"Filtering time text: Replaced '{s_time_only}' with '{time}'.")

        return name, time

    def split_app_name_and_notifications(s):
        # Find all the numbers in the string
        numbers = re.findall(misread_number_format_iOS, s)
        if not numbers and (not s[-1].isdigit() or abs(crop_right - row_right) > 0.2*crop_right):
            # If there are no (misread) numbers, and either
            #     the last character is a non-digit, or
            #     the text ends too close to the right edge of the image
            return s, ''

        # Get the last number
        if numbers:
            last_number = numbers[-1]
            # Find the index of the first digit of the last number
            index = s.rfind(last_number)
        else:
            index = len(s)

        name = s[:index].rstrip()
        s_filtered = filter_number_text(s[index:])
        number = re.split(r'\s|[a-zA-Z]', s_filtered)[0]
        try:
            number = int(number)
        except ValueError:
            print(f"Error: could not convert {number} to an integer. Number will be set to {NO_NUMBER}.")
            number = NO_NUMBER
        # Split the string at the index of the first digit of the last number

        return name, number

    moe_show_sites = round(min(np.log(len(key)) for key in KEYWORDS_FOR_SHOW_SITES_YOU_VISIT[img_lang]))
    previous_text = NUMBER  # initialize to handle the first text beginning with an app name (most likely case)

    if category == SCREENTIME:
        prev_row_bottom = -1
        for row_text, row_conf, row_left, row_right, row_top, row_bottom in zip(df['text'],
                                                                                df['conf'],
                                                                                df['left'],
                                                                                (df['left'] + df['width']),
                                                                                df['top'],
                                                                                (df['top'] + df['height'])):
            # row_text = re.sub(r'^[xX]{1,2}$', "X", row_text)  # X (Twitter) may show up here as xX
            row_height = row_bottom - row_top
            if min(OCRScript_v3.levenshtein_distance(row_text, key) for key in
                   KEYWORDS_FOR_SHOW_SITES_YOU_VISIT[img_lang]) < moe_show_sites:
                # A button saying 'Show sites that you visit' appears below the Google Chrome web browser, which is not
                # an app name, so it should be ignored
                continue

            app_name, app_number = split_app_name_and_screen_time(row_text)
            if app_name != '' and (app_number == '' and row_left + crop_left > (0.4 * screenshot.width)):
                # Sometimes there are 'pill' shapes above the app time; these can be misread as app names.
                # Ignore such apparent app names whose left edges lie beyond 40% of the screenshot width.
                continue
            if android_version == GOOGLE and app_name != '' and previous_text == NAME and \
                    row_top - prev_row_bottom < row_height and len(app_names) - 1 <= max_apps:
                num_missed_app_values += 1
                # Sometimes an app time that appears just below an app name can be interpreted as an app name,
                # even after filtering. Ignore such misread app times.
                continue

            build_app_and_number_dfs(app_name, app_number)
            prev_row_bottom = row_bottom
    elif category == NOTIFICATIONS:
        for row_text, row_conf, row_left, row_right in zip(df['text'],
                                                           df['conf'],
                                                           df['left'],
                                                           (df['left'] + df['width'])):
            # row_text = re.sub(r'^[xX]{1,2}$', "X", row_text)  # X (Twitter) may show up here as xX
            if min(OCRScript_v3.levenshtein_distance(row_text, key) for key in
                   KEYWORDS_FOR_SHOW_SITES_YOU_VISIT[img_lang]) < moe_show_sites:
                # A button saying 'Show sites that you visit' appears below the Google Chrome web browser, which is not
                # an app name, so it should be ignored
                continue
            app_name, app_number = split_app_name_and_notifications(row_text)
            moe = 2 * int(np.log(max(len(key) for key in GOOGLE_NOTIFICATIONS_FORMATS[img_lang])))
            if app_name != '' and app_number == '':  # Only app name found
                if row_left + crop_left > (0.4 * screenshot.width):
                    # See 'pill' comment in similar line above
                    continue
                elif min(levenshtein_distance("# " + app_name, key) for key in
                         GOOGLE_NOTIFICATIONS_FORMATS[img_lang]) <= moe:
                    # plus, sometimes in '## notifications', only 'notifications' is read.
                    num_missed_app_values += 1
                    continue
                elif app_name.split()[-1].isdigit() and crop_left + row_right > 0.85 * screenshot.width:
                    # For screenshots in SAMSUNG 2024 format or screenshots with weekly info,
                    # sometimes the number of notifications has no text after it
                    # (e.g. "14" instead of "14 notifications").
                    app_number = app_name.split()[-1]
                    app_name = ' '.join(app_name.split()[:-1])
            build_app_and_number_dfs(app_name, app_number)
    elif category == UNLOCKS:
        if android_version != GOOGLE:
            empty_rows = [{NAME: NO_TEXT, NAME_CONF: NO_CONF}] * max_apps
            app_names = pd.concat([app_names, empty_rows], ignore_index=True)
            empty_rows = [{NUMBER: NO_NUMBER, NUMBER_CONF: NO_CONF}] * max_apps
            app_numbers = pd.concat([app_numbers, empty_rows], ignore_index=True)
        else:
            # Only the Google Dashboard has app-level unlocks info;
            # other Dashboard formats only show total unlocks.
            for row_text, row_conf in zip(df['text'], df['conf']):
                # row_text = re.sub(r'^[xX]{1,2}$', "X", row_text)  # X (Twitter) may show up here as xX
                if min(OCRScript_v3.levenshtein_distance(row_text, key) for key in
                       KEYWORDS_FOR_SHOW_SITES_YOU_VISIT[img_lang]) < moe_show_sites:
                    # A button saying 'Show sites that you visit' appears below the Google Chrome web browser, which is
                    # not an app name, so it should be ignored
                    continue
                moe_unlocks = round(np.log(len(row_text)))
                row_text_filtered = re.sub(r'\d+', '#', row_text)
                if min(OCRScript_v3.levenshtein_distance(row_text_filtered, key) for key in
                       GOOGLE_UNLOCKS_FORMATS[img_lang]) <= moe_unlocks:
                    # Row text contains an unlocks value
                    try:
                        app_number = re.findall(r'\d+', row_text)[0]
                        app_number = int(app_number)
                        conf = row_conf
                    except (IndexError, ValueError) as e:
                        print(f"Error: Could not extract app number from '{row_text}'. \n"
                              f"App number will be set to {NO_NUMBER} (conf = {NO_CONF}).")
                        app_number = NO_NUMBER
                        conf = NO_CONF
                    if previous_text == NUMBER:
                        if len(app_names) <= max_apps:
                            num_missed_app_values += 1
                        app_names = empty_name_row if app_names.empty else (
                            pd.concat([app_names, empty_name_row], ignore_index=True))
                    new_number_row = pd.DataFrame({NUMBER: [app_number], NUMBER_CONF: [conf]})
                    app_numbers = new_number_row if app_numbers.empty else (
                        pd.concat([app_numbers, new_number_row], ignore_index=True))
                    previous_text = NUMBER
                else:
                    # Row text contains an app name
                    if previous_text == NAME:
                        if len(app_names) < max_apps:
                            num_missed_app_values += 1
                        app_numbers = empty_number_row if app_numbers.empty else (
                            pd.concat([app_numbers, empty_number_row], ignore_index=True))
                    new_name_row = pd.DataFrame({NAME: [row_text], NAME_CONF: [row_conf]})
                    app_names = new_name_row if app_names.empty else (
                        pd.concat([app_names, new_name_row], ignore_index=True))
                    previous_text = NAME

    if num_missed_app_values > 0:
        screenshot.add_error(ERR_MISSING_VALUE, num_missed_app_values)

    if (~app_numbers[NUMBER_CONF].eq(NO_CONF)).any() and screenshot.category_detected is None:
        # Sometimes the dashboard category is not detected, but app-level data from the correct category is extracted.
        # In this case, we can set the detected category to the same category as the data extracted.
        print(f"Dashboard category not detected, but {category} data extracted. "
              f"Setting detected category to '{category}'.")
        screenshot.set_category_detected(category)

    while app_names.shape[0] < max_apps + 1:
        app_names = pd.concat([app_names, empty_name_row], ignore_index=True)
    while app_numbers.shape[0] < max_apps + 1:
        app_numbers = pd.concat([app_numbers, empty_number_row], ignore_index=True)

    app_names[NAME] = app_names[NAME].apply(lambda x: re.sub(r'\bAl\b', 'AI', x))
    app_names[NAME] = app_names[NAME].apply(lambda x: re.sub(r'^4$', 'X', x))
    app_names[NAME] = app_names[NAME].apply(lambda x: re.sub(r'\.$', '', x))
    app_names, app_numbers = app_names.drop(app_names.index[0]), app_numbers.drop(app_numbers.index[0])

    # app_names.loc[app_names[NAME] == 'Lite', NAME] = 'Facebook Lite'  # The app "Facebook Lite" appears as 'Lite'

    # Having initialized app_names and app_numbers with an empty row (at index 0), the indexes of the app rows
    # line up with the app ordinals. (The 1st app in the screenshot is at index 1, etc.)
    # This makes it easier to compare a row of existing data to a row of new data (when the current screenshot is for
    # a person & day & category for which data already exists).
    top_n_app_names_and_numbers = pd.concat(
        [app_names.head(max_apps), app_numbers.head(max_apps)], axis=1)

    return top_n_app_names_and_numbers

