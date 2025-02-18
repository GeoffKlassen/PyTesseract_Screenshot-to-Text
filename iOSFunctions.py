"""This file contains iOS-specific dictionaries, functions, and variables."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import re
import warnings
import cv2

import OCRScript_v3
from RuntimeValues import *
from ConvenienceVariables import *
from OCRScript_v3 import extract_text_from_image, choose_between_two_values
from collections import namedtuple

"""
    iOS-Specific dictionaries
    
    Some keyword dictionaries are iOS-specific. For example, iOS has a LIMITATIONS heading while Android doesn't.
    Conversely, Android has a dictionary for HOURS, but iOS doesn't need one because all iOS phones use 'h' for hours.
"""
# Order in Dashboard: SCREEN TIME, (LIMITS), MOST USED, PICKUPS, FIRST PICKUP, FIRST USED AFTER PICKUP, NOTIFICATIONS
# The LIMITS heading only appears if the user's phone has a daily time limit set for at least one app.
KEYWORDS_FOR_SCREEN_TIME = {ITA: ['TEMPO DI UTILIZZO'],
                            ENG: ['SCREEN TIME'],
                            FRA: ["TEMPS D'ECRAN"],
                            GER: ['BILDSCHIRMZEIT']}

KEYWORDS_FOR_LIMITATIONS = {ITA: ['LIMITAZIONI'],
                            ENG: ['LIMITS'],
                            FRA: ['TODO FILL THIS IN'],  # TODO Fill this in
                            GER: ['TODO FILL THIS IN']}  # TODO Fill this in

KEYWORDS_FOR_MOST_USED = {ITA: ['PIU UTILIZZATE'],
                          ENG: ['MOST USED'],
                          FRA: ['LES PLUS UTILISEES'],  # LES PLUS UTILISÉES
                          GER: ['VERWENDET']}  # Real heading is AM HÄUFIGSTEN VERWENDET but VERWENDET is on its own line

KEYWORDS_FOR_PICKUPS = {ITA: ['ATTIVAZIONI SCHERMO'],
                        ENG: ['PICKUPS', 'PICK-UPS'],  # Some versions of iOS use the hyphenated form PICK-UPS
                        FRA: ['PRISES EN MAIN'],
                        GER: ['AKTIVIERUNGEN']}

KEYWORDS_FOR_FIRST_PICKUP = {ITA: ['1 attivazione schermo Totale', '1 attivazione schermo', 'schermo', 'Totale'],
                             ENG: ['First Pickup Total Pickups', 'First Pickup', 'Total Pickups'],
                             FRA: ['Premiere prise en main', 'Total des prises en main'],
                             GER: ['1 Aktivierung Aktivierungen insgesamt', '1 Aktivierung',
                                   'Aktivierungen insgesamt', 'insgesamt']}

KEYWORDS_FOR_FIRST_USED_AFTER_PICKUP = {ITA: ['PRIME APP UTILIZZATE DOPO LATTIVAZIONE',
                                              'PRIME APP UTILIZZATE DOPO', 'LATTIVAZIONE'],  # When on separate lines
                                        ENG: ['FIRST USED AFTER PICKUP', 'FIRST USED AFTER PICK UP',
                                              'USED AFTER PICKUP'],
                                        FRA: ['PREMIERE APP UTILISEE'],  # PREMIÈRE APP UTILISÉE
                                        GER: ['1 NUTZUNG NACH AKTIVIERUNG', 'AKTIVIERUNG']}

KEYWORDS_FOR_NOTIFICATIONS = {ITA: ['NOTIFICHE'],
                              GER: ['BENACHRICHTIGUNGEN', 'MITTEILUNGEN'],
                              ENG: ['NOTIFICATIONS', 'NOTIFICATIONS RECEIVED'],
                              FRA: ['NOTIFICATIONS', 'NOTIFICATIONS RECUES', 'RECUES']}

KEYWORDS_FOR_HOURS_AXIS = ['00 06', '06 12', '12 18',
                           '6 12 ', '6 18 ', '42 48',
                           '00h 06h', '06h 12h', '12h 18h',
                           r'12\s?AM 6', r'6\s?AM 12', r'12\s?PM 6',
                           'AM 6AM', 'AM 12PM', 'PM 6PM',
                           '6AM 6PM', '12AM 6PM',
                           'AM PM', 'PM AM', 'PM PM', 'AM AM',
                           'mele 12', '112 118', r'0\s+.*\s+12',
                           '00 Uhr 06 Uhr', '06 Uhr 12 Uhr', '12 Uhr 18 Uhr', 'Uhr Uhr',
                           r'^0\s.*12|6\s.*18$']  # TODO This method is a bit messy

# Variables for iOS time formats
# Even though the words for 'hours', 'minutes', and 'seconds' differ by language, iOS uses h/min/m/s for all languages.
MINUTES_FORMAT = r'(min|m)'
HOURS_FORMAT = 'h'
SECONDS_FORMAT = 's'

english_months = MONTH_ABBREVIATIONS[ENG]
month_mapping = {mon: english_months.index(mon) + 1 for mon in english_months}  # 1:January, 2:February, 3:March, etc.


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


def get_date_regex(lang):
    """
    Creates the date regular expression to use when looking for text in a screenshot that matches a date format.
    :param lang: The language to use for month abbreviations
    :return: The full date regex of all possible date formats for the given language
    """
    patterns = []
    for _format in DATE_FORMAT[lang]:
        # Replace the 'MMM's in DATE_FORMAT with the appropriate 3-4 letter abbreviations for the months.
        patterns.append(re.sub('MMM', ''.join(['(', '|'.join(MONTH_ABBREVIATIONS[lang]), ')']), _format))
    date_regex = '|'.join(patterns)
    return date_regex


def get_date_in_screenshot(screenshot):
    """
    Checks if any text found in the given screenshot matches the appropriate date pattern based on the language of the
    image. If there's a match, return it.
    :param screenshot: The screenshot to search for a date.
    :return: The date-format value of the date found in the screenshot text (if any), otherwise 'None'.
    """

    df = screenshot.text
    lang = get_best_language(screenshot)
    # Different languages display dates in different formats. Create the correct regex pattern for the date.
    date_pattern = get_date_regex(lang)

    try:
        # Pull out the first row of df where the text contains the date regex
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            row_with_date = df[df['text'].str.contains(date_pattern, regex=True, case=False)].iloc[0]

        # Extract the date, month, and day from that row of text, as strings
        date_detected = re.search(date_pattern, row_with_date['text'], flags=re.IGNORECASE).group()
        month_detected = re.search(r'[a-zA-Z]+', date_detected).group().lower()
        day_detected = re.search(r'\d+', date_detected).group()

        # Create a translation dictionary to replace non-English month names with English ones.
        months_to_replace = MONTH_ABBREVIATIONS[lang]
        for i, abbr in enumerate(months_to_replace):
            month_detected = month_detected.replace(abbr, english_months[i])
        month_detected = month_detected[0:3]  # datetime.strptime (used below) requires month to be 3 characters

        try:
            # Convert the string date to a date object
            date_object = datetime.strptime(f"{day_detected} {month_detected}", "%d %b")

            # Get the numeric month value from the mapping
            month_numeric = month_mapping.get(month_detected)
            if month_numeric:
                # Construct the complete date with the year
                complete_date = date_object.replace(year=int(screenshot.date_submitted.year), month=month_numeric)
                if (screenshot.date_submitted - complete_date.date()).days < 0:
                    # In case a screenshot of a previous year is submitted after the new year, correct the year.
                    complete_date = date_object.replace(year=(int(screenshot.date_submitted.year) - 1))
                print(f"Date text detected: \"{row_with_date['text']}\".  Setting date to {complete_date.date()}.")
                return complete_date.date()
            else:
                print("Invalid month abbreviation.")
        except ValueError:
            print("Invalid date format.")
    except:
        print("No date text detected.")

    return None


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


def get_day_type_in_screenshot(screenshot):
    """
    Determines whether the given screenshot contains daily data (today, yesterday, weekday) or weekly data.
    :param screenshot: The screenshot to find the date (range) for
    :returns: tuple: (A day identifier, the row of the text in the screenshot that contains the day identifier)
    """
    lang = get_best_language(screenshot)
    df = screenshot.text.copy()
    date_pattern = get_date_regex(lang)
    # moe = margin of error
    # Set how close a spelling can be to a keyword in order for that spelling to be considered the (misread) keyword.
    moe_yesterday = round(np.log(max((len(string) for string in KEYWORDS_FOR_YESTERDAY[lang]))))
    moe_today = round(np.log(max((len(string) for string in KEYWORDS_FOR_TODAY[lang]))))
    moe_weekday = round(np.log(max((len(string) for string in KEYWORDS_FOR_WEEKDAY_NAMES[lang]))))
    moe_week_keyword = round(np.log(max((len(string) for string in KEYWORDS_FOR_WEEK[lang]))))

    df['next_text'] = df['text'].shift(-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows_with_yesterday = df[(df['text'].apply(
            # Row contains yesterday, and (1) also contains date or (2) the next row contains a date
            lambda x: min(levenshtein_distance(row_word, key)
                          for row_word in str.split(x)
                          for key in KEYWORDS_FOR_YESTERDAY[lang])) <= moe_yesterday) &
                                 ((df['text'].str.contains(date_pattern, case=False)) |
                                  (df['next_text'].str.contains(date_pattern, case=False)))]
        # rows_with_yesterday.drop(columns=['next_text'], inplace=True)
        rows_with_today = df[(df['text'].apply(
            # Row contains today, and (1) also contains date, or (2) the next row contains a date
            lambda x: min(levenshtein_distance(row_word, key)
                          for row_word in str.split(x)
                          for key in KEYWORDS_FOR_TODAY[lang])) <= moe_today) &
                             ((df['text'].str.contains(date_pattern, case=False)) |
                              (df['next_text'].str.contains(date_pattern, case=False)))]
        # rows_with_today.drop(columns=['next_text'], inplace=True)

        rows_with_weekday = df[(df['text'].apply(
            # Row contains name of weekday, and (1) also contains date, or (2) the next row contains a date
            lambda x: min(levenshtein_distance(str.split(x)[0], key)
                          for key in KEYWORDS_FOR_WEEKDAY_NAMES[lang])) <= moe_weekday) &
                               ((df['text'].str.contains(date_pattern, case=False)) |
                                (df['next_text'].str.contains(date_pattern, case=False)))]
        # rows_with_weekday.drop(columns=['next_text'], inplace=True)

        rows_with_week_keyword = df[(df['text'].apply(
            # Row contains one of the keywords for a week-format screenshot (e.g. Daily Average)
            lambda x: min(levenshtein_distance(x, key) for key in KEYWORDS_FOR_WEEK[lang])) <= moe_week_keyword)]
        # rows_with_week_keyword.drop(columns=['next_text'], inplace=True)

    if rows_with_yesterday.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_yesterday.iloc[0]['text']}\".  Setting image type to '{YESTERDAY}'.")
        return YESTERDAY, rows_with_yesterday
    elif rows_with_today.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_today.iloc[0]['text']}\".  Setting image type to '{TODAY}'.")
        return TODAY, rows_with_today
    elif rows_with_weekday.shape[0] > 0:
        print(f"Day text detected: \"{rows_with_weekday.iloc[0]['text']}.\"  "
              f"Setting image type to '{DAY_OF_THE_WEEK}'.")
        return DAY_OF_THE_WEEK, rows_with_weekday
    elif rows_with_week_keyword.shape[0] > 0:
        print(f"Week text detected: \"{rows_with_week_keyword.iloc[0]['text']}\".  Setting image type to '{WEEK}'.")
        return WEEK, rows_with_week_keyword
    else:
        return None, None


def get_headings(screenshot):
    """
    Finds the rows of text within the given screenshot that contain key headings ("SCREENTIME", "MOST USED", etc.)
    :param screenshot: The screenshot to search for headings
    :return: A df that contains only the rows with headings from the text within the screenshot (if none, an empty df)
    """
    df = screenshot.text.copy()
    day_type_rows = screenshot.rows_with_day_type
    lang = get_best_language(screenshot)

    df[HEADING_COLUMN] = None

    # If a row in the screenshot (closely) matches the format of a heading, label that row as that heading type.
    for i in df.index:
        row_text = df['text'][i]
        error_margin = round(np.log(len(str(row_text))))
        if day_type_rows is not None and i in day_type_rows.index:
            df.loc[i, HEADING_COLUMN] = DAY_OR_WEEK_HEADING
        elif re.match(screenshot.date_format, row_text, re.IGNORECASE):
            df.loc[i, HEADING_COLUMN] = DATE_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in KEYWORDS_FOR_SCREEN_TIME[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = SCREENTIME_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in KEYWORDS_FOR_LIMITATIONS[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = LIMITS_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in KEYWORDS_FOR_MOST_USED[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = MOST_USED_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in KEYWORDS_FOR_PICKUPS[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = PICKUPS_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in
                 KEYWORDS_FOR_FIRST_PICKUP[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = FIRST_PICKUP_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in
                 KEYWORDS_FOR_FIRST_USED_AFTER_PICKUP[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = FIRST_USED_AFTER_PICKUP_HEADING
        elif min(levenshtein_distance(row_text, keyword) for keyword in
                 KEYWORDS_FOR_NOTIFICATIONS[lang]) <= error_margin:
            df.loc[i, HEADING_COLUMN] = NOTIFICATIONS_HEADING
        elif re.search('|'.join(KEYWORDS_FOR_HOURS_AXIS), row_text):  # or re.search(r'^0\s.*12|6\s.*18$', row_text):
            df.loc[i, HEADING_COLUMN] = HOURS_AXIS_HEADING
        else:
            df = df.drop(i)

    if df.shape[0] > 0:
        print("\n\nHeadings found:")
        print(df[['text', 'heading']])
    else:
        print("No headings found.")
    print()

    return df


def get_dashboard_category(screenshot):
    """
    Determines the categories of data present in the screenshot image, and chooses which one to assign to the screenshot.

    :param screenshot: The screenshot to search for categories.
    :return: (String) The category of data to search for in the image, or None (if no categories are detected).
    """
    # Headings in iOS Dashboard appear in this order:
    #     SCREEN TIME,  (LIMITS),  MOST USED,  PICKUPS,  FIRST PICKUP,  FIRST USED AFTER PICKUP,  NOTIFICATIONS
    # This function checks for the occurrence of these headings in that order.
    heads_df = screenshot.headings_df
    text_df = screenshot.text
    backup_category = screenshot.category_submitted

    categories_found = []

    if heads_df[HEADING_COLUMN].str.contains(SCREENTIME_HEADING).any() or \
            (heads_df[HEADING_COLUMN].str.contains(LIMITS_HEADING).any() and
             heads_df[HEADING_COLUMN].str.contains(HOURS_AXIS_HEADING).any() and
             heads_df[heads_df[HEADING_COLUMN] == LIMITS_HEADING].index[0] >
             heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].index[0]) or \
            heads_df[HEADING_COLUMN].str.contains(MOST_USED_HEADING).any() and (
            text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].index[0] + 1 or
            heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].iloc[-1]['top'] <
            0.9 * screenshot.height):
        # Found screentime heading; or
        # Found limits heading and hours row, and limits is below hours; or
        # Found most used heading and either:
        #     text_df has more data below the most used heading, or
        #     the most used heading is not too close to the bottom of the screenshot
        categories_found.append(SCREENTIME)

    if heads_df[HEADING_COLUMN].str.contains(PICKUPS_HEADING).any() or \
            heads_df[HEADING_COLUMN].str.contains(FIRST_USED_AFTER_PICKUP_HEADING).any() and (
            text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == FIRST_USED_AFTER_PICKUP_HEADING].index[0] + 1 or
            heads_df[heads_df[HEADING_COLUMN] == FIRST_USED_AFTER_PICKUP_HEADING].iloc[-1]['top'] <
            0.9 * screenshot.height):
        # Found pickups heading; or
        # Found "first used after pickup" heading and either:
        #     there's more data below, or
        #     the "first used after pickup" heading is not close to the bottom of the screenshot
        categories_found.append(PICKUPS)

    if heads_df[HEADING_COLUMN].str.contains(NOTIFICATIONS_HEADING).any() or \
            heads_df[HEADING_COLUMN].str.contains(HOURS_AXIS_HEADING).any() and (
            not heads_df[HEADING_COLUMN].str.contains(MOST_USED_HEADING).any() or
            heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].index[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].index[0]) and (
            text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].index[0] + 1 or
            heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].iloc[-1]['top'] <
            0.9 * screenshot.height):
        # Found notifications heading; or
        # Found hours row and either:
        #     there's more data below, or
        #     the hours row is not close to the bottom of the screenshot
        categories_found.append(NOTIFICATIONS)

    if not categories_found:
        print(f"No categories detected.")  # Defaulting to submitted category: {backup_category}")
        # Returning None to flag the screenshot for manual review;
        # screenshot category is set to backup_category after this function is called
        return None
    else:
        print(f"Category submitted: {screenshot.category_submitted}    ", end='')
        print(f"Categories detected: {categories_found}")
        if backup_category in categories_found:
            print(f"Setting category to: {backup_category}")
            return backup_category
        else:
            print(f"Screenshot submitted under {backup_category} category, but found {categories_found[0]} instead. "
                  f"Setting category to {categories_found[0]}.")
            return categories_found[0]


def filter_time_or_number_text(text, conf, f):
    if str(text) == NO_TEXT:
        return NO_TEXT, NO_CONF

    fmt = re.compile(f, re.IGNORECASE)
    matches = re.findall(fmt, text)
    if matches:
        text = str(max(matches, key=len))
    else:
        return NO_TEXT, NO_CONF

    # Replace common misread characters (e.g. pytesseract sometimes misreads '1h' as 'Th'/'th').
    text = re.sub(r'[TtIi](?=.?[hm])', '1', text)  # 1 (before h or m) can be misread as T/I
    text = re.sub('ah', '4h', text)  # 4h can be misread as ah
    text = re.sub('oh', '5h', text)  # 5h can be misread as oh
    text = re.sub('Qh', '9h', text)  # 9h can be misread as Qh
    text = re.sub(r'(Os|O s)', '0s', text)  # 0s can be misread as Os (letter O and letter s)
    text = re.sub('A', '4', text)  # 4 can be misread as A
    text = text.lower()

    # Remove any characters that aren't a digit or a 'time' character ('h' = hours, 'min' = minutes, 's' = seconds)
    text = re.sub(r'[^0-9hmins]', '', text)

    return text, conf


def get_daily_total_and_confidence(screenshot, img, category=None):
    df = screenshot.text.copy()
    category = screenshot.category_detected if category is None else category
    headings_df = screenshot.headings_df
    rows_with_day_type = screenshot.rows_with_day_type

    print(f"\nSearching for total {category}:")

    # Initialize first scan values
    daily_total_1st_scan = NO_TEXT
    daily_total_1st_scan_conf = NO_CONF

    value_pattern = misread_time_format if category == SCREENTIME else misread_number_format

    # Initialize the row above the total to an empty df
    row_above_total = df.drop(df.index)
    using_heading_row = False
    only_one_day_row = False

    if not headings_df.empty:
        # Take the closest heading (to the daily total) that's above where the daily total would appear
        heading_row = headings_df[headings_df[HEADING_COLUMN] == category]
        if not heading_row.empty:
            day_below_heading_row = headings_df[
                (headings_df.index == heading_row.index[-1] + 1) & (headings_df[HEADING_COLUMN] == DAY_OR_WEEK_HEADING)]
            if not day_below_heading_row.empty:
                row_above_total = day_below_heading_row.iloc[0]
            else:
                row_above_total = heading_row.iloc[-1]
                using_heading_row = True
        elif rows_with_day_type is not None:
            if rows_with_day_type.shape[0] == 1:
                row_above_total = rows_with_day_type.iloc[0]
                only_one_day_row = True
            else:
                if rows_with_day_type.index[1] == rows_with_day_type.index[0] + 1:
                    row_above_total = rows_with_day_type.iloc[1]
                else:
                    row_above_total = rows_with_day_type.iloc[0]

        if row_above_total.size > 0:
            index_to_start = row_above_total.name + 1
            crop_top = row_above_total['top'] + row_above_total['height']
            if using_heading_row:
                # The heading row is higher than the day row, so when using the heading row as the reference,
                # crop_top should be further down (on the screenshot)
                crop_top += 2 * row_above_total['height']
            elif row_above_total['left'] > (0.15 * screenshot.width):
                crop_top += row_above_total['height']

            # iOS shows the daily total in a larger font size than the headings,
            # so the crop region needs to be about 4x as tall as a heading
            # to increase the chance of the whole daily total falling within the crop region
            crop_bottom = crop_top + (4 * row_above_total['height'])
            if only_one_day_row:
                # The 'overlay' date row (at the top of the screen after scrolling) can sometimes be even further away
                crop_bottom += 2 * row_above_total['height']

        else:
            index_to_start = 0
            # This is a default, in case the screenshot was cropped before uploading,
            # in such a way that the Daily Total value is the first row of text.
            crop_top = 0
            crop_bottom = crop_top + int(screenshot.width / 8)  # Daily Total height is usually < 8x screenshot_width.

        if df.shape[0] > index_to_start + 1:
            loop_number = 0
            for i in df.index[(df.index >= index_to_start)]:
                # print(f"Looking at text: {df.loc[i]['text']}")
                loop_number += 1
                if loop_number > 2 or (not headings_df.empty and i in headings_df.index):
                    # The Daily Total is never more than 2 rows of text away from its heading, and
                    # it's also never on the same line as another heading
                    break
                # The Daily Total is always left aligned, and never too wide,
                # so skip text that starts more than 15% away from the left edge of the screenshot,
                # and skip text that ends more than 80% from the left edge of the screenshot.
                if df.loc[i]['left'] > (0.15 * screenshot.width) or \
                        df.loc[i]['left'] + df.loc[i]['width'] > (0.8 * screenshot.width):
                    continue
                row_text = df.loc[i]['text']
                row_conf = round(df.loc[i]['conf'], 4)  # 4 decimal points of precision for the confidence value

                if len(re.findall(value_pattern, row_text)) == 1:
                    daily_total_1st_scan, daily_total_1st_scan_conf = filter_time_or_number_text(row_text, row_conf, value_pattern)
                    break

            if daily_total_1st_scan_conf == NO_CONF:
                print(f"Total {category} not found on 1st scan.")

    else:
        print("No headings found on first scan.")
        # This is a default, in case the uploaded screenshot was cropped in such a way that the Daily Total value
        # is the first row of text.
        crop_top = 0
        crop_bottom = crop_top + int(screenshot.width / 8)  # Daily Total height is usually < 8x screenshot_width.

    crop_left = 0
    crop_right = int(0.6 * screenshot.width)
    # Right edge of Daily Total is not likely more than 60% away from the left edge of the screenshot

    cropped_image = img[crop_top:crop_bottom, crop_left:crop_right]
    scale = 0.5  # Pytesseract sometimes fails to read very large text; scale down the cropped region
    scaled_cropped_image = cv2.resize(cropped_image, None,
                                      fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    kernel_dim = int(screenshot.width / 500)
    kernel_dim -= 1 if kernel_dim % 2 == 0 else 0
    kernel_size = (kernel_dim, kernel_dim)
    scaled_cropped_image = cv2.GaussianBlur(scaled_cropped_image, kernel_size, 0)

    if category == SCREENTIME:
        _, rescan_df = extract_text_from_image(scaled_cropped_image)
    else:
        _, rescan_df = extract_text_from_image(scaled_cropped_image, cmd_config=r'--oem 3 --psm 6 outputbase digits')

    # For debugging.
    if show_images:
        OCRScript_v3.show_image(rescan_df, scaled_cropped_image)

    # Initialize second scan values
    daily_total_2nd_scan = NO_TEXT
    daily_total_2nd_scan_conf = NO_CONF

    if rescan_df.shape[0] > 0:
        for i in rescan_df.index:
            # Skip rows that are more than 20% away from the left edge of the screenshot.
            if rescan_df.loc[i]['left'] > 0.2 * scale * screenshot.width or (not headings_df.empty and rescan_df.loc[i]['height'] < (min(headings_df['height'])*scale)):
                continue
            row_text = rescan_df.loc[i]['text']
            row_conf = round(rescan_df.loc[i]['conf'], 4)  # 4 decimal points of precision for the confidence value
            # row_text = filter_time_or_number_text(row_text)
            # if 1 <= len(re.findall(value_pattern, row_text, re.IGNORECASE)) <= 2 and rescan_df.loc[i]['height'] > 0.01 * screenshot.height:
            if len(re.findall(value_pattern, row_text)) == 1 and \
                    rescan_df.loc[i]['height'] > 0.01 * screenshot.height and \
                    len(re.findall(r'AM|PM', row_text, re.IGNORECASE)) <= 1:
                # The row text contains a (misread) value, and
                # the height of that value's textbox is above a minimum threshold, and
                # that value is not an 'hours' row (Sometimes after cropping the image, the hours row is included
                # and gets misread as a value format because it may contain AM, which is also a misread form of 4m)
                daily_total_2nd_scan, daily_total_2nd_scan_conf = filter_time_or_number_text(row_text, row_conf, value_pattern)
                daily_total_2nd_scan_conf = row_conf
                break

    if daily_total_1st_scan_conf != NO_CONF:
        print(f"Total {category}, 1st scan: {daily_total_1st_scan} (conf = {daily_total_1st_scan_conf})")
    if daily_total_2nd_scan_conf != NO_CONF:
        print(f"Total {category}, 2nd scan: {daily_total_2nd_scan} (conf = {daily_total_2nd_scan_conf})")

    daily_tot, daily_tot_conf = choose_between_two_values(daily_total_1st_scan, daily_total_1st_scan_conf,
                                                          daily_total_2nd_scan, daily_total_2nd_scan_conf)

    return daily_tot, daily_tot_conf


def convert_text_time_to_minutes(time_as_string):
    """
        For Screentime screenshots, coverts the daily total (String) into a number of minutes (int).
        :return: (int) The daily total time converted to minutes
    """
    if str(time_as_string) == NO_TEXT:
        return NO_NUMBER

    def extract_unit_of_time_as_int(time_str, time_unit_format):
        """
            Finds the substring of time_str that represents a number of units of time (for a given unit) and
            extracts the number of those units from that substring.
            :param time_str: The string to search for a time format
            :param time_unit_format: The abbreviated form of a unit of time, as used by the OS (can be a regex)
            :return: (int) The number of units of time represented by the substring
        """

        time_format_regex = ''.join([r'\d+\s?', time_unit_format])
        extracted_time_as_str = re.search(time_format_regex, time_str)
        if extracted_time_as_str:
            extracted_time_as_str = extracted_time_as_str.group()
            extracted_time_int = int(re.sub(time_unit_format, '', extracted_time_as_str))
            return extracted_time_int
        else:
            return 0

    # Initialize the minutes counter
    total_usage_time_mins = 0

    # See if the time value has seconds in it -- if it does, it won't contain minutes, so the number of minutes is 0
    usage_time_seconds = extract_unit_of_time_as_int(time_as_string, SECONDS_FORMAT)
    if not usage_time_seconds:
        # The time was not in seconds, so look for hours, then minutes, and add them together for a total in minutes
        usage_time_hours = extract_unit_of_time_as_int(time_as_string, HOURS_FORMAT)
        usage_time_hours_to_minutes = (usage_time_hours * 60) if usage_time_hours else 0

        total_usage_time_mins = extract_unit_of_time_as_int(time_as_string, MINUTES_FORMAT)
        total_usage_time_mins += usage_time_hours_to_minutes

    return total_usage_time_mins


def get_total_pickups_2nd_location(screenshot, img):
    """
    Looks for the number of daily total pickups in its alternate location in the iOS dashboard.

    In the iOS dashboard, the number of daily total pickups appears twice—once in big numbers under the
    heading PICKUPS, and once below the bar chart, with the small heading "Total Pickups". Searching for both instances
    increases the chance of finding the correct value.

    :return: The number of daily total pickups as shown in the secondary location (if found).
    """

    headings_df = screenshot.headings_df
    text_df = screenshot.text
    value_format = misread_number_format

    print("\nLooking in 2nd location for total pickups:")
    row_with_first_pickup = headings_df[headings_df[HEADING_COLUMN] == FIRST_PICKUP_HEADING]
    row_with_first_used_after_pickup = headings_df[headings_df[HEADING_COLUMN] ==
                                                   FIRST_USED_AFTER_PICKUP_HEADING]

    if row_with_first_pickup.size > 0 and (row_with_first_used_after_pickup.size > 0 and
                                           row_with_first_used_after_pickup.index[0] - row_with_first_pickup.index[
                                               0] > 1 or
                                           text_df.shape[0] > row_with_first_pickup.index[-1] + 1):
        # Crop the image to the region between 'First pickup, Total Pickups' and 'First used after pickup'.
        row_with_total_pickups = text_df[text_df.index == row_with_first_pickup.index[-1] + 1].iloc[0]
        crop_top = row_with_first_pickup.iloc[-1]['top'] + row_with_first_pickup.iloc[-1]['height']
        crop_bottom = crop_top + (2 * row_with_first_pickup.iloc[-1]['height'])
        # This Daily total is the same height as its heading; 2x heading height is enough to capture the daily total.

    elif row_with_first_used_after_pickup.size > 0 and row_with_first_used_after_pickup.index[0] > 0:
        # Crop the image to a short area above the 'Total Pickups' row.
        row_with_total_pickups = text_df[text_df.index == row_with_first_used_after_pickup.index[0] - 1].iloc[0]
        crop_top = row_with_total_pickups['top']
        crop_bottom = crop_top + int(2.5 * row_with_total_pickups['height'])  # Using 2.5 because
        # 2x might crop in the middle of the Daily Total, and 3x might include too much of the heading above it.
    else:
        print(f"Could not find 2nd location for total pickups.  Returning {NO_NUMBER} (conf = {NO_CONF}).")
        return NO_NUMBER, NO_CONF

    last_word_in_row = str.split(row_with_total_pickups['text'])[-1]
    last_word_in_row_conf = round(row_with_total_pickups['conf'], 4)
    try:
        total_pickups_1st_scan, total_pickups_1st_scan_conf = filter_time_or_number_text(last_word_in_row,
                                                                                         last_word_in_row_conf,
                                                                                         value_format)
        print(f"Total pickups found in 2nd location: {total_pickups_1st_scan} (conf = {total_pickups_1st_scan_conf}).")
    except:
        print("Total pickups in 2nd location not found on first scan.")
        total_pickups_1st_scan = NO_NUMBER
        total_pickups_1st_scan_conf = NO_CONF

    # if total_pickups_1st_scan_conf > conf_limit:
    #     print(f"Conf > {conf_limit}. Keeping 1st scan.")
    #     return total_pickups_1st_scan, total_pickups_1st_scan_conf

    crop_left = int(0.25 * screenshot.width)
    crop_right = screenshot.width
    cropped_image = img[crop_top:crop_bottom, crop_left:crop_right]

    scale_factor = 0.5  # pytesseract sometimes fails to read oversize text.
    scaled_cropped_image = cv2.resize(cropped_image, None,
                                      fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    rescan_words, rescan_df = extract_text_from_image(scaled_cropped_image)

    if show_images:
        OCRScript_v3.show_image(rescan_words, scaled_cropped_image)

    total_pickups_2nd_scan = NO_NUMBER
    total_pickups_2nd_scan_conf = NO_CONF

    if rescan_words.size > 0:
        value_found = re.findall(misread_number_format, rescan_words['text'].iloc[-1])
        if value_found:
            total_pickups_2nd_scan = value_found[-1]
            total_pickups_2nd_scan_conf = round(rescan_df['conf'][0], 4)
            print(f"Total pickups found in 2nd location (rescan): {total_pickups_2nd_scan} "
                  f"(conf = {total_pickups_2nd_scan_conf}).")
        else:
            print("Total pickups not found in 2nd location (rescan).")
            total_pickups_2nd_scan = NO_NUMBER
            total_pickups_2nd_scan_conf = NO_CONF

    total, total_conf = choose_between_two_values(total_pickups_1st_scan, total_pickups_1st_scan_conf,
                                                  total_pickups_2nd_scan, total_pickups_2nd_scan_conf,
                                                  value_is_number=True)

    print(f"Total pickups, 2nd location: {total} (conf = {total_conf}).")
    return total, total_conf


def crop_image_to_app_area(screenshot, heading_above, heading_below):
    # Determine the region of the screenshot that (likely) contains the list of the top n apps.
    # Initialize the crop region -- the 'for' loop below trims it further

    headings_df = screenshot.headings_df
    text_df = screenshot.text

    crop_top = 0
    crop_bottom = screenshot.height
    crop_left = round(0.15 * screenshot.width)  # The app icons are typically within the leftmost 15% of the screenshot
    crop_right = round(
        0.87 * screenshot.width)  # Symbols (arrows, hourglass) typically appear in the rightmost 87% of the screenshot

    buffer = 18  # A small number of pixels to expand the left edge of the crop rectangle, to increase the chance
    # the crop edge is left of the app names
    if not headings_df.empty and not (headings_df[headings_df['left'] > 0]).empty:
        leftmost_heading_index = headings_df[headings_df['left'] > 0]['left'].idxmin()
        crop_left = min(headings_df['left']) + round(0.09 * screenshot.width) if headings_df[HEADING_COLUMN][
                                                                                             leftmost_heading_index] not in [
                                                                                             DAY_OR_WEEK_HEADING,
                                                                                             HOURS_AXIS_HEADING] else crop_left
        crop_left = round(0.15 * screenshot.width) if crop_left > round(0.2 * screenshot.width) else crop_left
        # The above line corrects for when left-aligned headings are not found (e.g., Date heading, or a partial
        # 'hours' row (i.e. it thinks the left edge of the 'hours' row is further right than it actually is)).
        for i in headings_df.index:
            if headings_df[HEADING_COLUMN][i] == heading_above or \
                    headings_df[HEADING_COLUMN][i] in [DAY_OR_WEEK_HEADING, DATE_HEADING] and crop_bottom == screenshot.height:
                crop_top = headings_df['top'][i] + headings_df['height'][i]
            elif headings_df[HEADING_COLUMN][i] == heading_below:
                crop_bottom = headings_df['top'][i]

    if (headings_df.empty or
            headings_df.iloc[-1][HEADING_COLUMN] in [SCREENTIME_HEADING, LIMITS_HEADING] or
            screenshot.time_period == WEEK or
            crop_top == 0 or
            screenshot.category_submitted == SCREENTIME and ~headings_df[HEADING_COLUMN].str.contains(MOST_USED_HEADING).any() and
            headings_df[HEADING_COLUMN].str.contains(FIRST_USED_AFTER_PICKUP_HEADING).any()):
        # No relevant app data to extract if:
        # there are no headings found, or
        # the last heading found is 'SCREENTIME' or 'LIMITS', or
        # the screenshot contains week-level data, or
        # the screenshot is for screentime and
        #   the 'MOST_USED' heading was not found (the heading for the screentime apps) and
        #   the 'FIRST_USED_AFTER_PICKUP' heading was found (the heading for pickups apps)
        print(f"Heading above app list was not found.")
        return screenshot.grey_image, None, None, None, None
        # app_area_heading_not_found = True
        # num_missed_app_values = 0

    # Crop the image and apply a different monochrome threshold (improves chances of catching missed text)
    cropped_grey_image = screenshot.grey_image[crop_top:crop_bottom, crop_left:crop_right]
    # Create a new monochrome image with a different threshold to ensure the bars below the app names show up well.
    if screenshot.is_light_mode:
        _, cropped_filtered_image = cv2.threshold(cropped_grey_image, 220, 255, cv2.THRESH_BINARY)
    else:
        _, cropped_filtered_image = cv2.threshold(cropped_grey_image, 50, 180, cv2.THRESH_BINARY)

    # cropped_filtered_image, c_top, c_left, c_bottom, c_right = crop_app_icons(cropped_filtered_image)

    return cropped_filtered_image, crop_top, crop_left, crop_bottom, crop_right


def erase_bars_below_app_names(screenshot, df, image):
    # Screentime and Notifications images have grey bars below the app names; the length of the bars
    # are proportional to the time/count for the app. This code erases those shapes to make the numbers
    # easier to read.
    # (Pickups images have the same bars, but they are coloured, so the filtering steps in
    # load_and_process_image removes these.)
    background_colour = WHITE if screenshot.is_light_mode else BLACK
    top_left_coordinates = []

    def find_and_erase_bar_and_icon(r, c, h_max, erase_icon=True):
        top_of_bar = 0
        bottom_of_bar = image_height
        left_of_bar = 0
        right_of_bar = 0
        # debug
        # Iterate through rows starting from start_row
        for row in range(r, min([image_height, r + h_max])):
            # Find the top of the bar
            if image[row, c] != image[r, c]:
                top_of_bar = row - 2
                break
        if erase_icon:
            cv2.rectangle(image, (0, top - int(0.5 * height)), (left - 2, bottom + h_max), background_colour, -1)
        if top_of_bar == 0 or image_height - top_of_bar < 0.01 * image_height:
            return
            # Find the bottom of the bar
        for row in range(top_of_bar + 2, min([image_height, top_of_bar + 2 + height])):
            if top_of_bar > 0 and image[row, c] == image[r, c]:
                bottom_of_bar = row + 2
                break
        bottom_of_bar = top_of_bar + h_max if bottom_of_bar == image_height else bottom_of_bar
        for col in range(c, image_width):
            # Find the right of the bar
            col_pixels = image[top_of_bar:bottom_of_bar, col]
            if np.all(col_pixels == col_pixels[0]):
                right_of_bar = col + int(0.01 * image_width)
                break
        for col in range(c, 0, -1):
            col_pixels = image[top_of_bar:bottom_of_bar, col]
            if np.all(col_pixels == col_pixels[0]):
                left_of_bar = col - int(0.01 * image_width)
                break

        cv2.rectangle(image, (left_of_bar, top_of_bar), (right_of_bar, bottom_of_bar),
                      background_colour, -1)
        top_left_coordinates.append([left_of_bar, top_of_bar, bottom_of_bar - top_of_bar])
        return

    if show_images:
        OCRScript_v3.show_image(df, image)
    image_height = image.shape[0]
    image_width = image.shape[1]
    for i in df.index:
        row_text, top, left, bottom, height = df['text'][i],   df['top'][i],   df['left'][i],   df['top'][i] + df['height'][i],   df['height'][i]
        prev_bottom = df['top'][i - 1] + df['height'][i - 1] if i > 0 else 0
        if i == 0 and df.shape[0] > 1 and max([df['left'][0], df['left'][1]]) < 0.1 * image_width and \
                abs(df['top'][1] - bottom) < 1.5*df['height'][1] and \
                not re.search(misread_time_or_number_format, df['text'][1]):
            cv2.rectangle(image, (0, top), (image_width, bottom), background_colour, -1)
            continue
        elif i > 0 and (top < prev_bottom + 0.01 * image_height or
                        left > 0.1 * image_width or
                        re.match(time_or_number_format, row_text)):  # or image_height - (top + df['height'][i]) < 0.02 *
            # Skip row if either:
            #   the textbox is too close (vertically) to the previous textbox, or
            #   the textbox is too far from the left edge of the cropped image, or
            #   the text in the textbox is a time/number
            continue
        start_row = bottom + round(0.01 * image_width)
        start_col = left + round(0.01 * image_width)
        max_height = min([image_height, bottom + int(df['height'][i]*1.5)])
        if start_row < max_height:
            find_and_erase_bar_and_icon(start_row, start_col, h_max=int(1.5*height))

    # Sometimes pytesseract doesn't read an app name, but we'd still like to have the bar below it erased.
    # This section determines the pixel spacing between two successive bars and, if any of the gaps between found bars
    # is large enough, it seeks out the missed bar and erases it.
    median_bar_height = int(np.median([coord[2] for coord in top_left_coordinates]))
    # print("The top left coordinates are")
    # print(top_left_coordinates)
    average_left = sum(coord[0] for coord in top_left_coordinates) / len(top_left_coordinates)
    filtered_coordinates = [coord for coord in top_left_coordinates if abs(coord[0] - average_left) <= 0.01*image_width]
    # print("The filtered left coordinates are:")
    # print(filtered_coordinates)
    top_coords = [coord[1] for coord in filtered_coordinates]
    differences = [abs(top_coords[i] - top_coords[i - 1]) for i in range(1, len(top_coords))]
    smallest_difference = min(differences)
    # print(f"of the differences in {differences} the smallest is {smallest_difference}")
    prev_top = -1
    for i, top_left in enumerate(filtered_coordinates):
        # print(f"top_left[1] = {top_left[1]}    smallest_difference = {smallest_difference}   prev_top = {prev_top}   0.01*image_width = {int(0.01*image_width)}   median bar height = {median_bar_height}")
        if (i == 0 and top_left[1] - smallest_difference > 0) or \
                (i > 0 and top_left[1] - prev_top > 1.5*smallest_difference):
            find_and_erase_bar_and_icon(top_left[1] - smallest_difference - int(0.005*image_height),
                                        top_left[0] + int(0.015*image_width),
                                        median_bar_height,
                                        erase_icon=False)
        prev_top = top_left[1]

    return image


def consolidate_overlapping_text(df):
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

        current_textbox = Rectangle(current_left, current_top, current_right, current_bottom)
        prev_textbox = Rectangle(prev_left, prev_top, prev_right, prev_bottom)

        current_num_digits = len(re.findall(r'\d', df['text'][i]))
        prev_num_digits = len(re.findall(r'\d', df['text'][i - 1]))

        if calculate_overlap(current_textbox, prev_textbox) > 0.3:
            # If two text boxes overlap by at least 30%, consider them to be two readings of the same text.
            # print(f"Comparing {df['text'][i]} to {df['text'][i - 1]}:")
            if re.search(misread_time_format, df.loc[i, 'text']) and not re.search(misread_time_format, df.loc[i - 1, 'text']):
                # print('debug 1')
                rows_to_drop.append(i - 1)
            elif not re.search(misread_time_format, df.loc[i, 'text']) and re.search(misread_time_format, df.loc[i - 1, 'text']):
                # print("debug 2")
                rows_to_drop.append(i)
            elif current_num_digits > prev_num_digits:
                # print('debug 3')
                rows_to_drop.append(i - 1)
            elif current_num_digits < prev_num_digits:
                # print("debug 4")
                rows_to_drop.append(i)
            elif len(str(df['text'][i - 1])) <= 2 < len(str(df['text'][i])):
                rows_to_drop.append(i - 1)
            elif len(str(df['text'][i])) <= 2 < len(str(df['text'][i - 1])):
                rows_to_drop.append(i)
            elif df['conf'][i - 1] > df['conf'][i]:
                if df['text'][i - 1] == 'min':
                    # print('debug 5')
                    rows_to_drop.append(i - 1)
                else:
                    # print('debug 6')
                    rows_to_drop.append(i)
            else:
                if df['text'][i] == 'min':
                    # print('debug 7')
                    rows_to_drop.append(i)
                else:
                    # print('debug 8')
                    rows_to_drop.append(i - 1)

    if 'level_0' in df.columns:
        df.drop(columns=['level_0'], inplace=True)

    merged_df = df.drop(index=rows_to_drop).reset_index()

    # consolidated_df = merge_df_rows_by_height(merged_df)

    return merged_df


def get_app_names_and_numbers(screenshot, df, category, max_apps):
    app_names = pd.DataFrame(columns=['name', 'name_conf'])
    app_numbers = pd.DataFrame(columns=['number', 'number_conf'])
    empty_name_row = pd.DataFrame({'name': [NO_TEXT], 'name_conf': [NO_CONF]})
    empty_number_row = pd.DataFrame({'number': [NO_TEXT], 'number_conf': [NO_CONF]}) if category == SCREENTIME else (
                       pd.DataFrame({'number': [NO_NUMBER], 'number_conf': [NO_CONF]}))

    with (warnings.catch_warnings()):
        warnings.simplefilter('ignore')
        
        # This section determines whether each row in the final app info df is an app name or a number/time
        # and separates them.
        regex_format = misread_time_format if category == SCREENTIME else misread_number_format
        prev_row_type = ''
        prev_app_top = -1
        prev_app_height = -1
        prev_row_height = -1
        num_missed_app_values = 0
        for i in df.index:
            row_text = df['text'][i]
            row_conf = round(df['conf'][i], 4)
            row_height = df['height'][i]
            if (len(str(row_text)) >= 3 or re.match(r'[xX]{1,2}', str(row_text))) and \
                    not re.match(regex_format, str(row_text), re.IGNORECASE) and \
                    row_height > 0.75 * df['height'].mean():  # if current row text is app name
                if prev_row_type == 'name':  # two app names in a row
                    if len(app_names) <= max_apps:
                        num_missed_app_values += 1
                    app_numbers = pd.concat([app_numbers, empty_number_row], ignore_index=True)
                elif prev_row_type == 'number' and prev_app_top >= 0 and \
                        df['top'][i] - prev_row_height > 3*np.median([row_height, prev_app_height]):

                    print(f"{df['top'][i]} - {prev_app_top} > 4*max({row_height}, {prev_app_height}) ?")
                    print(f"Suspected missing app between '{prev_app_name}' and '{row_text}'. Adding a blank row.")
                    app_names = pd.concat([app_names, empty_name_row], ignore_index=True)
                    app_numbers = pd.concat([app_numbers, empty_number_row], ignore_index=True)
                new_name_row = pd.DataFrame({'name': [row_text], 'name_conf': [row_conf]})
                app_names = pd.concat([app_names, new_name_row], ignore_index=True)
                prev_row_type = 'name'
                prev_app_name = row_text
                prev_app_height = row_height
            elif (category == SCREENTIME and re.match(misread_time_format, str(row_text), re.IGNORECASE) or
                  category != SCREENTIME and re.match(misread_number_format, str(row_text), re.IGNORECASE)):
                # if current row text is number
                row_text, row_conf = filter_time_or_number_text(row_text, row_conf, f=regex_format)
                if prev_row_type != 'name':  # two app numbers in a row, or first datum is a number
                    if len(app_names) < max_apps:
                        num_missed_app_values += 1
                    app_names = pd.concat([app_names, empty_name_row], ignore_index=True)
                new_number_row = pd.DataFrame({'number': [row_text], 'number_conf': [row_conf]})
                app_numbers = pd.concat([app_numbers, new_number_row], ignore_index=True)
                prev_row_type = 'number'
            else:  # row is neither a valid app name nor a number, so discard it
                pass
            prev_row_height = row_height
        # Making sure each list is the right length (fill any missing values with NO_TEXT/NO_NUMBER and NO_CONF)
        while app_names.shape[0] < max_apps:
            app_names = pd.concat([app_names, empty_name_row], ignore_index=True)
        while app_numbers.shape[0] < max_apps:
            app_numbers = pd.concat([app_numbers, empty_number_row], ignore_index=True)

        if category == SCREENTIME and app_numbers['number'][0] == screenshot.daily_total and app_numbers['number'][
            0] != NO_NUMBER:
            print(f"Daily total {category} matches first app time: {app_numbers['number'][0]}")
            print(f"Resetting daily total {category} to {NO_NUMBER}.")
            screenshot.set_daily_total(NO_TEXT, NO_CONF)
            screenshot.set_daily_total_minutes(NO_NUMBER)

    # Sometimes tesseract misreads (Italian) "Foto" as "mele"/"melee" or misreads ".AI" as ".Al"
    app_names['name'] = app_names['name'].replace({r'^melee$|^mele$': 'Foto'}, regex=True)
    app_names['name'] = app_names['name'].replace({r'.Al': '.AI'}, regex=True)
    
    top_n_app_names_and_numbers = pd.concat(
        [app_names.head(max_apps), app_numbers.head(max_apps)], axis=1)

    return top_n_app_names_and_numbers


def main():
    print("I am now in iOSFunctions.py")
