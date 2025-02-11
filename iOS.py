"""This file contains iOS-specific dictionaries, functions, and variables."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import re
import warnings
from RuntimeValues import *

"""
    Language abbreviations
"""
GER = 'German'
ITA = 'Italian'
ENG = 'English'
FRA = 'French'


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

# Variables for iOS time values
# Even though the words for 'hours', 'minutes', and 'seconds' differ by language, iOS uses h/min/m/s for all languages.
MIN_KEYWORD = r'min|m'
HOUR_KEYWORD = 'h'
SECONDS_KEYWORD = 's'
TIME_FORMAT_STR_FOR_MINUTES = r'\d+\s?(min|m)'
TIME_FORMAT_STR_FOR_HOURS = r'\d+\s?h'
TIME_FORMAT_STR_FOR_SECONDS = r'\d+\s?s'

HEADING_COLUMN = 'heading'
SCREENTIME_HEADING = 'screentime'
LIMITS_HEADING = 'limits'
MOST_USED_HEADING = 'most used'
PICKUPS_HEADING = 'pickups'
FIRST_PICKUP_HEADING = 'first pickup'
FIRST_USED_AFTER_PICKUP_HEADING = 'first used after pickup'
NOTIFICATIONS_HEADING = 'notifications'
HOURS_AXIS_HEADING = 'hours row'
DAY_OR_WEEK_HEADING = 'day or week'

TODAY = 'today'
YESTERDAY = 'yesterday'
DAY_OF_THE_WEEK = 'weekday'
WEEK = 'week'

misread_time_format = r'^[\d|t]+\s?[hn]$|^[\d|t]+\s?[hn]\s?[\d|tA]+\s?(min|m)$|^.{0,2}\s?[0-9AIt]+\s?(min|m)$|\d+\s?s$'
misread_number_format = r'^[0-9A]+$'
misread_time_or_number_format = '|'.join([misread_time_format, misread_number_format])

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
    moe_weekday = round(np.log(max((len(string) for string in KEYWORDS_FOR_DAYS_OF_THE_WEEK[lang]))))
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
                          for key in KEYWORDS_FOR_DAYS_OF_THE_WEEK[lang])) <= moe_weekday) &
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

    return df


def get_dashboard_category(heads_df):
    # Headings in iOS Dashboard appear in this order:
    #     SCREEN TIME,  (LIMITS),  MOST USED,  PICKUPS,  FIRST PICKUP,  FIRST USED AFTER PICKUP,  NOTIFICATIONS
    # This function checks for the occurrence of these headings in that order.
    categories_found = []
    backup_category = image_response_type

    if heads_df[HEADING_COLUMN].str.contains(SCREENTIME_HEADING).any() or \
            (heads_df[HEADING_COLUMN].str.contains(LIMITS_HEADING).any() and
             heads_df[HEADING_COLUMN].str.contains(HOURS_AXIS_HEADING).any() and
             heads_df[heads_df[HEADING_COLUMN] == LIMITS_HEADING].index[0] >
             heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].index[0]) or \
            heads_df[HEADING_COLUMN].str.contains(MOST_USED_HEADING).any() and (
            text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].index[0] + 1 or
            heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].iloc[-1]['top'] <
            0.9 * screenshot_height):
        # Found screentime heading; or
        # Found limits heading and hours row, and limits is below hours; or
        # Found most used heading and either:
        #     text_df has more data below the most used heading, or
        #     the most used heading is not too close to the bottom
        categories_found.append(SCREENTIME)
        # return SCREENTIME

    if heads_df[HEADING_COLUMN].str.contains(PICKUPS_HEADING).any() or \
            heads_df[HEADING_COLUMN].str.contains(FIRST_USED_AFTER_PICKUP_HEADING).any() and (
            text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == FIRST_USED_AFTER_PICKUP_HEADING].index[0] + 1 or
            heads_df[heads_df[HEADING_COLUMN] == FIRST_USED_AFTER_PICKUP_HEADING].iloc[-1]['top'] <
            0.9 * screenshot_height):
        # Found pickups heading; or
        # Found "first used after pickup" heading and either:
        #     there's more data below, or
        #     the "first used after pickup" heading is not close to the bottom
        categories_found.append(PICKUPS)
        # return PICKUPS

    if heads_df[HEADING_COLUMN].str.contains(NOTIFICATIONS_HEADING).any() or \
            heads_df[HEADING_COLUMN].str.contains(HOURS_AXIS_HEADING).any() and (
            not heads_df[HEADING_COLUMN].str.contains(MOST_USED_HEADING).any() or
            heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].index[0] > heads_df[heads_df[HEADING_COLUMN] == MOST_USED_HEADING].index[0]) and (
            text_df.shape[0] > heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].index[0] + 1 or
            heads_df[heads_df[HEADING_COLUMN] == HOURS_AXIS_HEADING].iloc[-1]['top'] <
            0.9 * screenshot_height):
        # Found notifications heading; or
        # Found hours row and either:
        #     there's more data below, or
        #     the hours row is not close to the bottom
        categories_found.append(NOTIFICATIONS)
        # return NOTIFICATIONS

    if not categories_found:
        print(f"Category not detected.  Defaulting to submitted category: {backup_category}")
        # Returning None to flag the screenshot for manual review;
        # screenshot category is set to backup_category after this function is called
        return None
    else:
        print(f"Categories detected: {categories_found}")
        if backup_category in categories_found:
            print(f"Setting category to: {backup_category}")
            return backup_category
        else:
            print(f"Screenshot submitted under {backup_category} category, but found {categories_found[0]} instead. "
                  f"Setting category to {categories_found[0]}.")
            return categories_found[0]


def convert_text_time_to_minutes(text):
    if str(text) == NO_TEXT:
        return NO_NUMBER

    def extract_unit_of_time_as_int(time_str, time_format_regex=None, time_key_word=None):
        # Extract the time and return it as an integer.
        extracted_time_as_str = re.search(time_format_regex, time_str)
        if extracted_time_as_str:
            extracted_time_as_str = extracted_time_as_str.group()
            extracted_time_int = int(re.sub(time_key_word, '', extracted_time_as_str))
            return extracted_time_int
        return 0

    total_usage_time_mins = 0
    usage_time_seconds = extract_unit_of_time_as_int(text, time_format_regex=TIME_FORMAT_STR_FOR_SECONDS,
                                                     time_key_word=SECONDS_KEYWORD)
    if not usage_time_seconds:
        usage_time_hours = (extract_unit_of_time_as_int(text, time_format_regex=TIME_FORMAT_STR_FOR_HOURS,
                                                        time_key_word=HOUR_KEYWORD))
        usage_time_hours_to_minutes = (usage_time_hours * 60) if usage_time_hours else 0
        total_usage_time_mins = extract_unit_of_time_as_int(text, time_format_regex=TIME_FORMAT_STR_FOR_MINUTES,
                                                            time_key_word=MIN_KEYWORD)
        total_usage_time_mins += usage_time_hours_to_minutes

    return total_usage_time_mins


def filter_time_or_number_text(text):
    if str(text) == NO_TEXT:
        return NO_TEXT

    # Replace common misread characters (e.g. pytesseract sometimes misreads '1h' as 'Th'/'th').
    text = re.sub(r'[TtIi](?=.?[hm])', '1', text)  # gets used
    text = re.sub('ah', '4h', text)  # gets used
    text = re.sub(r'(Os|O s)', '0s', text)  # gets used
    text = re.sub('A', '4', text)  # gets used

    # Remove any characters that aren't a digit or a 'time' character ('h' = hours, 'min' = minutes, 's' = seconds)
    text = re.sub(r'[^0-9hmins]', '', text)

    return text


def get_daily_total_and_confidence(df, heading):
    print(f"\nSearching for total {heading}:")

    # Initialize first scan values
    daily_total_1st_scan = ''
    daily_total_1st_scan_conf = NO_CONF

    number_pattern = r'^\d+h|^\d+m|\d+(m|min)$|^\d+s$' if heading == SCREENTIME_HEADING else r'^\d+$'

    # Initialize the row above the total to an empty df
    row_above_total = df.drop(df.index)
    using_heading_row = False
    only_one_day_row = False

    if not headings_df.empty:
        # Take the closest heading (to the daily total) that's above where the daily total would appear
        heading_row = headings_df[headings_df[HEADING_COLUMN] == heading]
        if not heading_row.empty:
            day_below_heading_row = headings_df[
                (headings_df.index == heading_row.index[-1] + 1) & (headings_df[HEADING_COLUMN] == DAY_OR_WEEK_HEADING)]
            if not day_below_heading_row.empty:
                row_above_total = day_below_heading_row.iloc[0]
            else:
                row_above_total = heading_row.iloc[-1]
                using_heading_row = True
        elif rows_with_day_or_week is not None:
            if rows_with_day_or_week.shape[0] == 1:
                row_above_total = rows_with_day_or_week.iloc[0]
                only_one_day_row = True
            else:
                if rows_with_day_or_week.index[1] == rows_with_day_or_week.index[0] + 1:
                    row_above_total = rows_with_day_or_week.iloc[1]
                else:
                    row_above_total = rows_with_day_or_week.iloc[0]

        if row_above_total.size > 0:
            index_to_start = row_above_total.name + 1
            crop_top = row_above_total['top'] + row_above_total['height']
            if using_heading_row:
                # The heading row is higher than the day row, so when using the heading row as the reference,
                # crop_top should be further down (on the screenshot)
                crop_top += 2 * row_above_total['height']
            elif row_above_total['left'] > (0.15 * screenshot_width):
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
            crop_bottom = crop_top + int(screenshot_width / 8)  # Daily Total height is usually < 8x screenshot_width.

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
                # so skip text that starts more than 20% away from the left edge of the screenshot,
                # and skip text that ends more than 80% from the left edge of the screenshot.
                if df.loc[i]['left'] > (0.2 * screenshot_width) or \
                        df.loc[i]['left'] + df.loc[i]['width'] > (0.8 * screenshot_width):
                    continue
                row_text = df.loc[i]['text']
                row_conf = round(df.loc[i]['conf'], 4)  # 4 decimal points of precision for the confidence value
                row_text = filter_time_or_number_text(row_text)
                if re.search(number_pattern, row_text):
                    daily_total_1st_scan = row_text
                    daily_total_1st_scan_conf = row_conf
                    break

            if daily_total_1st_scan_conf == NO_CONF:
                print(f"Total {heading} not found on 1st scan.")

    else:
        print("No headings found on first scan.")
        # This is a default, in case the uploaded screenshot was cropped in such a way that the Daily Total value
        # is the first row of text.
        crop_top = 0
        crop_bottom = crop_top + int(screenshot_width / 8)  # Daily Total height is usually < 8x screenshot_width.

    crop_left = 0
    crop_right = int(0.6 * screenshot_width)
    # Right edge of Daily Total is not likely more than 60% away from the left edge of the screenshot

    cropped_image = filtered_image[crop_top:crop_bottom, crop_left:crop_right]
    scale = 0.5  # Pytesseract sometimes fails to read very large text; scale down the cropped region
    scaled_cropped_image = cv2.resize(cropped_image, None,
                                      fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if heading == SCREENTIME_HEADING:
        _, rescan_df = extract_text_from_image(scaled_cropped_image)
    else:
        _, rescan_df = extract_text_from_image(scaled_cropped_image, cmd_config=r'--oem 3 --psm 6 outputbase digits')

    # For debugging.
    if show_images:
        show_text_on_image(rescan_df, scaled_cropped_image)

    # Initialize second scan values
    daily_total_2nd_scan = NO_TEXT
    daily_total_2nd_scan_conf = NO_CONF

    if rescan_df.shape[0] > 0:
        for i in rescan_df.index:
            # Skip rows that are more than 20% away from the left edge of the screenshot.
            if rescan_df.loc[i]['left'] > 0.2 * scale_factor * screenshot_width:
                continue
            row_text = rescan_df.loc[i]['text']
            row_conf = round(rescan_df.loc[i]['conf'], 4)  # 4 decimal points of precision for the confidence value
            row_text = filter_time_or_number_text(row_text)
            if 1 <= len(re.findall(number_pattern, row_text)) <= 2 and rescan_df.loc[i]['height'] > 0.01 * screenshot_height:
                # TODO: can this be just a re.search? number_pattern has ^ and $ in it, so there shouldn't be > 1.
                daily_total_2nd_scan = row_text
                daily_total_2nd_scan_conf = row_conf
                break

    if daily_total_1st_scan_conf != NO_CONF:
        print(f"Total {heading}, 1st scan: {daily_total_1st_scan} (conf = {daily_total_1st_scan_conf})")
    if daily_total_2nd_scan_conf != NO_CONF:
        print(f"Total {heading}, 2nd scan: {daily_total_2nd_scan} (conf = {daily_total_2nd_scan_conf})")

    daily_total, daily_total_conf = choose_between_two_values(daily_total_1st_scan, daily_total_1st_scan_conf,
                                                              daily_total_2nd_scan, daily_total_2nd_scan_conf)

    return daily_total, daily_total_conf


def get_total_pickups_2nd_location():
    # The Daily Total for pickups can appear twice on the screenshot; once in the top left corner of the Pickups block,
    # and once near the bottom of the Pickups block (under the 'First Pickup, Total Pickups' row).
    # Look in this 2nd place as well.

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

    try:
        total_pickups_1st_scan = filter_time_or_number_text(str.split(row_with_total_pickups['text'])[-1])
        total_pickups_1st_scan_conf = round(row_with_total_pickups['conf'], 4)  # 4 decimal points of precision
        print(f"Total pickups, 1st rescan: {total_pickups_1st_scan} (conf = {total_pickups_1st_scan_conf}).")
    except:
        print("Total pickups not found on 1st rescan.")
        total_pickups_1st_scan = ''
        total_pickups_1st_scan_conf = NO_CONF

    if total_pickups_1st_scan_conf > conf_limit:
        print(f"Conf > {conf_limit}. Keeping 1st rescan.")
        return total_pickups_1st_scan, total_pickups_1st_scan_conf

    crop_left = int(0.25 * screenshot_width)
    crop_right = screenshot_width
    cropped_image = filtered_image[crop_top:crop_bottom, crop_left:crop_right]

    scaled_cropped_image = cv2.resize(cropped_image, None,
                                      fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    _, rescan_df = extract_text_from_image(scaled_cropped_image)

    if show_images:
        show_text_on_image(rescan_df, scaled_cropped_image)

    total_pickups_2nd_scan = ''
    total_pickups_2nd_scan_conf = NO_CONF

    if rescan_df.size > 0:
        total_pickups_2nd_scan = rescan_df['text'][0]
        total_pickups_2nd_scan_conf = round(rescan_df['conf'][0], 4)
        print(f"Total pickups, 2nd rescan: {total_pickups_2nd_scan} (conf = {total_pickups_2nd_scan_conf}).")

    total, total_conf = choose_between_two_values(total_pickups_1st_scan, total_pickups_1st_scan_conf,
                                                  total_pickups_2nd_scan, total_pickups_2nd_scan_conf)

    print(f"Total pickups, 2nd location: {total} (conf = {total_conf}).")
    return total, total_conf


def erase_bars_below_app_names(df, image):
    # Screentime and Notifications images have grey bars below the app names; the length of the bars
    # are proportional to the time/count for the app. This code erases those shapes to make the numbers
    # easier to read.
    # (Pickups images have the same bars, but they are coloured, so the filtering steps in
    # load_and_process_image removes these.)

    for i in df.index:
        # Black bars only exist below rows with app names, which are always near the left edge in the cropped screenshot
        # Skip rows if they aren't near the left edge or if they match a time/number format.
        if i > 0 and (df['top'][i] < df['top'][i - 1] + df['height'][i - 1] + 0.02 * screenshot_width or
                      df['left'][i] > 0.1 * screenshot_width or
                      re.match(time_or_number_format, df['text'][i])):
            continue

        start_row = df['top'][i] + df['height'][i] + round(0.01 * screenshot_width)
        start_col = df['left'][i] + round(0.01 * screenshot_width)

        top_of_bar = 0
        bottom_of_bar = image.shape[0]
        left_of_bar = 0
        right_of_bar = 0
        # Iterate through rows starting from start_row
        for row in range(start_row, image.shape[0]):
            # Find the top of the bar
            if top_of_bar == 0 and image[row, start_col] != image[row - 1, start_col]:
                top_of_bar = row - 2
                continue
            # Find the bottom of the bar
            elif top_of_bar > 0 and image[row, start_col] != image[row - 1, start_col]:
                bottom_of_bar = row + 2
                break
            else:
                continue
        top_of_bar = bottom_of_bar - 2 if top_of_bar == 0 else top_of_bar
        middle_of_bar = round(0.5 * (bottom_of_bar + top_of_bar))

        for col in range(start_col, image.shape[1]):
            # Find the right of the bar
            if image[middle_of_bar, col] != image[middle_of_bar, col - 1]:
                right_of_bar = col + 5
                break
            else:
                continue
            # The bars always extend to the left edge of the cropped screenshot,
            # so we'll just paint all the way to the left edge.

        box_color_to_paint = (255, 255, 255) if is_lightmode else (0, 0, 0)
        cv2.rectangle(image, (left_of_bar, top_of_bar), (right_of_bar, bottom_of_bar),
                      box_color_to_paint, -1)

    return image



def main():
    print("I am now in iOS.py")