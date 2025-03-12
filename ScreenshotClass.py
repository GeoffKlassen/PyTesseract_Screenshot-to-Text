"""
    The Screenshot class will be used to store the raw image and the extracted data from a screenshot,
    as well as metadata about the screenshot as taken from the database of URLs.
"""
import pandas as pd
import RuntimeValues
from ConvenienceVariables import *


def initialize_data_row():
    """
    Initialize a data row for the Screenshot object.
    Columns include:
        image_url
        participant_id
        device_id
        language
        device_os
        android_version
        date_submitted
        date_detected
        relative_day
        category_submitted
        category_detected
        daily total
        For the top 'n' apps, as requested from the study:
            app_n_name
            app_n_number
        hashed
        review_count

    :return: The initialized data row, with all the necessary columns.
    """

    df = pd.DataFrame(columns=[IMAGE_URL, PARTICIPANT_ID, DEVICE_ID, LANGUAGE,
                               DEVICE_OS, ANDROID_VERSION,
                               DATE_SUBMITTED, DATE_DETECTED, RELATIVE_DAY,
                               CATEGORY_SUBMITTED, CATEGORY_DETECTED])
    df[DAILY_TOTAL] = None
    for i in range(1, RuntimeValues.max_apps_per_category + 1):
        df[f'{APP}_{i}_{NAME}'] = None
        df[f'{APP}_{i}_{NUMBER}'] = None
    df[HASHED] = None
    df[REVIEW_COUNT] = None
    return df


class Screenshot:

    def __init__(self, participant, url=None, device_id=None, device_os=None, date=None, category=None):
        """Initialize an object of type Screenshot
            :param participant:     the participant (object) that submitted this screenshot (has variable user_id)
            :param url:             the full https://.../image.jpg URL as taken from the database of URLs

        Other variables:
            filename:               the filename portion of the url above (or the local name of the file)

            user_id:                the Avicenna User ID for the participant who submitted the given image

            device_id:              the Device ID of the phone/tablet/etc. as stored in the URL dataframe

            grey_image:             the np array of the image, converted to greyscale

            height:                 the height (in pixels) of the grey image

            width:                  the width (in pixels) of the grey image

            is_light_mode:          a binary value of whether the image has an overall light or dark background

            text:                   a dataframe of the lines of text taken during the initial extraction of data
                                        (contains columns 'left', 'top', 'width', 'height', 'text', 'conf')

            words_df:               a dataframe of the individual words of text taken during the initial extraction of data

            text_hash:              a hashed string of all the text extracted from the initial scan
                                        (used for determining if two submitted images contain the same data)

            language:               the language detected in the image, based on keywords defined in ConvenienceVariables.py

            date_format:            Regex for the date format to look for, incorporating the month abbreviations for the
                                        image's language (e.g. Jan 23, 12 Feb, March 31 (English); 18 avr (French))

            device_os_submitted:    the Operating System of the phone/tablet/etc. as determined by the device_id

            device_os_detected:     the Operating System of the phone/tablet/etc. as detected by examining the image

            android_version:        the version of the Android Dashboard detected by examining the image
                                        (i.e. VERSION_2018, SAMSUNG_2021, SAMSUNG_2024, GOOGLE)

            time_format_short,      Regex for the short format of a time to look for, e.g. 1 hr 23 min (Android only)

            time_format_long,       Regex for the long format of a time to look for, e.g. 1 hour 23 minutes (Android only)

            time_format_eol:        Regex for the short format of a time that must occur at the end of a line (Android only)

            date_submitted:         the date the image was submitted, as stored in the URL dataframe

            date_detected:          the date detected in the image

            category_submitted:     the category under which the image was submitted
                                        (i.e. SCREENTIME, PICKUPS/UNLOCKS, NOTIFICATIONS)

            category_detected:      the category of data in the image, as detected by examining the text in the image

            relative_day:           the day text extracted from the image (e.g. "today", "yesterday", "weekday", "week")
                                        (Note: this variable may sometimes be 'week' to reflect that Dashboards can show weekly data)

            rows_with_day_type:     a subset dataframe of text that contains only the rows that match a day_type

            rows_with_date:         a subset dataframe of text that contains only the rows that match a date format

            headings_df:            a subset dataframe of text that contains only the rows that match a heading
                                        (also includes a column called HEADING_COLUMN that indicates which heading was detected)

            total_heading_found:    a Boolean indicating whether the row heading with/above the daily total was found
                                        ('with' for Android; 'above' for iOS, as the daily totals have no other text on their rows)

            daily_total:            the daily total screentime/pickups/notifications (depending on category_detected)

            daily_total_minutes:    the daily total converted to an integer number of minutes (screentime only)

            app_data:               a dataframe containing the app names and app numbers (and minutes, for screentime)
                                        extracted from the image; the first index of this dataframe is 1 (not 0)

            errors:                 a list of errors (a.k.a. review reasons) detected during the data extraction process
                                        (all possible errors are listed in ConvenienceVariables.py)

            num_values_below_conf:  the count of app names and app numbers whose confidence values are below a certain limit
                                        (the limit is defined as conf_limit in RuntimeValues.py)

            num_missed_values:      the (estimated) count of missed app names and app numbers calculated during the
                                        function get_app_names_and_numbers (both iOSFunctions.py and AndroidFunctions.py)

            data_row:               a row representing the data extracted from the image, to be appended to the master
                                        list of all screenshots data and output to CSV
        """

        self.participant = participant
        self.url = url
        self.filename = url[url.rfind('/') + 1:] if url is not None else None
        self.user_id = participant.user_id
        self.device_id = device_id
        self.grey_image = None
        self.height = 0
        self.width = 0
        self.is_light_mode = None
        self.text = pd.DataFrame
        self.words_df = pd.DataFrame
        self.text_hash = None
        self.language = None
        self.date_format = None
        self.device_os_submitted = device_os
        self.device_os_detected = device_os
        self.android_version = None
        self.time_format_short = None
        self.time_format_long = None
        self.time_format_end_of_line = None
        self.date_submitted = date
        self.date_detected = None
        self.category_submitted = category
        self.category_detected = None
        self.relative_day = None
        self.rows_with_day_type = pd.DataFrame
        self.rows_with_date = pd.DataFrame
        self.headings_df = pd.DataFrame
        self.total_heading_found = False
        self.daily_total = None
        self.daily_total_conf = None
        self.daily_total_minutes = None
        self.app_data = None
        self.errors = []
        self.num_values_low_conf = 0
        self.num_missed_values = 0
        self.data_row = initialize_data_row()

    def __str__(self):
        s_date = f"Date submitted: {self.date_submitted}".ljust(33)
        s_cat = f"Category submitted: {self.category_submitted}"
        return f"{s_date}{s_cat}"

    def set_image(self, img):
        self.grey_image = img

    def set_dimensions(self, dim):
        self.height, self.width = dim[0], dim[1]

    def set_is_light_mode(self, tf):
        self.is_light_mode = tf

    def set_text(self, text):
        self.text = text

    def set_words_df(self, words):
        self.words_df = words

    def set_hash(self, _hash):
        self.text_hash = _hash

    def set_language(self, lang):
        self.language = lang

    def set_date_format(self, fmt):
        self.date_format = fmt

    def set_device_os(self, dev_os):
        self.device_os_submitted = dev_os
        self.device_os_detected = dev_os

    def set_device_os_detected(self, _os):
        self.device_os_detected = _os

    def set_android_version(self, ver):
        self.android_version = ver

    def set_time_formats(self, fmts):
        self.time_format_short = fmts[0]
        self.time_format_long = fmts[1]
        self.time_format_end_of_line = fmts[2]

    def set_date_detected(self, date):
        self.date_detected = date

    def set_category_detected(self, cat):
        self.category_detected = cat

    def set_relative_day(self, day):
        self.relative_day = day

    def set_rows_with_day_type(self, df):
        self.rows_with_day_type = df

    def set_rows_with_date(self, rows_df):
        self.rows_with_date = rows_df

    def set_headings(self, df):
        self.headings_df = df

    def set_total_heading_found(self, tf):
        self.total_heading_found = tf

    def set_daily_total(self, value, conf=NO_CONF):
        self.daily_total = value
        self.daily_total_conf = conf

    def set_daily_total_minutes(self, minutes):
        self.daily_total_minutes = minutes

    def set_app_data(self, data):
        self.app_data = data

    def add_error(self, error, num=0):
        if error not in self.errors:
            self.errors.append(error)
            self.data_row[error] = True
            if num > 0:
                if error == RuntimeValues.ERR_CONFIDENCE:
                    self.num_values_low_conf = num
                elif error == ERR_MISSING_VALUE:
                    self.num_missed_values = num
