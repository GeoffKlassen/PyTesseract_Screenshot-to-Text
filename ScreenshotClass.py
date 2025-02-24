"""The Screenshot class will be used to store the raw image and the extracted data from a screenshot,
   as well as metadata about the screenshot as taken from the database of URLs."""
import pandas as pd
import RuntimeValues
from ConvenienceVariables import *


def initialize_data_row():
    df = pd.DataFrame(columns=['image_url', 'participant_id', 'language', 'device_os',
                               'date_submitted', 'date_detected', 'day_type',
                               'category_submitted', 'category_detected'])
    df[f'daily_total'] = None
    for i in range(1, RuntimeValues.max_apps_per_category + 1):
        df[f'app_{i}_name'] = None
        df[f'app_{i}_number'] = None
    df['num_review_reasons'] = None
    return df


class Screenshot:

    def __init__(self, participant, url, device_os=None, date=None, category=None):
        """Initialize an object of type Screenshot
            :param participant: the participant (object) that submitted this screenshot (has variable user_id)
            :param url: the full https://.../image.jpg URL as taken from the database of URLs
        """

        self.participant = participant
        self.url = url
        self.filename = url[url.rfind('/') + 1:] if url is not None else None
        self.user_id = participant.user_id
        self.device_os = device_os
        self.android_version = None
        self.date_submitted = date
        self.category_submitted = category
        self.height = 0
        self.width = 0
        self.is_light_mode = None
        self.scale_factor = None  # Might not be necessary?
        self.language = None
        self.date_format = None
        self.grey_image = None
        self.date_detected = None
        self.category_detected = None
        self.text = None
        self.words_df = None
        self.time_period = None
        self.rows_with_day_type = None
        self.headings_df = None
        self.daily_total = None
        self.daily_total_conf = None
        self.daily_total_minutes = None
        self.total_heading_found = False
        self.app_data = None
        self.screentime_subheading_found = None
        self.pickups_subheading_found = None
        self.notifications_subheading_found = None
        self.rows_with_date = None
        self.data_row = initialize_data_row()
        self.errors = []
        self.num_values_low_conf = 0
        self.num_missed_values = 0

    def __str__(self):
        s_user_id = f"User ID: {self.user_id}".ljust(22)
        s_device_os = f"Device OS: {self.device_os}".ljust(23)
        s_date = f"Date submitted: {self.date_submitted}".ljust(33)
        s_cat = f"Category submitted: {self.category_submitted}"
        return f"URL: {self.url}\n{s_user_id}{s_device_os}{s_date}{s_cat}"

    def set_dimensions(self, dim):
        self.height, self.width = dim[0], dim[1]

    def set_language(self, lang):
        self.language = lang

    def set_image(self, img):
        self.grey_image = img

    def set_date_detected(self, date):
        self.date_detected = date

    def set_category_detected(self, cat):
        self.category_detected = cat
        df = self.headings_df
        if self.category_detected == SCREENTIME:
            self.screentime_subheading_found = True if (self.category_detected is not None and
                                                        MOST_USED_HEADING in df[HEADING_COLUMN].values) else False
        elif self.category_detected == PICKUPS:
            self.pickups_subheading_found = True if (self.category_detected is not None and
                                                     FIRST_USED_AFTER_PICKUP_HEADING in df[HEADING_COLUMN].values) else False
        elif self.category_detected == NOTIFICATIONS:
            self.notifications_subheading_found = True if (self.category_detected is not None and
                                                           HOURS_AXIS_HEADING in df[HEADING_COLUMN].values) else False
        else:
            pass

    def set_text(self, text):
        self.text = text

    def set_time_period(self, period):
        self.time_period = period

    def set_rows_with_day_type(self, df):
        self.rows_with_day_type = df

    def set_headings(self, df):
        self.headings_df = df

    def set_daily_total(self, value, conf=NO_CONF):
        self.daily_total = value
        self.daily_total_conf = conf

    def set_scale_factor(self, scale):
        self.scale_factor = scale

    def set_daily_total_minutes(self, minutes):
        self.daily_total_minutes = minutes

    def set_is_light_mode(self, tf):
        self.is_light_mode = tf

    def set_date_format(self, fmt):
        self.date_format = fmt

    def set_app_data(self, data):
        self.app_data = data

    def set_android_version(self, ver):
        self.android_version = ver

    def set_rows_with_date(self, rows_df):
        self.rows_with_date = rows_df

    def set_words_df(self, words):
        self.words_df = words

    def set_screentime_subheading_found(self, tf):
        self.screentime_subheading_found = tf

    def set_pickups_subheading_found(self, tf):
        self.pickups_subheading_found = tf

    def set_notifications_subheading_found(self, tf):
        self.notifications_subheading_found = tf

    def set_total_heading_found(self, tf):
        self.total_heading_found = tf

    def add_error(self, error, num=0):
        if error not in self.errors:
            self.errors.append(error)
            self.data_row[f"ERR {error}"] = True
            if num > 0:
                if "Values below" in error:
                    self.num_values_low_conf = num
                elif "Missed values" in error:
                    self.num_missed_values = num

