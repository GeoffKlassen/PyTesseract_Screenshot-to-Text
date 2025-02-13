"""The Screenshot class will be used to store the raw image and the extracted data from a screenshot,
   as well as metadata about the screenshot as taken from the database of URLs."""
from ConvenienceVariables import *
import re

MINUTES_FORMAT = r'(min|m)'
HOURS_FORMAT = 'h'
SECONDS_FORMAT = 's'


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
        self.date_submitted = date
        self.category_submitted = category
        self.height = 0
        self.width = 0
        self.scale_factor = None
        self.language = None
        self.grey_image = None
        self.date_detected = None
        self.category_detected = None
        self.text = None
        self.time_period = None
        self.rows_with_day_type = None
        self.headings_df = None
        self.daily_total = None
        self.daily_total_conf = None
        self.daily_total_minutes = None
        self.heading_above_applist = None
        self.heading_below_applist = None


    def convert_text_time_to_minutes(self):
        """
            For Screentime screenshots, coverts the daily total (String) into a number of minutes (int).
        :return: (int) The daily total time converted to minutes
        """
        if str(self.daily_total) == NO_TEXT:
            return NO_NUMBER

        def extract_unit_of_time_as_int(time_str, time_format):
            """
            Finds the portion of the string time_str that represents a number of units of time (for a given unit) and
            extracts the number of those units from that portion.
            :param time_str: The string to search for a time format
            :param time_format: The abbreviated form of a unit of time, as used by the OS
            :return: (int) The number of units of time present in the given string
            """
            time_format_regex = ''.join([r'\d+\s?', time_format])
            extracted_time_as_str = re.search(time_format_regex, time_str)
            if extracted_time_as_str:
                extracted_time_as_str = extracted_time_as_str.group()
                extracted_time_int = int(re.sub(time_format, '', extracted_time_as_str))
                return extracted_time_int
            else:
                return 0

        total_usage_time_mins = 0
        usage_time_seconds = extract_unit_of_time_as_int(self.daily_total, time_format=SECONDS_FORMAT)
        if not usage_time_seconds:
            usage_time_hours = (extract_unit_of_time_as_int(self.daily_total, time_format=HOURS_FORMAT))
            usage_time_hours_to_minutes = (usage_time_hours * 60) if usage_time_hours else 0
            total_usage_time_mins = extract_unit_of_time_as_int(self.daily_total, time_format=MINUTES_FORMAT)
            total_usage_time_mins += usage_time_hours_to_minutes

        return total_usage_time_mins

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
        if cat == SCREENTIME:
            self.heading_above_applist = MOST_USED_HEADING
            self.heading_below_applist = PICKUPS_HEADING
        elif cat == PICKUPS:
            self.heading_above_applist = FIRST_USED_AFTER_PICKUP_HEADING
            self.heading_below_applist = NOTIFICATIONS_HEADING
        elif cat == NOTIFICATIONS:
            self.heading_above_applist = HOURS_AXIS_HEADING
            self.heading_below_applist = ''
        else:
            self.heading_above_applist = ''
            self.heading_below_applist = ''

    def set_text(self, text):
        self.text = text

    def set_time_period(self, period):
        self.time_period = period

    def set_rows_with_day_type(self, df):
        self.rows_with_day_type = df

    def set_headings(self, df):
        self.headings_df = df

    def set_daily_total(self, value, conf):
        self.daily_total = value
        self.daily_total_conf = conf
        self.daily_total_minutes = self.convert_text_time_to_minutes() if self.category_detected == SCREENTIME else None


    def set_scale_factor(self, scale):
        self.scale_factor = scale
