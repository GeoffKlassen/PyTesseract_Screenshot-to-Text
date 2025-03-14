"""This file contains variables used as dictionary keys, dataframe column names, etc.
They are stored in this one location for consistency."""
from win32api import GetSystemMetrics
screen_height = GetSystemMetrics(1)

"""
    Operating Systems
"""
IOS = 'iOS'
ANDROID = 'Android'
UNKNOWN = 'Unknown'


"""
    Categories of data that appear on the iOS & Android dashboards
"""
SCREENTIME = 'screentime'
PICKUPS = 'pickups'
NOTIFICATIONS = 'notifications'
UNLOCKS = 'unlocks'
""" 'PICKUPS' on iOS is similar to 'UNLOCKS' on Android, except that, at the app level, iOS measures the number of
times each app was the first app used after pickup, while Android measures the number of times each app was opened
overall, regardless of whether it was the first app opened after pickup. As such, on iOS, the sum of all app pickups
should equal the total pickups, and on Android the total unlocks is generally lower than the sum of all app unlocks
(but technically they are simply correlated, so it's possible for total unlocks to be greater).
"""

"""
    Heading column names for dataframes
"""
HEADING_COLUMN = 'heading'
OS_COLUMN = 'OS'

DAY_OR_WEEK_HEADING = 'day or week'
DATE_HEADING = 'date'
SCREENTIME_HEADING = SCREENTIME
NOTIFICATIONS_HEADING = NOTIFICATIONS

# The following headings are unique to iOS:
LIMITS_HEADING = 'limits'
MOST_USED_HEADING = 'most used'
PICKUPS_HEADING = PICKUPS
FIRST_PICKUP_HEADING = 'first pickup'
FIRST_USED_AFTER_PICKUP_HEADING = 'first used after pickup'
HOURS_AXIS_HEADING = 'hours row'

LIMIT_USAGE_HEADING = 'limit usage'
COMMUNICATION_HEADING = 'communication'
SEE_ALL_ACTIVITY = 'see all activity'

IOS_EXCLUSIVE_HEADINGS = [LIMITS_HEADING,
                          MOST_USED_HEADING,
                          PICKUPS_HEADING,
                          FIRST_PICKUP_HEADING,
                          FIRST_USED_AFTER_PICKUP_HEADING,
                          LIMIT_USAGE_HEADING,
                          COMMUNICATION_HEADING,
                          SEE_ALL_ACTIVITY]

# The following headings are *usually* only found in Android screenshots:
TOTAL_SCREENTIME = 'total screentime'
MOST_USED_APPS_HEADING = 'most used apps'
TOTAL_NOTIFICATIONS = 'total notifications'
MOST_NOTIFICATIONS_HEADING = 'most notifications'
UNLOCKS_HEADING = 'unlocks'
DAYS_AXIS_HEADING = 'day axis'
TOTAL_UNLOCKS = 'total unlocks'
DAY_NAME_HEADING = 'day name'
VIEW_MORE_HEADING = 'view more'
REST_OF_THE_DAY = 'rest of the day'
SEE_ALL_N_APPS = 'See all # apps'
SEARCH_APPS = 'Search apps'
DAY_WEEK_MONTH = 'Day Week Month'
APP_ACTIVITY = 'App activity'

ANDROID_EXCLUSIVE_HEADINGS = [MOST_USED_APPS_HEADING,
                              MOST_NOTIFICATIONS_HEADING,
                              UNLOCKS_HEADING,
                              VIEW_MORE_HEADING,
                              REST_OF_THE_DAY,
                              SEARCH_APPS,
                              SEE_ALL_N_APPS,
                              DAY_WEEK_MONTH,
                              APP_ACTIVITY]

GOOGLE = 'Google'
VERSION_2018 = 'Android 2018'
SAMSUNG_2021 = 'Samsung 2021'
SAMSUNG_2024 = 'Samsung 2024'

OLD_SCREENTIME_HEADING = '2018 screentime'
OLD_MOST_USED_HEADING = '2018 most used'
OLD_UNLOCKS_HEADING = '2018 unlocks'


"""
    Values to use when data is missing or not found
"""
NO_NUMBER = -1
NO_TEXT = "_"  # str(NO_NUMBER)
NO_CONF = -1

"""
    Column names for the dataframe of URLs
"""
PARTICIPANT_ID = 'participant_id'
DEVICE_ID = 'device_id'
IS_RESEARCHER = 'is_researcher'
RESPONSE_DATE = 'response_date'
IMG_RESPONSE_TYPE = 'img_response_type'
IMG_URL = 'img_url'


"""
    Classifiers for the 'day range' of data present in a screenshot
"""
TODAY = 'today'
YESTERDAY = 'yesterday'
DAY_BEFORE_YESTERDAY = 'day before yesterday'
DAY_OF_THE_WEEK = 'weekday'
WEEK = 'week'

"""
    Language abbreviations
"""
GER = 'German'
ITA = 'Italian'
ENG = 'English'
FRA = 'French'

"""
    Study attributes
"""
NAME = 'name'
DIRECTORY = 'Directory'
DEFAULT_LANGUAGE = 'Default Language'
SURVEY_LIST = 'Survey List'
CATEGORIES = 'Categories'
USER_ID_COLUMN = 'User ID Column Name'
DATE_COLUMN = 'Date Column Name'
DEVICE_ID_COLUMN = 'Device ID Column Name'
MAX_APPS = 'Maximum Apps per Category'

"""
    Survey attributes
"""
CSV_FILE = 'csv_file'
SCREEN_COLS = 'screentime'
PICKUP_COLS = 'pickups'
NOTIFY_COLS = 'notifications'
URL_COLUMNS = 'urls'

MISREAD_TIME_FORMAT_IOS = (r'\b[12T]?[0-9toAQ]\s?[hn]\s?[1-5tA]?[0-9tA]\s?mi?n?\b'
                           r'|\b[12T]?[0-9toA]\s?[hn]\b'
                           r'|\b[1-5tA]?[0-9itA]\s?mi?n?\b'
                           r'|\b[1-5]?[0-9O]\s?s\b')
MISREAD_NUMBER_FORMAT = r'\b[0-9ASLlTK]+\b'
MISREAD_TIME_OR_NUMBER_FORMAT = '|'.join([MISREAD_TIME_FORMAT_IOS, MISREAD_NUMBER_FORMAT])

PROPER_TIME_FORMAT = r'^\d+h$|^\d+h\s?\d+(m|min)$|^\d+(m|min)$|^\d+s$'
PROPER_NUMBER_FORMAT = r'^\d+$'
PROPER_TIME_OR_NUMBER_FORMAT = '|'.join([PROPER_TIME_FORMAT, PROPER_NUMBER_FORMAT])

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (165, 165, 165)  # Not used as a pixel colour, just used as a shorthand for the colour name

APP = 'app'
# NAME = 'name'  # Already defined above
NUMBER = 'number'
TIME = 'time'
MINUTES = 'minutes'
NAME_CONF = 'name_conf'
NUMBER_CONF = 'number_conf'
TOTAL = 'total'

"""
    Error Messages    
"""
ERR = 'ERR'
ERR_UNREADABLE_DATA = 'ERR Unreadable data format'
ERR_APP_AREA = 'ERR App area crop failed'
ERR_FILE_NOT_FOUND = 'ERR File not found'
ERR_DEVICE_ID = 'ERR Unusual device ID'
ERR_NO_TEXT = 'ERR No text found'
ERR_LANGUAGE = 'ERR Language not detected'
ERR_DAY_TEXT = 'ERR Day text not detected'
ERR_DEVICE_OS = 'ERR Different device OS detected'
ERR_CATEGORY = 'ERR Category not detected'
ERR_DAILY_TOTAL = 'ERR Daily total not found'
ERR_APP_DATA = 'ERR App-level data not found'
ERR_DAILY_TOTAL_MISSED = 'ERR Daily total missed'
ERR_OS_NOT_FOUND = 'ERR OS not detected'
ERR_MISSING_APP = 'ERR Suspected missed app(s)'
ERR_MISSING_VALUE = 'ERR Missed values'
ERR_TOTAL_SCREENTIME = 'ERR Daily total matched an app time'
ERR_DUPLICATE_DATA = 'ERR Data matches other screenshot'
ERR_TOTAL_BELOW_APP_SUM = 'ERR Daily total less than app sum'
ERR_MISREAD_TIME = 'ERR Misread time value'
ERR_NOT_A_NUMBER = 'ERR Daily total not a number'
ERR_DUPLICATE_COUNTS = 'ERR Duplicate data occurrences'
SAME_USER = 'SAME USER'
MULTIPLE_USERS = 'MULTIPLE USERS'

"""
    Column names for all_screenshots_df
"""
IMAGE_URL = 'image_url'
# PARTICIPANT_ID = 'Participant ID'  # Already defined above
# DEVICE_ID = 'Device ID'  # Already defined above
LANGUAGE = 'language'
DEVICE_OS = 'device_os'
ANDROID_VERSION = 'android_version'
DATE_SUBMITTED = 'date_submitted'
DATE_DETECTED = 'date_detected'
RELATIVE_DAY = 'relative_day'
CATEGORY_SUBMITTED = 'category_submitted'
CATEGORY_DETECTED = 'category_detected'
DAILY_TOTAL = 'daily_total'
# APP, NAME, NUMBER = 'App', 'Name', 'Number'  # Already defined above
HASHED = 'hashed'
REVIEW_COUNT = 'review_count'
