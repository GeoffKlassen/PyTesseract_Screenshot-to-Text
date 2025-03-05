"""This file contains variables used as dictionary keys, dataframe column names, etc.
They are stored in this one location for consistency."""


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
# 'PICKUPS' on iOS is similar to 'UNLOCKS' on Android, except that, at the app level,
# iOS measures the number of times each app was the first app used after pickup, while Android measures the number of
# times each app was opened overall, regardless of whether it was the first app opened after pickup. As such, on iOS
# the sum of all app pickups should equal the total pickups on iOS, and on Android the total unlocks is generally lower
# than the sum of all app unlocks (but technically they are simply correlated, so it's possible for total unlocks to be greater).

"""
    Heading column names for dataframes
"""
HEADING_COLUMN = 'heading'

DAY_OR_WEEK_HEADING = 'day or week'
DATE_HEADING = 'date'

# The following headings are unique to iOS:
SCREENTIME_HEADING = ' '.join(['screentime', IOS])
LIMITS_HEADING = ' '.join(['limits', IOS])
MOST_USED_HEADING = ' '.join(['most used', IOS])
PICKUPS_HEADING = ' '.join(['pickups', IOS])
FIRST_PICKUP_HEADING = ' '.join(['total pickups', IOS])
FIRST_USED_AFTER_PICKUP_HEADING = ' '.join(['first used after pickup', IOS])
NOTIFICATIONS_HEADING = ' '.join(['notifications', IOS])
HOURS_AXIS_HEADING = ' '.join(['hours row', IOS])

LIMIT_USAGE_HEADING = ' '.join(['limit usage', IOS])
COMMUNICATION_HEADING = ' '.join(['communication', IOS])
SEE_ALL_ACTIVITY = ' '.join(['see all activity', IOS])

# The following headings are *usually* only found in Android screenshots:
TOTAL_SCREENTIME = 'total screentime'
MOST_USED_APPS_HEADING = ' '.join(['most used apps', ANDROID])
TOTAL_NOTIFICATIONS = 'total notifications'
MOST_NOTIFICATIONS_HEADING = ' '.join(['most notifications', ANDROID])
UNLOCKS_HEADING = ' '.join(['unlocks', ANDROID])
DAYS_AXIS_HEADING = 'day axis'
TOTAL_UNLOCKS = ' '.join(['most used', ANDROID])
DAY_NAME_HEADING = 'day name'
VIEW_MORE_HEADING = ' '.join(['most used', ANDROID])
REST_OF_THE_DAY = ' '.join(['rest of the day', ANDROID])

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
NO_TEXT = str(NO_NUMBER)
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
NAME = 'Name'
DIRECTORY = 'Directory'
DEFAULT_LANGUAGE = 'Default Language'
SURVEY_LIST = 'Survey List'
CATEGORIES = 'Categories'
MAX_APPS = 'Maximum Apps per Category'

"""
    Survey attributes
"""
CSV_FILE = 'csv_file'
SCREEN_COLS = 'screentime'
PICKUP_COLS = 'pickups'
NOTIFY_COLS = 'notifications'
URL_COLUMNS = 'urls'

# misread_time_format = r'^[\d|t]+\s?[hn]$|^[\d|t]+\s?[hn]\s?[\d|tA]+\s?(min|m)$|^.{0,2}\s?[0-9AIt]+\s?(min|m)$|\d+\s?s$'
misread_time_format = r'\b[12T]?[0-9toAQ]\s?[hn]\s?[1-5tA]?[0-9tA]\s?mi?n?\b|\b[12T]?[0-9toA]\s?[hn]\b|\b[1-5tA]?[0-9itA]\s?mi?n?\b|\b[1-5]?[0-9O]\s?s\b'
misread_number_format = r'\b[0-9AS]+\b'
misread_time_or_number_format = '|'.join([misread_time_format, misread_number_format])

# I don't think these are necessary
# misread_hr_format = r'\b[0-9to]{1,2}\s?[hn]\b'
# misread_hrmin_format = r'\b[0-9to]{1,2}\s?[hn]\s?[0-9tA]{1,2}\s?(min|m))\b'
# misread_min_format = r'\b[0-9AIt]{1,2}\s?mi?n?)\b'
# misread_sec_format = r'\b\d{1,2}\s?s\b'

time_format = r'^\d+h$|^\d+h\s?\d+(m|min)$|^\d+(m|min)$|^\d+s$'
number_format = r'^\d+$'
time_or_number_format = '|'.join([time_format, number_format])

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (165, 165, 165)  # Not used as a pixel colour, just used as a shorthand for the colour name

APP = 'app'
NUMBER = 'number'

"""
    Error Messages    
"""
ERR_UNRELATED_DATA = 'Unrelated data detected'
ERR_APP_AREA = 'App area not detected'
ERR_DATA_NOT_READ = 'Error reading data'
ERR_NO_TEXT = 'No text found'
ERR_LANGUAGE = 'Language not detected'
ERR_DAY_TEXT = 'Day text not detected'
ERR_DEVICE_OS = 'Different device OS detected'
ERR_CATEGORY = 'Category not detected'
ERR_DAILY_TOTAL = 'Daily total not found'
ERR_APP_DATA = 'App-level data not found'
ERR_DAILY_TOTAL_MISSED = 'Daily total missed'
ERR_OS_NOT_FOUND = 'OS not detected'
ERR_MISSING_APP = 'Suspected missed app(s)'
ERR_MISSING_VALUE = 'Missed values'
ERR_TOTAL_SCREENTIME = 'Daily total matched an app time'