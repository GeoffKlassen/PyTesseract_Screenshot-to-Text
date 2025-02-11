"""This file contains variables used as dictionary keys, dataframe column names, etc.
They are stored in this one location for consistency."""

"""
    Categories of data that appear on the iOS & Android dashboards
"""
SCREENTIME = 'screentime'
PICKUPS = 'pickups'
NOTIFICATIONS = 'notifications'

"""
    Heading column names for dataframes
"""
HEADING_COLUMN = 'heading'
SCREENTIME_HEADING = SCREENTIME
LIMITS_HEADING = 'limits'
MOST_USED_HEADING = 'most used'
PICKUPS_HEADING = PICKUPS
FIRST_PICKUP_HEADING = 'first pickup'
FIRST_USED_AFTER_PICKUP_HEADING = 'first used after pickup'
NOTIFICATIONS_HEADING = NOTIFICATIONS
HOURS_AXIS_HEADING = 'hours row'
DAY_OR_WEEK_HEADING = 'day or week'

"""
    Values to use when data is missing or not found
"""
NO_TEXT = '-99999'
NO_NUMBER = -99999
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
    Operating Systems
"""
IOS = 'iOS'
ANDROID = 'Android'
UNKNOWN = 'Unknown'

"""
    Classifiers for the 'day range' of data present in a screenshot
"""
TODAY = 'today'
YESTERDAY = 'yesterday'
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

"""
    Survey attributes
"""
CSV_FILE = 'csv_file'
SCREEN_COLS = 'screentime'
PICKUP_COLS = 'pickups'
NOTIFY_COLS = 'notifications'
URL_COLUMNS = 'urls'
