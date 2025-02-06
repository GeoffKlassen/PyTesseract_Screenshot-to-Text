from ScreenshotClass import Screenshot
from LanguageDictionaries import *
from RuntimeValues import *
import pandas as pd
from datetime import datetime

"""
    Variables defined for use as column names
"""
SCREENTIME = 'screentime'
PICKUPS = 'pickups'
NOTIFICATIONS = 'notifications'

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

NO_TEXT = '-99999'
NO_NUMBER = -99999
NO_CONF = -1

TODAY = 'today'
YESTERDAY = 'yesterday'
DAY_OF_THE_WEEK = 'weekday'
WEEK = 'week'

PARTICIPANT_ID = 'participant_id'
IS_RESEARCHER = 'is_researcher'
RESPONSE_DATE = 'response_date'
IMG_RESPONSE_TYPE = 'img_response_type'
IMG_URL = 'img_url'


def compile_list_of_urls(df, screentime_cols, pickups_cols, notifications_cols,
                         time_col='Record Time', id_col='Participant ID', label_col='Participant Label'):
    """Create a dataframe of URLs from a provided dataframe of survey responses.
    Args:
        df:                 The dataframe with columns that contain URLs
        screentime_cols:    An array of the column names in df that are for URLs of screentime images
        pickups_cols:       An array of the column names in df that are for URLs of pickups (unlocks) images
        notifications_cols: An array of the column names in df that are for URLs of notifications images
        time_col:           The name of the column in df that lists the date & time stamp of submission
                            (For Avicenna CSVs, this is usually 'Record Time')
        id_col:             The name of the column in df that contains the ID of the user
                            (For Avicenna CSVs, this is usually 'Participant ID')
        label_col:          The name of the column in df that indicates whether the row is for a IS_RESEARCHER
                            (For Avicenna CSVs, this is usually 'Participant Label')
    Returns:
        pd.Dataframe:       A dataframe of image URLs, along with the user ID, response date, and category the image was
                            submitted in.

    NOTE: iOS & Android Dashboards have three categories of usage statistics: screentime, pickups (aka unlocks), and
    notifications. Because of this, the dataframe provided may have multiple columns of URLs (e.g. one column for each
    usage category, multiple columns for only one category, or multiple columns for all three categories).
    """
    url_df = pd.DataFrame(columns=[PARTICIPANT_ID, IS_RESEARCHER, RESPONSE_DATE, IMG_RESPONSE_TYPE, IMG_URL])
    for i in df.index:
        user_id = df[id_col][i]

        # Extract the response date from the date column, in date format
        try:
            response_date = datetime.strptime(str(df[time_col][i])[0:10], "%Y-%m-%d")  # YYYY-MM-DD = 10 chars
        except ValueError:  #
            response_date = datetime.strptime("1999-01-01", "%Y-%m-%d")  # TODO: try None?

        for col_name in screentime_cols + pickups_cols + notifications_cols:
            # Cycle through each column in which a user might have submitted a screenshot and add its URL to the list
            url = str(df[col_name][i])
            if not url.startswith("https://file.avicennaresearch.com"):
                continue

            if col_name in screentime_cols:
                img_response_type = SCREENTIME
            elif col_name in pickups_cols:
                img_response_type = PICKUPS
            elif col_name in notifications_cols:
                img_response_type = NOTIFICATIONS
            else:
                img_response_type = None
            # Note: For the Boston Children's Hospital data, all images are of type SCREENTIME

            new_row = {PARTICIPANT_ID: user_id,
                       IS_RESEARCHER: True if df.loc[i, id_col] == IS_RESEARCHER else False,
                       RESPONSE_DATE: response_date.date(),
                       IMG_RESPONSE_TYPE: img_response_type,
                       IMG_URL: url}
            url_df = pd.concat([url_df, pd.DataFrame([new_row])], ignore_index=True)

    return url_df


if __name__ == '__main__':
    dir_for_downloaded_images = "Saved Images"
    if not os.path.exists(dir_for_downloaded_images):
        os.makedirs(dir_for_downloaded_images)

    # Read in the list of URLs for the appropriate Study (as specified in RuntimeValues.py)
    url_list = pd.DataFrame()
    for survey in survey_list:
        print(f"Compiling URLs from {survey[CSV_FILE]}...", end='')
        survey_csv = pd.read_csv(survey[CSV_FILE])
        current_list = compile_list_of_urls(survey_csv, survey[SCREEN_COLS], survey[PICKUP_COLS], survey[NOTIF_COLS])
        print(f"Done.\n{current_list.shape[0]} URLs found.")
        url_list = pd.concat([url_list, current_list], ignore_index=True)
    print(f'All URLs compiled. Total URLs: {url_list.shape[0]}')

    # Cycle through the images, creating a screenshot object for each one
    screenshots = []
    for index in url_list.index:
        current_screenshot = Screenshot(url=url_list[IMG_URL][index],
                                        user_id=url_list[PARTICIPANT_ID][index],
                                        date=url_list[RESPONSE_DATE][index],
                                        category=url_list[IMG_RESPONSE_TYPE][index])
        print(current_screenshot)