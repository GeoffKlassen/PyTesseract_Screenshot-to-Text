"""This file contains the definitions for the different studies that have been analyzed using this program so far.
So far:
    HappyB (1.0)
    HappyB2.0
    BCH
"""

CSV_FILE = 'csv_file'
SCREEN_COLS = 'screen_cols'
PICKUP_COLS = 'pickup_cols'
NOTIF_COLS = 'notif_cols'

"""
    HappyB (1.0)
"""
happyb_baseline_survey = {CSV_FILE: 'activity_response_2207_14679_10.csv',
                          SCREEN_COLS: ['[24_IMG] Question 24 of Survey 14679'],
                          PICKUP_COLS: ['[25_IMG] Question 25 of Survey 14679'],
                          NOTIF_COLS: ['[26_IMG] Question 26 of Survey 14679']}
happyb_daily_survey = {CSV_FILE: 'activity_response_2207_14718_9.csv',
                       SCREEN_COLS: ['[2_IMG] iOS screentime'],
                       PICKUP_COLS: ['[3_IMG] iOS unlocks'],
                       NOTIF_COLS: ['[4_IMG] iOS notifications']}

"""
    HappyB2.0
"""
happyb2_baseline_survey = {CSV_FILE: 'study-3749-export-173-survey-responses-20973-2024-10-23-20-20-34.csv',
                           SCREEN_COLS: ['[24_IMG] iOS_screenshot_1_duration'],
                           PICKUP_COLS: ['[25_IMG] iOS_screenshot_2_unlocks'],
                           NOTIF_COLS: ['[26_IMG] iOS_screenshot_3_notifications']}
happyb2_daily_survey = {CSV_FILE: 'study-3749-export-174-survey-responses-20976-2024-10-23-20-20-53.csv',
                        SCREEN_COLS: ['[2_IMG] iOS screentime'],
                        PICKUP_COLS: ['[3_IMG] iOS unlocks'],
                        NOTIF_COLS: ['[4_IMG] iOS notifications']}

"""
    Boston Children's Hospital
"""
bch_survey = {CSV_FILE: 'study-2037-export-3-survey-responses-16872-2024-10-01-14-25-42.csv',
              SCREEN_COLS: ['[3_IMG] Question 3 of Survey 16872', '[4_IMG] Question 4 of Survey 16872'],
              PICKUP_COLS: [],
              NOTIF_COLS: []}
# For the BCH Study, there is only one CSV, in which there are two columns for SCREENTIME data.
