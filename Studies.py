"""This file contains the definitions for the different studies that have been analyzed using this program so far.
So far:
    HappyB (1.0)
    HappyB2.0
    BCH
"""
import pytesseract
from ConvenienceVariables import *

coding_location = 'home'  # uni or home

if coding_location == 'uni':
    pc_user = 'gbk546'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\gbk546\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
elif coding_location == 'home':
    pc_user = 'Geoff'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
else:
    pc_user = ''
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


"""
    HappyB (1.0)
"""
happyb_baseline_survey = {CSV_FILE: 'activity_response_2207_14679_10.csv',
                          URL_COLUMNS: {SCREEN_COLS: ['[24_IMG] Question 24 of Survey 14679'],
                                        PICKUP_COLS: ['[25_IMG] Question 25 of Survey 14679'],
                                        NOTIFY_COLS: ['[26_IMG] Question 26 of Survey 14679']
                                        }
                          }
happyb_daily_survey = {CSV_FILE: 'activity_response_2207_14718_9.csv',
                       URL_COLUMNS: {SCREEN_COLS: ['[2_IMG] iOS screentime'],
                                     PICKUP_COLS: ['[3_IMG] iOS unlocks'],
                                     NOTIFY_COLS: ['[4_IMG] iOS notifications']
                                     }
                       }

"""
    HappyB2.0 May 2024 (Pre-launch preparations)
"""
happyb2_prep_baseline_survey = {CSV_FILE: 'OCRScript_Android_v2\\study-3749-export-1-survey-responses-20973-2024-07-03-20-37-24.csv',
                                URL_COLUMNS: {SCREEN_COLS: ['[24_IMG] iOS_screenshot_1_duration'],
                                              PICKUP_COLS: ['[25_IMG] iOS_screenshot_2_unlocks'],
                                              NOTIFY_COLS: ['[26_IMG] iOS_screenshot_3_notifications']
                                              }
                                }
happyb2_prep_daily_survey_ios = {CSV_FILE: 'OCRScript_iOS_v2\\study-3749-export-1-survey-responses-20976-2024-07-26-22-11-50.csv',
                                 URL_COLUMNS: {SCREEN_COLS: ['[2_IMG] iOS screentime'],
                                               PICKUP_COLS: ['[3_IMG] iOS unlocks'],
                                               NOTIFY_COLS: ['[4_IMG] iOS notifications']
                                              }
                                 }
happyb2_prep_daily_survey_android = {CSV_FILE: 'OCRScript_Android_v2\\study-3749-export-2-survey-responses-20977-2024-07-03-20-38-28.csv',
                                     # Note: OCRScript_Android_v2\\study-3749-export-175-survey-responses-20977-2024-10-23
                                     # -20-21-05.csv contains ~40,000 URLs. Many of the Android users were found to be
                                     # fraudulent.
                                     URL_COLUMNS: {SCREEN_COLS: ['[2_IMG] AndroidScreentime'],
                                                   PICKUP_COLS: ['[3_IMG] Android unlocks'],
                                                   NOTIFY_COLS: ['[4_IMG] Android notifications']
                                                  }
                                     }

"""
    HappyB2.0 Aug-Dec 2024
"""
happyb2_baseline_survey = {CSV_FILE: 'Baseline Survey 2024, Unique IDs.csv',
                           URL_COLUMNS: {SCREEN_COLS: ['X.24_IMG..iOS_screenshot_1_duration', 'X.28_IMG..Android_screenshot_1_duration'],
                                         PICKUP_COLS: ['X.25_IMG..iOS_screenshot_2_unlocks', 'X.29_IMG..Android_screenshot_2_unlocks'],
                                         NOTIFY_COLS: ['X.26_IMG.Metadata..iOS_screenshot_3_notifications', 'X.30_IMG..Android_screenshot_3_notifications']
                                         }
                           }
happyb2_daily_survey_ios = {CSV_FILE: 'Screenshot Survey iOS 2024, Unique IDs.csv',
                            URL_COLUMNS: {SCREEN_COLS: ['X.2_IMG..iOS.screentime'],
                                          PICKUP_COLS: ['X.3_IMG..iOS.unlocks'],
                                          NOTIFY_COLS: ['X.4_IMG..iOS.notifications']
                                         }
                            }
happyb2_daily_survey_android = {CSV_FILE: 'Screenshot Survey Android 2024, Unique IDs.csv',
                                URL_COLUMNS: {SCREEN_COLS: ['X.2_IMG..AndroidScreentime'],
                                              PICKUP_COLS: ['X.3_IMG..Android.unlocks'],
                                              NOTIFY_COLS: ['X.4_IMG..Android.notifications']
                                             }
                                }


"""
    Boston Children's Hospital (BCH)
"""
bch_survey = {CSV_FILE: 'study-2037-export-3-survey-responses-16872-2024-10-01-14-25-42.csv',
              URL_COLUMNS: {SCREEN_COLS: ['[3_IMG] Question 3 of Survey 16872', '[4_IMG] Question 4 of Survey 16872'],
                            PICKUP_COLS: [],
                            NOTIFY_COLS: []
                            }
              }
# For the BCH Study, there is only one CSV, in which there are two columns of SCREENTIME screenshots.

"""
    Definitions of the studies the code has been used on
"""
study_happyb2_0 = {NAME: "HappyB2.0 Prep",
                   DIRECTORY: f'C:\\Users\\{pc_user}\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0',
                   DEFAULT_LANGUAGE: ENG,
                   SURVEY_LIST: [happyb2_prep_baseline_survey, happyb2_prep_daily_survey_ios, happyb2_prep_daily_survey_android],
                   CATEGORIES: [SCREENTIME, PICKUPS, NOTIFICATIONS],
                   'User ID Column Name': 'Participant ID',
                   'Date Column Name': 'Record Time',
                   'Device ID Column Name': 'Device ID',
                   MAX_APPS: 3}

study_happyb2_0_2024 = {NAME: "HappyB2.0 2024 Aug-Dec",
                        DIRECTORY: f'C:\\Users\\{pc_user}\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0 2024',
                        DEFAULT_LANGUAGE: ENG,
                        SURVEY_LIST: [happyb2_baseline_survey, happyb2_daily_survey_ios],  # happyb2_daily_survey_android],
                        CATEGORIES: [SCREENTIME, PICKUPS, NOTIFICATIONS],
                        'User ID Column Name': 'Participant.ID',
                        'Date Column Name': 'Record.Time',
                        'Device ID Column Name': 'Device.ID',
                        MAX_APPS: 3}

study_bch = {NAME: "BCH",
             DIRECTORY: f'C:\\Users\\{pc_user}\\OneDrive - University of Saskatchewan\\Grad Studies\\Boston Childrens Hospital',
             DEFAULT_LANGUAGE: ENG,
             SURVEY_LIST: [bch_survey],
             CATEGORIES: [SCREENTIME],
             'User ID Column Name': 'Participant ID',
             'Date Column Name': 'Record Time',
             'Device ID Column Name': 'Device ID',
             MAX_APPS: 8}

studies = [study_happyb2_0, study_happyb2_0_2024, study_bch]
