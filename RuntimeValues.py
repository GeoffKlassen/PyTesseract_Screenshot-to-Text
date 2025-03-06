"""This file contains variables that the user can change prior to runtime."""

import os

from Studies import *
from LanguageDictionaries import *

"""
    User can specify one of the pre-set studies that have been analyzed with this program already.
    Otherwise, the values below can be customized.
    
    As of Feb 10, 2025, there are 2 preset studies available: HappyB2.0 and BCH.
"""


study_to_analyze = studies[1]
# Use studies[0] for the HappyB2.0 study (pre-launch prep)
# Use studies[1] for the HappyB2.0 study (2024 Aug-Dec)
# Use studies[2] for the BCH study

if study_to_analyze in studies:
    os.chdir(study_to_analyze['Directory'])
    default_language = study_to_analyze['Default Language']
    survey_list = study_to_analyze['Survey List']
    categories_included = study_to_analyze['Categories']
    max_apps_per_category = study_to_analyze['Maximum Apps per Category']
    date_record_col_name = study_to_analyze['Date Column Name']
    device_id_col_name = study_to_analyze['Device ID Column Name']
    user_id_col_name = study_to_analyze['User ID Column Name']

else:

    """ These are the values that can be customized per study """
    # Set the current working directory (CWD) where the survey CSV files are located
    os.chdir(
        'C:\\Users\\gbk546\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0\\OCRScript_iOS_v2')  # For HappyB 2.0

    # This is the default language to set images to, in case none of the language keywords are found in the image.
    default_language = ENG  # For HappyB (1.0): ITA

    # Specify the list of surveys to extract URLs from.
    # survey_list = [happyb_baseline_survey, happyb_daily_survey]       # For HappyB (1.0)
    # survey_list = [happyb2_baseline_survey, happyb2_daily_survey]     # For HappyB2.0
    survey_list = [bch_survey]  # For Boston Children's Hospital
    categories_included = [SCREENTIME]
    max_apps_per_category = 8  # The

dir_for_downloaded_images = "Saved Images"  # Where to store downloaded images (a sub-folder within the CWD)
use_downloaded_images = True  # If False, local copies of images are not used (all images are downloaded at runtime).
save_downloaded_images = True  # If True, images downloaded at runtime are saved to a local folder for quicker access.

# Login credentials for downloading images from www.avicennaresearch.com
user = "geoff.klassen@usask.ca"
passw = "Phi1*618ah"

show_images = True  # If True, images of the screenshots will be shown during runtime (mostly for debugging).

app_area_scale_factor = 1  # In addition to the screenshot_scale_factor, this is how much to scale the cropped image
                              # when searching for app-level data

conf_limit = 80
ERR_CONFIDENCE = f"ERR Values below {int(conf_limit)}% confidence"
# Location of PyTesseract on local drive
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\gbk546\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # At U of S

test_lower_bound = 3296
test_upper_bound = 19313
# 553 for HappyB2.0 (pre-launch)
# 19313 URLs for HappyB2.0 2024 Aug-Dec
# 452 for BCH study
