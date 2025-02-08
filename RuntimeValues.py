"""This file contains variables that the user can change prior to runtime."""
import os
import pytesseract
from Studies import *
from LanguageDictionaries import *

"""
    User can specify one of the pre-set studies that have been analyzed with this program already.
    As of now, that is either HappyB2.0 or BCH.
    Otherwise, the values below can be customized.
"""
studies = ["HappyB2.0",      "BCH"]
#           studies[0], studies[1] 
study_to_analyze = studies[1]   

pc_user = 'geoff'  # gbk546 (when at University of Saskatchewan) or geoff (when at home)
if study_to_analyze == "HappyB2.0":
    os.chdir(
        f'C:\\Users\\{pc_user}\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0\\OCRScript_iOS_v2')
    default_language = ENG
    survey_list = [happyb2_baseline_survey, happyb2_daily_survey]
elif study_to_analyze == "BCH":
    os.chdir(
        f'C:\\Users\\{pc_user}\\OneDrive - University of Saskatchewan\\Grad Studies\\Boston Childrens Hospital')
    default_language = ENG
    survey_list = [bch_survey]
else:
    """ These are the values that can be customized per study """
    # Set the current working directory (CWD)
    os.chdir(
        'C:\\Users\\gbk546\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0\\OCRScript_iOS_v2')  # For HappyB 2.0

    # This is the default language to set images to, in case none of the language keywords are found in the image.
    default_language = ENG  # For HappyB (1.0): ITA

    # Specify the list of surveys to extract URLs from.
    # survey_list = [happyb_baseline_survey, happyb_daily_survey]       # For HappyB (1.0)
    # survey_list = [happyb2_baseline_survey, happyb2_daily_survey]     # For HappyB2.0
    survey_list = [bch_survey]  # For Boston Children's Hospital

# Where to store downloaded images (a subfolder within the CWD)
dir_for_downloaded_images = "Saved Images"
use_downloaded_images = True  # If False, local copies of images are not used (always download images from URLs).

# Login credentials for downloading images from www.avicennaresearch.com
user = "geoff.klassen@usask.ca"
passw = "Phi1*618ah"

# Location of PyTesseract on local drive
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\gbk546\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # At U of S
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'  # At home

show_images = False  # If True, images of the screenshots will be shown during runtime (mostly for debugging).

conf_limit = 80

test_lower_bound = 100
test_upper_bound = 110
