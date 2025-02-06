"""This file contains variables that the user can change prior to runtime."""
import os
import pytesseract
from Studies import *

"""
    User can specify one of the pre-set studies that have been analyzed with this program already.
    As of now, that is either HappyB2.0 or BCH.
"""
main_study = "BCH"  # "HappyB2.0" or "BCH"
# If main_study is not one of the presets, the user can specify the parameters below.

# Current working directory (CWD)
os.chdir('C:\\Users\\gbk546\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0\\OCRScript_iOS_v2')  # For HappyB 2.0

# Login credentials for downloading images from www.avicennaresearch.com
user = "geoff.klassen@usask.ca"
passw = "Phi1*618ah"

# This is the default language to set images to, in case none of the language keywords are found in the image.
default_language = 'eng'    # For HappyB: ita

use_downloaded_images = True  # If False, local copies of images are not used (always fetch images from the website)

# Location of PyTesseract on local drive
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\gbk546\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # At U of S

conf_limit = 80

# Specify the list of surveys to extract URLs from
# survey_list = [happyb_baseline_survey, happyb_daily_survey]       # For HappyB (1.0)
# survey_list = [happyb2_baseline_survey, happyb2_daily_survey]     # For HappyB2.0
survey_list = [bch_survey]                                          # For Boston Children's Hospital


test_lower_bound = 1
test_upper_bound = 99999

if main_study == "HappyB2.0":
    os.chdir(
        'C:\\Users\\gbk546\\OneDrive - University of Saskatchewan\\Grad Studies\\HappyB 2.0\\OCRScript_iOS_v2')
    default_language = 'eng'
    survey_list = [happyb2_baseline_survey, happyb2_daily_survey]
elif main_study == "BCH":
    os.chdir(
        'C:\\Users\\gbk546\\OneDrive - University of Saskatchewan\\Grad Studies\\Boston Childrens Hospital')
    default_language = 'eng'
    survey_list = [bch_survey]