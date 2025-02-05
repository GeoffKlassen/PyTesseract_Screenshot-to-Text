from ScreenshotClass import Screenshot
from LanguageDictionaries import *

"""Variables defined for convenience"""
TODAY = 'today'
YESTERDAY = 'yesterday'
DAY_OF_THE_WEEK = 'weekday'
WEEK = 'week'

test_screenshot = Screenshot("hi")

list_of_screenshots = []
print("doing it")
list_of_screenshots.append(test_screenshot)
for ss in list_of_screenshots:
    print(ss)

list_of_screenshots[0].url = "changing URL"
print(list_of_screenshots[0])