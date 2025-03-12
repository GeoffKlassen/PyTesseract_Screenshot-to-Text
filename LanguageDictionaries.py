"""This file contains the dictionaries of keywords in various languages, as well as some useful variables for time values.
There are more dictionaries specific to each device OS (Android, iOS) included in the respective .py files.

The dictionaries are used to determine:
    the language of the image,
    the date (or date range) present in the image,
    the category/categories of data present in the image, and
    the location of the category-specific data within the image.

Current languages include:
    English
    French
    German
    Italian
Languages to add:
    Spanish (two versions?)
    Portuguese

Dictionaries included:
    MONTH_ABBREVIATIONS             - used for determining the date (or date range) as it appears in the image
    DATE_FORMAT                     - "

    KEYWORDS_FOR_TODAY                  - used for determining the date range of the data in a given image
    KEYWORDS_FOR_YESTERDAY              - "
    KEYWORDS_FOR_DAYS_OF_THE_WEEK       - "
    KEYWORDS_FOR_WEEK                   - "
    KEYWORDS_FOR_DAY_BEFORE_YESTERDAY   - "

    DAY_ABBREVIATIONS       - used for finding rows of text in the image that contain a bunch of day names/abbreviations
"""

from ConvenienceVariables import *

LANGUAGE_KEYWORDS = {GER: ['Gestern', 'Heute', 'Benachrichtigungen', 'Entsperrungen'],
                     ITA: ['Tempo di utilizzo''utilizzo', 'dello schermo', ' leri ', '\bleri\b', 'notifiche',
                           'NOTIFICHE', 'UTILIZZATE', 'DOPO', 'ricevute', 'sblocchi', 'Sblocchi', 'blocchi', 'volte',
                           'batteria', 'atteria', 'Oggi'],
                     ENG: ['Screen time', 'SCREEN', 'Updated', 'Yesterday', 'yesterday', 'Today', 'View more', 'today',
                           'received', 'Unlocks', 'unlocks', 'Digital Wellbeing', 'View all', 'Show sites',
                           'Set limits for', 'Set timers for', 'Activity details', 'Activity history'],
                     FRA: ['Applications les plus', 'fois', 'Hier', 'Deverrouillages', 'numerique', 'AUJOURDHUI']
                     # French used to include 'Notifications', but it can cause English images to be misclassified as French
                     }
""" If one of the keywords above is found in a given screenshot, that screenshot is classified as being of the language
of that keyword. Otherwise, the screenshot is classified as being of the previously detected language for its participant
or, if that's not available, the study's default language as set in RuntimeValues.py.

Note: The strings found in the image must match these keywords EXACTLY (no levenshtein_distance applied), which is why
there are several 'misspellings' included. The reason for the exact match is that some of the keywords are quite short,
such as 'today', 'volte', 'Oggi', 'fois', and 'Hier', so we don't want any misread words or other strings to give us
false positives. """


"""
    Date dictionaries
"""
MONTH_ABBREVIATIONS = {ITA: ['gen', 'feb', 'mar', 'apr', 'mag',  'giu',  'lug',  'ago',  'set', 'ott', 'nov', 'dic'],
                       ENG: ['jan', 'feb', 'mar', 'apr', 'may',  'jun',  'jul',  'aug',  'sep', 'oct', 'nov', 'dec'],
                       GER: ['jan', 'feb', 'mar', 'apr', 'mai',  'jun',  'jul',  'aug',  'sep', 'okt', 'nov', 'dez'],
                       FRA: ['jan', 'fev', 'mar', 'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec']}
# Abbreviations must be in chronological order, from January to December.
ENGLISH_MONTHS = MONTH_ABBREVIATIONS[ENG]
MONTH_MAPPING = {mon: ENGLISH_MONTHS.index(mon) + 1 for mon in ENGLISH_MONTHS}  # 1:January, 2:February, 3:March, etc.

DATE_FORMAT = {ITA: [r'\d{1,2}\s?MMM'],
               ENG: [r'\d{1,2}\s?MMM', r'MMM[a-z]*\s?\d{1,2}'],
               GER: [r'\d{1,2}\s?MMM'],
               FRA: [r'\d{1,2}\s?MMM']}
DATE_RANGE_FORMAT = {ITA: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-d{1,2}\s?MMM'],
                     ENG: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'MMM\s?\d{1,2}-\d{1,2}',
                           r'MMM*\s?\d{1,2}-MMM*\s?\d{1,2}'],
                     GER: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-\d{1,2}\s?MMM'],
                     FRA: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-\d{1,2}\s?MMM']}
# MMM stands in for the 3-4 letter abbreviation for each month.
# The list of abbreviations for the necessary language will be substituted in as needed to create the full regex.

"""
    Day/week dictionaries
"""
KEYWORDS_FOR_TODAY = {ITA: ['Oggi', 'oggi'],
                      GER: ['Heute', 'heute'],
                      ENG: ['Today', 'today'],
                      FRA: ['Aujourd', 'aujourd']}  # Abbreviated version for aujourd'hui

KEYWORDS_FOR_YESTERDAY = {ITA: ['leri'],  # True word is Ieri, but pytesseract usually reads the 'I' as 'l'.
                          ENG: ['Yesterday'],
                          FRA: ['Hier'],
                          GER: ['Gestern']}

KEYWORDS_FOR_WEEKDAY_NAMES = {ITA: ['lunedi',  'martedi', 'mercoledi',    'giovedi',  'venerdi',   'sabato', 'domenica'],
                              ENG: ['monday',  'tuesday', 'wednesday',   'thursday',   'friday', 'saturday',   'sunday'],
                              FRA: [ 'lundi',    'mardi',  'mercredi',      'jeudi', 'vendredi',   'samedi', 'dimanche'],
                              GER: ['montag', 'dienstag',  'mittwoch', 'donnerstag',  'freitag',  'samstag',  'sonntag']}

KEYWORDS_FOR_WEEK = {ITA: ['Questa settimana', 'Media giornaliera'],
                     ENG: ['This week', 'Daily Average'],
                     GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                     FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
# Keywords used to determine if screenshot contains 'week' data (instead of 'day' data)

KEYWORDS_FOR_DAY_BEFORE_YESTERDAY = {ITA: ['Laltro ieri'],  # Actual phrase is L'altro ieri
                                     ENG: ['English has no word for the day before yesterday'],  # Can't be empty string
                                     GER: ['Vorgestern'],
                                     FRA: ['Avant-hier']}

DAY_ABBREVIATIONS = {ITA: {"TODO FILL THIS IN"},
                     ENG: {"M", "T", "W", "F", "S", "w", "s",
                           "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
                           "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"},
                     GER: {"TODO FILL THIS IN"},
                     FRA: {"TODO FILL THIS IN"}
                     }
# Used to find rows of text that are 'Day' rows (i.e. contain at least 3 of these abbreviation words)