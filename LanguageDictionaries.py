"""This file contains the dictionaries of keywords in various languages, as well as some useful variables for time values.
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

Dictionaries included:
    MONTH_ABBREVIATIONS             - used for determining the date (or date range) as it appears in the screenshot
    DATE_FORMAT                     - "

    KEYWORDS_FOR_TODAY              - used for determining the date range of the data in a given screenshot
    KEYWORDS_FOR_YESTERDAY          - "
    KEYWORDS_FOR_DAYS_OF_THE_WEEK   - "
    KEYWORDS_FOR_WEEK               - "

    KEYWORDS_FOR_SCREEN_TIME                - used for determining the categories of data contained within a screenshot
    KEYWORDS_FOR_LIMITATIONS                - "
    KEYWORDS_FOR_MOST_USED                  - "
    KEYWORDS_FOR_PICKUPS                    - "
    KEYWORDS_FOR_FIRST_PICKUP               - "
    KEYWORDS_FOR_FIRST_USED_AFTER_PICKUP    - "
    KEYWORDS_FOR_NOTIFICATIONS              - "
    KEYWORDS_FOR_HOURS_AXIS                 - "

Variables included:
    MIN_KEYWORD                     - a regex for the abbreviations for "minutes" as it appears in time values
    HOUR_KEYWORD                    - the abbreviation for "hours" as it appears in time values
    SECONDS_KEYWORD                 - the abbreviation for "seconds" as it appears in time values
    TIME_FORMAT_STR_FOR_MINUTES     - a regex for the numerical format for a time value in minutes
    TIME_FORMAT_STR_FOR_HOURS       - a regex for the numerical format for a time value in hours
    TIME_FORMAT_STR_FOR_SECONDS     - a regex for the numerical format for a time value in seconds
"""

"""
    Language abbreviations
"""
GER = 'German'
ITA = 'Italian'
ENG = 'English'
FRA = 'French'

LANGUAGE_KEYWORDS = {GER: ['Gestern', 'Heute', 'Benachrichtigungen', 'Entsperrungen'],
                     ITA: ['Tempo di utilizzo''utilizzo', 'dello schermo', ' leri ', '\bleri\b', 'notifiche',
                           'NOTIFICHE', 'UTILIZZATE', 'DOPO', 'ricevute', 'sblocchi', 'Sblocchi', 'blocchi', 'volte',
                           'batteria', 'atteria', 'Oggi'],
                     ENG: ['Screen time', 'SCREEN', 'Updated', 'Yesterday', 'yesterday', 'Today', 'View more', 'today',
                           'received', 'Unlocks', 'unlocks'],
                     FRA: ['Applications les plus', 'fois', 'Notifications', 'Hier']
                     }
# If one of the keywords above is found in a given screenshot, that screenshot is classified as being of the language of
# that keyword. Otherwise, the screenshot is classified as being of the previously detected language for its participant
# or, if that's not available, the study's default language as set in RuntimeValues.py.

"""
    Date dictionaries
"""
MONTH_ABBREVIATIONS = {ITA: ['gen', 'feb', 'mar', 'apr', 'mag',  'giu',  'lug',  'ago',  'set', 'ott', 'nov', 'dic'],
                       ENG: ['jan', 'feb', 'mar', 'apr', 'may',  'jun',  'jul',  'aug',  'sep', 'oct', 'nov', 'dec'],
                       GER: ['jan', 'feb', 'mar', 'apr', 'mai',  'jun',  'jul',  'aug',  'sep', 'okt', 'nov', 'dez'],
                       FRA: ['jan', 'fev', 'mar', 'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec']}
# Abbreviations must be in chronological order, from January to December.

DATE_FORMAT = {ITA: [r'\d{1,2}\s?MMM'],
               ENG: [r'\d{1,2}\s?MMM', r'MMM[a-z]*\s?\d{1,2}'],
               GER: [r'\d{1,2}\s?MMM'],
               FRA: [r'\d{1,2}\s?MMM']}
# MMM stands in for the 3-4 letter abbreviation for each month.
# The list of abbreviations for the necessary language will be subbed in as needed to create the full regex.

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

KEYWORDS_FOR_DAYS_OF_THE_WEEK = {ITA: ['lunedi',  'martedi', 'mercoledi',    'giovedi',  'venerdi',   'sabato', 'domenica'],
                                 ENG: ['monday',  'tuesday', 'wednesday',   'thursday',   'friday', 'saturday',   'sunday'],
                                 FRA: [ 'lundi',    'mardi',  'mercredi',      'jeudi', 'vendredi',   'samedi', 'dimanche'],
                                 GER: ['montag', 'dienstag',  'mittwoch', 'donnerstag',  'freitag',  'samstag',  'sonntag']}

KEYWORDS_FOR_WEEK = {ITA: ['Questa settimana', 'Media giornaliera'],
                     ENG: ['This week', 'Daily Average'],
                     GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                     FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
# Keywords used to determine if screenshot contains 'week' data (instead of 'day' data)



