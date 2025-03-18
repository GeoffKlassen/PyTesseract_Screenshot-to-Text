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

LANGUAGE_KEYWORDS = {ENG: ['Screen time', 'SCREEN', 'Updated', 'Yesterday', 'yesterday', 'Today', 'View more', 'today',
                           'received', 'Unlocks', 'unlocks', 'Digital Wellbeing', 'View all', 'Show sites',
                           'Set limits for', 'Set timers for', 'Activity details', 'Activity history'],
                     ITA: ['Tempo di utilizzo''utilizzo', 'dello schermo', ' leri ', '\bleri\b', 'notifiche',
                           'NOTIFICHE', 'UTILIZZATE', 'DOPO', 'ricevute', 'sblocchi', 'Sblocchi', 'blocchi', 'volte',
                           'batteria', 'atteria', 'Oggi'],
                     GER: ['Gestern', 'Heute', 'Benachrichtigungen', 'Entsperrungen'],
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
MONTH_ABBREVIATIONS = {ENG: ['jan', 'feb', 'mar', 'apr', 'may',  'jun',  'jul',  'aug',  'sep', 'oct', 'nov', 'dec'],
                       ITA: ['gen', 'feb', 'mar', 'apr', 'mag',  'giu',  'lug',  'ago',  'set', 'ott', 'nov', 'dic'],
                       GER: ['jan', 'feb', 'mar', 'apr', 'mai',  'jun',  'jul',  'aug',  'sep', 'okt', 'nov', 'dez'],
                       FRA: ['jan', 'fev', 'mar', 'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec']}
# Abbreviations must be in chronological order, from January to December.
ENGLISH_MONTHS = MONTH_ABBREVIATIONS[ENG]
MONTH_MAPPING = {mon: ENGLISH_MONTHS.index(mon) + 1 for mon in ENGLISH_MONTHS}  # 1:January, 2:February, 3:March, etc.

DATE_FORMAT = {ENG: [r'\d{1,2}\s?MMM', r'MMM[a-z]*\s?\d{1,2}'],
               ITA: [r'\d{1,2}\s?MMM'],
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
KEYWORDS_FOR_TODAY = {
    ENG: ['Today', 'today'],
    ITA: ['Oggi', 'oggi'],
    GER: ['Heute', 'heute'],
    FRA: ['Aujourd', 'aujourd']
}  # Abbreviated version for aujourd'hui

KEYWORDS_FOR_YESTERDAY = {
    ENG: ['Yesterday'],
    ITA: ['leri'],  # Actual word is "Ieri", but pytesseract usually reads the 'I' as 'l'
    GER: ['Gestern'],
    FRA: ['Hier']
}

KEYWORDS_FOR_WEEKDAY_NAMES = {
    ENG: ['monday', 'tuesday',  'wednesday', 'thursday',   'friday',   'saturday', 'sunday',
          'Mon',    'Tue',      'Wed',       'Thu',        'Fri',      'Sat',      'Sun'     ],
    ITA: ['lunedi', 'martedi',  'mercoledi', 'giovedi',    'venerdi',  'sabato',   'domenica'],
    GER: ['montag', 'dienstag', 'mittwoch',  'donnerstag', 'freitag',  'samstag',  'sonntag' ],
    FRA: ['lundi',  'mardi',    'mercredi',  'jeudi',      'vendredi', 'samedi',   'dimanche']
}

KEYWORDS_FOR_WEEK = {
    ENG: ['This week', 'Daily Average'],
    ITA: ['Questa settimana', 'Media giornaliera'],
    GER: ['TODO FILL THIS IN'],  # TODO Fill this in
    FRA: ['TODO FILL THIS IN']   # TODO Fill this in
}
# Keywords used to determine if screenshot contains 'week' data (instead of 'day' data)

KEYWORDS_FOR_DAY_BEFORE_YESTERDAY = {
    ENG: ['English has no word for the day before yesterday'],  # Can't be empty string
    ITA: ['Laltro ieri'],  # Actual phrase is "L'altro ieri" but apostrophes (') are ignored in the initial scan
    GER: ['Vorgestern'],
    FRA: ['Avant-hier']
}

DAY_ABBREVIATIONS = {  # TODO Verify these!
    ENG: {"M", "T", "W", "F", "S", "w", "s",  # Single letters
          "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",  # 3 letters
          "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",  # Full words
          "SS"},  # Common misreadings

    ITA: {"L", "M", "G", "V", "S", "D", "s",                                                 # Single letters
          "lun",    "mar",     "mer",       "gio",     "ven",     "sab",    "dom",           # 3 letters
          "lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato", "domenica"},     # Full words

    GER: {"M", "D", "F", "S", "s",                                                           # Single letters
          "Mo.",    "Di.",      "Mi.",      "Do.",        "Fr.",     "Sa.",     "So.",       # 2 letters plus period
          "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"},  # Full words

    FRA: {"L", "M", "J", "V", "S", "D", "s",                                                 # Single letters
          "lun",   "mar",   "mer",      "jeu",   "ven",      "sam",    "dim",                # 3 letters
          "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"}           # Full words
}
# Used to find rows of text that are 'Day' rows (i.e. contain at least 3 of these abbreviation words)

KEYWORDS_FOR_UNRELATED_SCREENSHOTS = {ITA: ['USO BATTERIA', 'Benessere digitale'],
                                      ENG: ['BATTERY USE', 'BATTERY USAGE',
                                            'Digital wellbeing', 'Digital Wellbeing & parental',
                                            'No limit', 'Manage notifications', 'Weekly report',
                                            'TAKE PHOTO'],  # 'TAKE PHOTO' appears in each Avicenna study question that
                                                            # asks participants to upload a screenshot
                                      GER: ['TODO FILL THIS IN'],
                                      FRA: ['TODO FILL THIS IN']}
# Some screenshots show only Battery Usage info; these screenshots do not contain any of the requested info.

