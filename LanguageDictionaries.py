"""This file contains the dictionaries of keywords in various languages, as well as some useful variables for time values.
The keywords are used to determine the language of the image, the date (or date range) present in the image,
the category/categories of data present in the image, and the location of the category-specific data within the image.

Current languages include:
    English
    French
    German
    Italian
Languages to add:
    Spanish

Dictionaries included:
    MONTH_ABBREVIATIONS             - used for determining the date (or date range) as it appears in the screenshot
    DATE_FORMAT                     - "

    KEYWORDS_FOR_TODAY              - used for determining the date range of the data in a given screenshot
    KEYWORD_FOR_YESTERDAY           - "
    KEYWORDS_FOR_DAYS_OF_THE_WEEK   - "
    KEYWORDS_FOR_WEEK               - "

    KEYWORD_FOR_SCREEN_TIME                 - used for determining the categories of data contained within a screenshot
    KEYWORD_FOR_LIMITATIONS                 - "
    KEYWORD_FOR_MOST_USED                   - "
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

KEYWORD_GERMAN = ['Gestern', 'Benachrichtigungen', 'Entsperrungen']
KEYWORD_ITALIAN = ['utilizzo', 'dello schermo', ' leri ', 'leri', 'notifiche', 'NOTIFICHE', 'UTILIZZATE', 'DOPO',
                   # duplicate words with differences in spaces.
                   'ricevute', 'sblocchi', 'Sblocchi', 'blocchi', 'volte', 'batteria', 'Oggi']
KEYWORD_ENGLISH = ['Screen time', 'SCREEN', 'Updated', 'Yesterday', 'yesterday', 'Today', 'today', 'received']
KEYWORD_FRENCH = ['Applications les plus', 'fois', 'Notifications', 'Hier']
# If one of the keywords above is found in a given screenshot, that screenshot is classified as being of the language of that keyword.

MONTH_ABBREVIATIONS = {'ita': ['gen', 'feb', 'mar', 'apr', 'mag',  'giu',  'lug',  'ago',  'set', 'ott', 'nov', 'dic'],
                       'eng': ['jan', 'feb', 'mar', 'apr', 'may',  'jun',  'jul',  'aug',  'sep', 'oct', 'nov', 'dec'],
                       'ger': ['jan', 'feb', 'mar', 'apr', 'mai',  'jun',  'jul',  'aug',  'sep', 'okt', 'nov', 'dez'],
                       'fra': ['jan', 'fev', 'mar', 'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec']}
# Abbreviations must be in chronological order, from January to December.

DATE_FORMAT = {'ita': [r'\d{1,2}\s?MMM'],
               'eng': [r'\d{1,2}\s?MMM', r'MMM[a-z]*\s?\d{1,2}'],
               'ger': [r'\d{1,2}\s?MMM'],
               'fra': [r'\d{1,2}\s?MMM']}
# MMM stands in for the 3-4 letter abbreviation for each month.
# The list of abbreviations for the necessary language will be subbed in as needed to create the full regex.

KEYWORDS_FOR_TODAY = {'ita': ['Oggi', 'oggi'],
                      'ger': ['Heute', 'heute'],
                      'eng': ['Today', 'today'],
                      'fra': ['Aujourd', 'aujourd']}  # Abbreviated version for French (aujourd'hui)

KEYWORD_FOR_YESTERDAY = {'ita': 'leri',  # true word is Ieri, but pytesseract usually reads the 'I' as 'l'.
                         'eng': 'Yesterday',
                         'fra': 'Hier',
                         'ger': 'Gestern'}

KEYWORDS_FOR_DAYS_OF_THE_WEEK = {'ita': ['lunedi', 'martedi', 'mercoledi', 'giovedi', 'venerdi', 'sabato', 'domenica'],
                                 'eng': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                                 'fra': ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'],
                                 'ger': ['montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag',
                                         'sonntag']}

KEYWORDS_FOR_WEEK = {'ita': ['Questa settimana', 'Media giornaliera'],
                     'eng': ['This week', 'Daily Average'],
                     'ger': ['TODO FILL THIS IN'],  # TODO Fill this in
                     'fra': ['TODO FILL THIS IN']}  # TODO Fill this in
# Keywords used to determine if screenshot contains 'week' data (instead of 'day' data)

# Keywords for headings
# Order in Dashboard: SCREEN TIME, (LIMITS), MOST USED, PICKUPS, FIRST PICKUP, FIRST USED AFTER PICKUP, NOTIFICATIONS
# The LIMITS heading only appears if the user's phone has a daily time limit set for at least one app.
KEYWORD_FOR_SCREEN_TIME = {'ita': 'TEMPO DI UTILIZZO',
                           'eng': 'SCREEN TIME',
                           'fra': 'TODO FILL THIS IN',  # TODO Fill this in
                           'ger': 'TODO FILL THIS IN'}  # TODO Fill this in

KEYWORD_FOR_LIMITATIONS = {'ita': 'LIMITAZIONI',
                           'eng': 'LIMITS',
                           'fra': 'TODO FILL THIS IN',  # TODO Fill this in
                           'ger': 'TODO FILL THIS IN'}  # TODO Fill this in

KEYWORD_FOR_MOST_USED = {'ita': 'PIU UTILIZZATE',
                         'eng': 'MOST USED',
                         'fra': 'TODO FILL THIS IN',  # TODO Fill this in
                         'ger': 'VERWENDET'}  # Real heading is AM HÄUFIGSTEN VERWENDET but VERWENDET is on its own line

KEYWORDS_FOR_PICKUPS = {'ita': ['ATTIVAZIONI SCHERMO'],
                        'eng': ['PICKUPS', 'PICK-UPS'],  # Some versions of iOS use the hyphenated form PICK-UPS
                        'fra': ['TODO FILL THIS IN'],  # TODO Fill this in
                        'ger': ['AKTIVIERUNGEN']}

KEYWORDS_FOR_FIRST_PICKUP = {'ita': ['1 attivazione schermo Totale', '1 attivazione schermo', 'schermo', 'Totale'],
                             'eng': ['First Pickup Total Pickups', 'First Pickup', 'Total Pickups'],
                             'fra': ['TODO FILL THIS IN'],  # TODO Fill this in
                             'ger': ['1 Aktivierung Aktivierungen insgesamt', '1 Aktivierung',
                                     'Aktivierungen insgesamt', 'insgesamt']}

KEYWORDS_FOR_FIRST_USED_AFTER_PICKUP = {'ita': ['PRIME APP UTILIZZATE DOPO LATTIVAZIONE',
                                                'PRIME APP UTILIZZATE DOPO', 'LATTIVAZIONE'],  # When on separate lines
                                        'eng': ['FIRST USED AFTER PICKUP', 'FIRST USED AFTER PICK UP', 'USED AFTER PICKUP'],
                                        'fra': ['TODO FILL THIS IN'],  # TODO Fill this in
                                        'ger': ['1 NUTZUNG NACH AKTIVIERUNG', 'AKTIVIERUNG']}

KEYWORDS_FOR_NOTIFICATIONS = {'ita': ['NOTIFICHE'],
                              'ger': ['BENACHRICHTIGUNGEN', 'MITTEILUNGEN'],
                              'eng': ['NOTIFICATIONS', 'NOTIFICATIONS RECEIVED'],
                              'fra': ['NOTIFICATIONS', 'NOTIFICATIONS RECUES', 'RECUES']}

KEYWORDS_FOR_HOURS_AXIS = ['00 06', '06 12', '12 18',
                           '6 12 ', '6 18 ', '42 48',
                           '00h 06h', '06h 12h', '12h 18h',
                           r'12\s?AM 6', r'6\s?AM 12', r'12\s?PM 6',
                           'AM 6AM', 'AM 12PM', 'PM 6PM',
                           '6AM 6PM', '12AM 6PM',
                           'AM PM', 'PM AM', 'PM PM', 'AM AM',
                           'mele 12', '112 118', r'0\s+.*\s+12',
                           '00 Uhr 06 Uhr', '06 Uhr 12 Uhr', '12 Uhr 18 Uhr',
                           'Uhr Uhr']  # TODO This method is a bit messy

MIN_KEYWORD = r'min|m'
HOUR_KEYWORD = 'h'
SECONDS_KEYWORD = 's'
TIME_FORMAT_STR_FOR_MINUTES = r'\d+\s?(min|m)'
TIME_FORMAT_STR_FOR_HOURS = r'\d+\s?h'
TIME_FORMAT_STR_FOR_SECONDS = r'\d+\s?s'
