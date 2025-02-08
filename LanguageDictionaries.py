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
    Spanish (two versions)

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

"""
    Language keywords
"""
GER = 'German'
ITA = 'Italian'
ENG = 'English'
FRA = 'French'

LANGUAGE_KEYWORDS = {GER: ['Gestern', 'Benachrichtigungen', 'Entsperrungen'],
                     ITA: ['utilizzo', 'dello schermo', ' leri ', 'leri', 'notifiche', 'NOTIFICHE', 'UTILIZZATE', 'DOPO',
                             # duplicate words with differences in spaces.
                             'ricevute', 'sblocchi', 'Sblocchi', 'blocchi', 'volte', 'batteria', 'Oggi'],
                     ENG: ['Screen time', 'SCREEN', 'Updated', 'Yesterday', 'yesterday', 'Today', 'today', 'received'],
                     FRA: ['Applications les plus', 'fois', 'Notifications', 'Hier']
                     }
# If one of the keywords above is found in a given screenshot, that screenshot is classified as being of the language of
# that keyword. Otherwise, the screenshot is classified as being of the default language, as set in RuntimeValues.py.

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
                      FRA: ['Aujourd', 'aujourd']}  # Abbreviated version for French (aujourd'hui)

KEYWORD_FOR_YESTERDAY = {ITA: 'leri',  # true word is Ieri, but pytesseract usually reads the 'I' as 'l'.
                         ENG: 'Yesterday',
                         FRA: 'Hier',
                         GER: 'Gestern'}

KEYWORDS_FOR_DAYS_OF_THE_WEEK = {ITA: ['lunedi', 'martedi', 'mercoledi', 'giovedi', 'venerdi', 'sabato', 'domenica'],
                                 ENG: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                                 FRA: ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'],
                                 GER: ['montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag',
                                         'sonntag']}

KEYWORDS_FOR_WEEK = {ITA: ['Questa settimana', 'Media giornaliera'],
                     ENG: ['This week', 'Daily Average'],
                     GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                     FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
# Keywords used to determine if screenshot contains 'week' data (instead of 'day' data)


"""
    iOS Heading dictionaries
"""
# Order in Dashboard: SCREEN TIME, (LIMITS), MOST USED, PICKUPS, FIRST PICKUP, FIRST USED AFTER PICKUP, NOTIFICATIONS
# The LIMITS heading only appears if the user's phone has a daily time limit set for at least one app.
KEYWORD_FOR_SCREEN_TIME = {ITA: 'TEMPO DI UTILIZZO',
                           ENG: 'SCREEN TIME',
                           FRA: 'TODO FILL THIS IN',  # TODO Fill this in
                           GER: 'TODO FILL THIS IN'}  # TODO Fill this in

KEYWORD_FOR_LIMITATIONS = {ITA: 'LIMITAZIONI',
                           ENG: 'LIMITS',
                           FRA: 'TODO FILL THIS IN',  # TODO Fill this in
                           GER: 'TODO FILL THIS IN'}  # TODO Fill this in

KEYWORD_FOR_MOST_USED = {ITA: 'PIU UTILIZZATE',
                         ENG: 'MOST USED',
                         FRA: 'TODO FILL THIS IN',  # TODO Fill this in
                         GER: 'VERWENDET'}  # Real heading is AM HÄUFIGSTEN VERWENDET but VERWENDET is on its own line

KEYWORDS_FOR_PICKUPS = {ITA: ['ATTIVAZIONI SCHERMO'],
                        ENG: ['PICKUPS', 'PICK-UPS'],  # Some versions of iOS use the hyphenated form PICK-UPS
                        FRA: ['TODO FILL THIS IN'],  # TODO Fill this in
                        GER: ['AKTIVIERUNGEN']}

KEYWORDS_FOR_FIRST_PICKUP = {ITA: ['1 attivazione schermo Totale', '1 attivazione schermo', 'schermo', 'Totale'],
                             ENG: ['First Pickup Total Pickups', 'First Pickup', 'Total Pickups'],
                             FRA: ['TODO FILL THIS IN'],  # TODO Fill this in
                             GER: ['1 Aktivierung Aktivierungen insgesamt', '1 Aktivierung',
                                     'Aktivierungen insgesamt', 'insgesamt']}

KEYWORDS_FOR_FIRST_USED_AFTER_PICKUP = {ITA: ['PRIME APP UTILIZZATE DOPO LATTIVAZIONE',
                                                'PRIME APP UTILIZZATE DOPO', 'LATTIVAZIONE'],  # When on separate lines
                                        ENG: ['FIRST USED AFTER PICKUP', 'FIRST USED AFTER PICK UP', 'USED AFTER PICKUP'],
                                        FRA: ['TODO FILL THIS IN'],  # TODO Fill this in
                                        GER: ['1 NUTZUNG NACH AKTIVIERUNG', 'AKTIVIERUNG']}

KEYWORDS_FOR_NOTIFICATIONS = {ITA: ['NOTIFICHE'],
                              GER: ['BENACHRICHTIGUNGEN', 'MITTEILUNGEN'],
                              ENG: ['NOTIFICATIONS', 'NOTIFICATIONS RECEIVED'],
                              FRA: ['NOTIFICATIONS', 'NOTIFICATIONS RECUES', 'RECUES']}

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

"""
    Variables for time values
"""
MIN_KEYWORD = r'min|m'
HOUR_KEYWORD = 'h'
SECONDS_KEYWORD = 's'
TIME_FORMAT_STR_FOR_MINUTES = r'\d+\s?(min|m)'
TIME_FORMAT_STR_FOR_HOURS = r'\d+\s?h'
TIME_FORMAT_STR_FOR_SECONDS = r'\d+\s?s'

