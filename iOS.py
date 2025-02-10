"""This file contains iOS-specific dictionaries, functions, and variables."""

"""
    Language abbreviations
"""
GER = 'German'
ITA = 'Italian'
ENG = 'English'
FRA = 'French'

"""
    iOS-Specific dictionaries
    
    Some keyword dictionaries are iOS-specific. For example, iOS has a LIMITATIONS heading while Android doesn't.
    Conversely, Android has a dictionary for HOURS, but iOS doesn't need one because all iOS phones use 'h' for hours.
"""
# Order in Dashboard: SCREEN TIME, (LIMITS), MOST USED, PICKUPS, FIRST PICKUP, FIRST USED AFTER PICKUP, NOTIFICATIONS
# The LIMITS heading only appears if the user's phone has a daily time limit set for at least one app.
KEYWORDS_FOR_SCREEN_TIME = {ITA: ['TEMPO DI UTILIZZO'],
                            ENG: ['SCREEN TIME'],
                            FRA: ["TEMPS D'ECRAN"],
                            GER: ['BILDSCHIRMZEIT']}

KEYWORDS_FOR_LIMITATIONS = {ITA: ['LIMITAZIONI'],
                            ENG: ['LIMITS'],
                            FRA: ['TODO FILL THIS IN'],  # TODO Fill this in
                            GER: ['TODO FILL THIS IN']}  # TODO Fill this in

KEYWORDS_FOR_MOST_USED = {ITA: ['PIU UTILIZZATE'],
                          ENG: ['MOST USED'],
                          FRA: ['LES PLUS UTILISEES'],  # LES PLUS UTILISÉES
                          GER: ['VERWENDET']}  # Real heading is AM HÄUFIGSTEN VERWENDET but VERWENDET is on its own line

KEYWORDS_FOR_PICKUPS = {ITA: ['ATTIVAZIONI SCHERMO'],
                        ENG: ['PICKUPS', 'PICK-UPS'],  # Some versions of iOS use the hyphenated form PICK-UPS
                        FRA: ['PRISES EN MAIN'],
                        GER: ['AKTIVIERUNGEN']}

KEYWORDS_FOR_FIRST_PICKUP = {ITA: ['1 attivazione schermo Totale', '1 attivazione schermo', 'schermo', 'Totale'],
                             ENG: ['First Pickup Total Pickups', 'First Pickup', 'Total Pickups'],
                             FRA: ['Premiere prise en main', 'Total des prises en main'],
                             GER: ['1 Aktivierung Aktivierungen insgesamt', '1 Aktivierung',
                                   'Aktivierungen insgesamt', 'insgesamt']}

KEYWORDS_FOR_FIRST_USED_AFTER_PICKUP = {ITA: ['PRIME APP UTILIZZATE DOPO LATTIVAZIONE',
                                              'PRIME APP UTILIZZATE DOPO', 'LATTIVAZIONE'],  # When on separate lines
                                        ENG: ['FIRST USED AFTER PICKUP', 'FIRST USED AFTER PICK UP',
                                              'USED AFTER PICKUP'],
                                        FRA: ['PREMIERE APP UTILISEE'],  # PREMIÈRE APP UTILISÉE
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

# Variables for iOS time values
# Even though the words for 'hours', 'minutes', and 'seconds' differ by language, iOS uses h/min/m/s for all languages.
MIN_KEYWORD = r'min|m'
HOUR_KEYWORD = 'h'
SECONDS_KEYWORD = 's'
TIME_FORMAT_STR_FOR_MINUTES = r'\d+\s?(min|m)'
TIME_FORMAT_STR_FOR_HOURS = r'\d+\s?h'
TIME_FORMAT_STR_FOR_SECONDS = r'\d+\s?s'


misread_time_format = r'^[\d|t]+\s?[hn]$|^[\d|t]+\s?[hn]\s?[\d|tA]+\s?(min|m)$|^.{0,2}\s?[0-9AIt]+\s?(min|m)$|\d+\s?s$'
misread_number_format = r'^[0-9A]+$'
misread_time_or_number_format = '|'.join([misread_time_format, misread_number_format])


def main():
    print("I am now in iOS.py")