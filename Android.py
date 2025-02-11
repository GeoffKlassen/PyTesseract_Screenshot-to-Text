"""This file contains Android-specific dictionaries, functions, and variables."""

from ConvenienceVariables import *

"""
    Android-Specific dictionaries

    Some keyword dictionaries are Android-specific. For example, Android has a dictionary for HOURS, but iOS doesn't need
    one because iOS uses 'h' for hours in all languages. Conversely, iOS has a LIMITATIONS heading while Android doesn't.
"""

DATE_RANGE_FORMAT = {ITA: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-d{1,2}\s?MMM'],
                     ENG: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'MMM\s?\d{1,2}-\d{1,2}',
                           r'MMM*\s?\d{1,2}-MMM*\s?\d{1,2}'],
                     GER: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-\d{1,2}\s?MMM'],
                     FRA: [r'\d{1,2}-\d{1,2}\s?MMM',
                           r'\d{1,2}\s?MMM[a-z]*-\d{1,2}\s?MMM']}

# hhh/HHH stands for the short/long format for hours; mmm/MMM stands for the short/long format for minutes.
# The list of abbreviations for the necessary language will be subbed in as needed to create the full regex.
TIME_FORMATS = [r'^[01ilLT]?[0-9aAilLStT]\s?hhh\s?[0-5aAilLT]?[0-9aAilLStT]\s?mmm$',  # Format for ## hr ## min
                r'^[01ilLT]?[0-9aAilLStT]\s?HHH$',                                    # Format for ## hours
                r'^[0-5aAilT]?[0-9AlLOStT]\s?MMM$']                                   # Format for ## minutes
# Sometimes pytesseract mistakes digits for A, I, L, S, or T (e.g.  A = 4,   I/L/T = 1,   S = 5)
# Including these letters in the regex ensures that times with a misread digit still match a time format.

KEYWORDS_FOR_HOURS = {'ita': ['ore', 'ora'],
                      'eng': ['hours', 'hour'],
                      'ger': ['Stunden', 'Stunde'],
                      'fra': ['heures', 'heure']}
KEYWORDS_FOR_MINUTES = {'ita': ['minuti', 'minuto'],
                        'eng': ['minutes', 'minute'],
                        'ger': ['Minuten', 'Minute'],
                        'fra': ['minutes', 'minute']}
KEYWORD_FOR_HR = {'ita': 'h e',
                  'eng': 'hr',
                  'ger': 'Std',
                  'fra': 'h et'}
KEYWORD_FOR_MIN = {'ita': 'min',
                   'eng': 'min',
                   'ger': 'Min',
                   'fra': 'min'}
H = 'h'
MIN = '(mi?n?)'

KEYWORDS_FOR_2018_SCREENTIME = {'ita': ['DURATA SCHERMO', 'DURATA SCHERMO Altro'],
                                'eng': ['TODO FILL THIS IN'],
                                'ger': ['TODO FILL THIS IN'],
                                'fra': ['TODO FILL THIS IN']}
KEYWORDS_FOR_2018_MOST_USED = {'ita': ['UTILIZZO APP'],
                               'eng': ['TODO FILL THIS IN'],
                               'ger': ['TODO FILL THIS IN'],
                               'fra': ['TODO FILL THIS IN']}
KEYWORDS_FOR_2018_UNLOCKS = {'ita': ['SBLOCCHI'],
                             'eng': ['TODO FILL THIS IN'],
                             'ger': ['TODO FILL THIS IN'],
                             'fra': ['TODO FILL THIS IN']}

KEYWORDS_FOR_SCREEN_TIME = {'ita': ['Tempo di utilizzo', 'Tempo di utilizzo dello schermo',
                                    'DURATA SCHERMO', 'DURATA SCHERMO Altro'],
                            'eng': ['Screen time'],
                            'ger': ['TODO: FILL THIS IN'],  # TODO Fill this in
                            'fra': ['Temps dutilisation des écrans', 'Temps decran']}
# Actual phrases are "Temps d'utilisation des ecrans" and "Temps d'ecran"
KEYWORDS_FOR_MOST_USED_APPS = {'ita': ['Applicazioni piu utilizzate', 'UTILIZZO APP', 'Applicazioni utilizzate'],
                               'eng': ['Most used apps'],
                               'ger': ['Geratenutzungsdauer'],
                               'fra': ['Applications les plus', 'Applications les plus utilisees']}
KEYWORDS_FOR_NOTIFICATIONS_RECEIVED = {'ita': ['Notifiche ricevute'],
                                       'eng': ['Notifications received'],
                                       'ger': ['TODO FILL THIS IN'],  # TODO Fill this in
                                       'fra': ['TODO FILL THIS IN']}  # TODO Fill this in
KEYWORDS_FOR_MOST_NOTIFICATIONS = {'ita': ['Piu notifiche'],
                                   'eng': ['Most notifications'],
                                   'ger': ['TODO FILL THIS IN'],  # TODO Fill this in
                                   'fra': ['Notifications les plus nombreuses', 'Notifications les', 'plus nombreuses']}
KEYWORDS_FOR_TIMES_OPENED = {'ita': ['Numero di aperture', 'Sblocchi', 'SBLOCCHI'],
                             'eng': ['Times opened', 'Unlocks'],
                             'ger': ['Wie oft geoffnet'],  # TODO Fill this in
                             'fra': ['Nombre douvertures']}  # Actual phrase is Nombre d'ouvertures
KEYWORDS_FOR_VIEW_MORE = {'ita': ['Visualizza altro'],
                          'eng': ['View more', 'View all'],
                          'ger': ['TODO FILL THIS IN'],
                          'fra': ['Afficher plus']}

GOOGLE_SCREENTIME_FORMATS = {'ita': ['# ora', '# h e # min', '# minuti', '1 minuto', 'Meno di 1 minuto'],
                             'eng': ['# hours', '# hr # min', '# minutes', '1 minute', 'Less than 1 minute'],
                             'ger': ['# Stunde', '# Std # Min', '# Minuten', '1 Minute', 'Weniger als 1 Minute'],
                             'fra': ['# heures', '# h et # min', '# minutes', '1 minute', 'Moins de 1 minute']}
GOOGLE_NOTIFICATIONS_FORMATS = {'ita': ['# notifiche'],
                                'eng': ['# notifications', '# notification'],
                                'ger': ['# Benachrichtigungen'],
                                'fra': ['# notifications', '# notification']}
GOOGLE_UNLOCKS_FORMATS = {'ita': ['# sblocchi', '# aperture'],
                          'eng': ['# unlocks', 'Opened # times'],
                          'ger': ['# Entsperrungen', '# Mal geoffnet'],
                          'fra': ['Déverrouillé # fois', 'Ouverte # fois']}
SAMSUNG_NOTIFICATIONS_FORMATS = {'ita': ['# notifiche ricevute', "# ricevute"],
                                 'eng': ['# notifications', '# notification', '# received'],
                                 # TODO Should this include '# notifications received'?
                                 'ger': ['# Benachrichtigungen'],  # TODO Make sure this is correct
                                 'fra': ['# notifications', '# notification']}  # TODO Fill this in
SAMSUNG_UNLOCKS_FORMAT = {'ita': ['# volte', '# in totale'],
                          'eng': ['# times'],
                          'ger': [''],  # TODO Fill this in
                          'fra': ['']}  # TODO Fill this in

# YOU_CAN_SET_DAILY_TIMERS = {'ita': 'Imposta i timer per le app',
#                             'eng': 'You can set daily timers',
#                             'ger': 'Timer fur Apps einrichten',
#                             'fra': ''}  # TODO Fill this in
# "You can set daily timers" is a tip box that appears in the Google version of Dashboard, until a user clears it.

REST_OF_THE_DAY = {'ita': 'giornata',  # full phrase is 'resto della giornata' but 'giornata' is sometimes its own line
                   'eng': 'rest of the day',
                   'ger': 'Rest des Tages pausiert',
                   'fra': ''}
# "rest of the day" is the last text in the dialogue box for "You can set daily timers".

SHOW_SITES_YOU_VISIT = {'ita': 'Mostra i siti visitati',
                        'eng': 'Show sites that you visit',  # TODO: used to be 'Show sites you visit' for HappyB - check if this is the correct phrase
                        'ger': 'Besuchte Websites anzeigen',
                        'fra': ''}  # TODO Fill this in
# "Show sites you visit" can appear in the Google version of Dashboard, under the Chrome app (if it's in the app list).
# Thus, it can be mistaken for an app name, so we need to ignore it.

KEYWORDS_FOR_UNRELATED_SCREENSHOTS = {'ita': ['USO BATTERIA', 'Benessere digitale'],
                                      'eng': ['BATTERY USE', 'BATTERY USAGE', 'Digital wellbeing'],
                                      'ger': ['TODO FILL THIS IN'],
                                      'fra': ['TODO FILL THIS IN']}
# Some screenshots show only Battery Usage info; these screenshots do not contain any of the requested info.

# Determine if the screenshot contains unrelated data -- if so, skip it
# Determine date and day in screenshot
# Get headings from text_df
# Determine the version of Android
# Determine which type of data is visible on screen
#   Might be able to systematically search for all 3 kinds of data
# Extract the daily total (and confidence)
# Crop the image to the app-specific region
# Extract app-specific data
# Sort the app-specific data into app names and app usage numbers
# Collect some review-oriented statistics on the screenshot
# Put the data from the screenshot into the current screenshot's collection
# Put the data from the screenshot into the master CSV for all screenshots
# Check if data already exist for this user & date
#   If so, determine how to combine the existing data and current data so they fit together properly


def main():
    print("I am now in Android.py")
