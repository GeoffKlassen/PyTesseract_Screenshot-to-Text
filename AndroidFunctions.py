"""This file contains Android-specific dictionaries, functions, and variables."""
import numpy as np

import OCRScript_v3
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

KEYWORDS_FOR_HOURS = {ITA: ['ore', 'ora'],
                      ENG: ['hours', 'hour'],
                      GER: ['Stunden', 'Stunde'],
                      FRA: ['heures', 'heure']}
KEYWORDS_FOR_MINUTES = {ITA: ['minuti', 'minuto'],
                        ENG: ['minutes', 'minute'],
                        GER: ['Minuten', 'Minute'],
                        FRA: ['minutes', 'minute']}
KEYWORD_FOR_HR = {ITA: 'h e',
                  ENG: 'hr',
                  GER: 'Std',
                  FRA: 'h et'}
KEYWORD_FOR_MIN = {ITA: 'min',
                   ENG: 'min',
                   GER: 'Min',
                   FRA: 'min'}
H = 'h'
MIN = '(mi?n?)'

KEYWORDS_FOR_2018_SCREENTIME = {ITA: ['DURATA SCHERMO', 'DURATA SCHERMO Altro'],
                                ENG: ['TODO FILL THIS IN'],
                                GER: ['TODO FILL THIS IN'],
                                FRA: ['TODO FILL THIS IN']}
KEYWORDS_FOR_2018_MOST_USED = {ITA: ['UTILIZZO APP'],
                               ENG: ['TODO FILL THIS IN'],
                               GER: ['TODO FILL THIS IN'],
                               FRA: ['TODO FILL THIS IN']}
KEYWORDS_FOR_2018_UNLOCKS = {ITA: ['SBLOCCHI'],
                             ENG: ['TODO FILL THIS IN'],
                             GER: ['TODO FILL THIS IN'],
                             FRA: ['TODO FILL THIS IN']}

KEYWORDS_FOR_SCREEN_TIME = {ITA: ['Tempo di utilizzo', 'Tempo di utilizzo dello schermo',
                                    'DURATA SCHERMO', 'DURATA SCHERMO Altro'],
                            ENG: ['Screen time'],
                            GER: ['TODO: FILL THIS IN'],  # TODO Fill this in
                            FRA: ['Temps dutilisation des écrans', 'Temps decran']}
# Actual phrases are "Temps d'utilisation des ecrans" and "Temps d'ecran"
KEYWORDS_FOR_MOST_USED_APPS = {ITA: ['Applicazioni piu utilizzate', 'UTILIZZO APP', 'Applicazioni utilizzate'],
                               ENG: ['Most used apps'],
                               GER: ['Geratenutzungsdauer'],
                               FRA: ['Applications les plus', 'Applications les plus utilisees']}
KEYWORDS_FOR_NOTIFICATIONS_RECEIVED = {ITA: ['Notifiche ricevute'],
                                       ENG: ['Notifications received'],
                                       GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                                       FRA: ['TODO FILL THIS IN']}  # TODO Fill this in
KEYWORDS_FOR_MOST_NOTIFICATIONS = {ITA: ['Piu notifiche'],
                                   ENG: ['Most notifications'],
                                   GER: ['TODO FILL THIS IN'],  # TODO Fill this in
                                   FRA: ['Notifications les plus nombreuses', 'Notifications les', 'plus nombreuses']}
KEYWORDS_FOR_TIMES_OPENED = {ITA: ['Numero di aperture', 'Sblocchi', 'SBLOCCHI'],
                             ENG: ['Times opened', 'Unlocks'],
                             GER: ['Wie oft geoffnet'],  # TODO Fill this in
                             FRA: ['Nombre douvertures']}  # Actual phrase is Nombre d'ouvertures
KEYWORDS_FOR_VIEW_MORE = {ITA: ['Visualizza altro'],
                          ENG: ['View more', 'View all'],
                          GER: ['TODO FILL THIS IN'],
                          FRA: ['Afficher plus']}

GOOGLE_SCREENTIME_FORMATS = {ITA: ['# ora', '# h e # min', '# minuti', '1 minuto', 'Meno di 1 minuto'],
                             ENG: ['# hours', '# hr # min', '# minutes', '1 minute', 'Less than 1 minute'],
                             GER: ['# Stunde', '# Std # Min', '# Minuten', '1 Minute', 'Weniger als 1 Minute'],
                             FRA: ['# heures', '# h et # min', '# minutes', '1 minute', 'Moins de 1 minute']}
GOOGLE_NOTIFICATIONS_FORMATS = {ITA: ['# notifiche'],
                                ENG: ['# notifications', '# notification'],
                                GER: ['# Benachrichtigungen'],
                                FRA: ['# notifications', '# notification']}
GOOGLE_UNLOCKS_FORMATS = {ITA: ['# sblocchi', '# aperture'],
                          ENG: ['# unlocks', 'Opened # times'],
                          GER: ['# Entsperrungen', '# Mal geoffnet'],
                          FRA: ['Déverrouillé # fois', 'Ouverte # fois']}
SAMSUNG_NOTIFICATIONS_FORMATS = {ITA: ['# notifiche ricevute', "# ricevute"],
                                 ENG: ['# notifications', '# notification', '# received'],
                                 # TODO Should this include '# notifications received'?
                                 GER: ['# Benachrichtigungen'],  # TODO Make sure this is correct
                                 FRA: ['# notifications', '# notification']}  # TODO Fill this in
SAMSUNG_UNLOCKS_FORMAT = {ITA: ['# volte', '# in totale'],
                          ENG: ['# times'],
                          GER: [''],  # TODO Fill this in
                          FRA: ['']}  # TODO Fill this in

# YOU_CAN_SET_DAILY_TIMERS = {ITA: 'Imposta i timer per le app',
#                             ENG: 'You can set daily timers',
#                             GER: 'Timer fur Apps einrichten',
#                             FRA: ''}  # TODO Fill this in
# "You can set daily timers" is a tip box that appears in the Google version of Dashboard, until a user clears it.

REST_OF_THE_DAY = {ITA: 'giornata',  # full phrase is 'resto della giornata' but 'giornata' is sometimes its own line
                   ENG: 'rest of the day',
                   GER: 'Rest des Tages pausiert',
                   FRA: ''}
# "rest of the day" is the last text in the dialogue box for "You can set daily timers".

SHOW_SITES_YOU_VISIT = {ITA: 'Mostra i siti visitati',
                        ENG: 'Show sites that you visit',  # TODO: used to be 'Show sites you visit' for HappyB - check if this is the correct phrase
                        GER: 'Besuchte Websites anzeigen',
                        FRA: ''}  # TODO Fill this in
# "Show sites you visit" can appear in the Google version of Dashboard, under the Chrome app (if it's in the app list).
# Thus, it can be mistaken for an app name, so we need to ignore it.

KEYWORDS_FOR_UNRELATED_SCREENSHOTS = {ITA: ['USO BATTERIA', 'Benessere digitale'],
                                      ENG: ['BATTERY USE', 'BATTERY USAGE', 'Digital wellbeing'],
                                      GER: ['TODO FILL THIS IN'],
                                      FRA: ['TODO FILL THIS IN']}
# Some screenshots show only Battery Usage info; these screenshots do not contain any of the requested info.
SCREENTIME = 'screentime'
UNLOCKS = 'unlocks'
NOTIFICATIONS = 'notifications'

HEADING_COLUMN = 'heading'
SCREENTIME_HEADING = SCREENTIME
TOTAL_SCREENTIME = 'total screentime'
# LIMITS_HEADING = 'limits'
MOST_USED_HEADING = 'most used'
NOTIFICATIONS_HEADING = NOTIFICATIONS
TOTAL_NOTIFICATIONS = 'total notifications'
MOST_NOTIFICATIONS_HEADING = 'most notifications'
UNLOCKS_HEADING = UNLOCKS
TOTAL_UNLOCKS = 'total unlocks'
DAY_NAME_HEADING = 'day name'
DATE_HEADING = 'date'
VIEW_MORE_HEADING = 'view more'

OLD_SCREENTIME_HEADING = '2018 screentime'
OLD_MOST_USED_HEADING = '2018 most used'
OLD_UNLOCKS_HEADING = '2018 unlocks'

GOOGLE = 'google'
_2018 = '2018'
SAMSUNG_2021 = 'samsung 2021'
SAMSUNG_2024 = 'samsung 2024'


def screenshot_contains_unrelated_data(ss):
    text_df = ss.text
    img_lang = ss.language
    print(KEYWORDS_FOR_UNRELATED_SCREENSHOTS[img_lang])
    moe = int(np.log(min(len(k) for k in KEYWORDS_FOR_UNRELATED_SCREENSHOTS[img_lang]))) + 1  # margin of error for text
    #          (number of characters two strings can differ by and still be considered the same text)
    if any(text_df['text'].apply(lambda x: min(OCRScript_v3.levenshtein_distance(x[0:len(key)], key)
                                               for key in KEYWORDS_FOR_UNRELATED_SCREENSHOTS[img_lang])) < moe):
        print("Unrelated screenshot")
        # One of the rows of text_df starts with one of the keywords for unrelated screenshots
        return True
    print("Screenshot is indeed relevant. Hooray!")
    return False

def main():
    print("I am now in AndroidFunctions.py")
