import pandas as pd

from RuntimeValues import *
import OCRScript_v3
import iOSFunctions

# ANDROID = ConvenienceVariables.ANDROID
# SCREENTIME = ConvenienceVariables.SCREENTIME
# PICKUPS = ConvenienceVariables.PICKUPS
# NOTIFICATIONS = ConvenienceVariables.NOTIFICATIONS
# UNLOCKS = ConvenienceVariables.UNLOCKS
# MAX_APPS = RuntimeValues.max_apps_per_category
# NO_TEXT = ConvenienceVariables.NO_TEXT
# NO_NUMBER = ConvenienceVariables.NO_NUMBER
# NO_CONF = ConvenienceVariables.NO_CONF
# ERR_FILE_NOT_FOUND = ConvenienceVariables.ERR_FILE_NOT_FOUND
# ERR_UNREADABLE_DATA = ConvenienceVariables.ERR_UNREADABLE_DATA
# ERR_DUPLICATE_DATA = ConvenienceVariables.ERR_DUPLICATE_DATA
# PARTICIPANT_ID = ConvenienceVariables.PARTICIPANT_ID

EMPTY_CELL = ''


def edit_distance(s1, s2):
    """
    NOTE: This is an exact copy of levenshtein_distance from OCRScript_v3.py.

    Determines the number of character insertions/deletions/substitutions required to transform s1 into s2.
    :param s1: (String) One of the strings
    :param s2: (String) The other string
    :return: (int) The distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]


def initialize_usage_df():
    df = pd.DataFrame(columns=[PARTICIPANT_ID, LANGUAGE, DATE_DETECTED])
    for cat in categories_included:
        df[f'{TOTAL}_{cat}'] = None
        if cat == SCREENTIME:
            df[f'{TOTAL}_{SCREENTIME}_{MINUTES}'] = None
        for i in range(1, max_apps_per_category + 1):
            df[f'{cat}_{APP}_{i}_{NAME}'] = None
            df[f'{cat}_{APP}_{i}_{NUMBER}'] = None
            if cat == SCREENTIME:
                df[f'{SCREENTIME}_{APP}_{i}_{MINUTES}'] = None
    return df


class Participant:

    def __init__(self, user_id=None, device_id=None, device_os=None, lang=None):
        self.user_id = user_id
        self.device_id = device_id
        self.device_os = device_os
        self.language = lang
        self.usage_data = initialize_usage_df()
        self.usage_data_conf = initialize_usage_df()

    def __str__(self):
        return f"User ID: {self.user_id}    Language = {self.language}    Device_OS = {self.device_os}"

    def set_language(self, lang):
        self.language = lang

    def add_screenshot_data(self, ss):
        category = ss.category_detected
        device_os = ss.device_os_detected
        if category in [PICKUPS, UNLOCKS]:
            category = PICKUPS

        if ERR_FILE_NOT_FOUND in ss.errors or ERR_UNREADABLE_DATA in ss.errors:
            print("No data to add to participant's temporal data.")
            return

        if ss.time_period not in [YESTERDAY, DAY_OF_THE_WEEK]:
            print("Screenshot does not contain data for a previous day. "
                  "Screenshot data will not be added to participant's temporal data.")
            return
        elif ss.date_detected is None:
            print("Date not detected. Screenshot data will not be added to participant's temporal data.")
            return
        elif category is None:
            print("Category not detected. Screenshot data will not be added to participant's temporal data.")
            return

        try:
            date_index = self.usage_data[self.usage_data[DATE_DETECTED] == ss.date_detected].index[0]
        except IndexError:
            date_index = len(self.usage_data)
            self.usage_data.loc[date_index] = EMPTY_CELL

        if self.usage_data[f'{TOTAL}_{category}'][date_index] == EMPTY_CELL:
            print(f"Data from current screenshot added to participant: {self.user_id}    "
                  f"date: {ss.date_detected}    "
                  f"category: '{category}'")
            self.usage_data.loc[date_index, PARTICIPANT_ID] = self.user_id
            self.usage_data.loc[date_index, LANGUAGE] = self.language
            self.usage_data.loc[date_index, DATE_DETECTED] = ss.date_detected
            self.usage_data.loc[date_index, f'{TOTAL}_{category}'] = ss.daily_total

            self.usage_data_conf.loc[date_index, PARTICIPANT_ID] = self.user_id
            self.usage_data_conf.loc[date_index, LANGUAGE] = self.language
            self.usage_data_conf.loc[date_index, DATE_DETECTED] = ss.date_detected
            self.usage_data_conf.loc[date_index, f'{TOTAL}_{category}'] = ss.daily_total_conf

            if category == SCREENTIME:
                self.usage_data.loc[date_index, f'{TOTAL}_{category}_{MINUTES}'] = ss.daily_total_minutes

            for i in range(1, max_apps_per_category + 1):
                self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'] = ss.app_data[NAME][i]
                self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'] = ss.app_data[NUMBER][i]
                self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}'] = ss.app_data[NAME_CONF][i]
                self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'] = ss.app_data[NUMBER_CONF][i]
                if category == SCREENTIME:
                    self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{MINUTES}'] = ss.app_data[MINUTES][i]
                    self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{MINUTES}'] = \
                        ss.app_data[NUMBER_CONF][i]

        else:
            print(f"\nExisting {category} data found for participant {self.user_id} on {ss.date_detected}. "
                  f"Comparisons must be made.")
            value_format = ss.time_format_long if (category == SCREENTIME and device_os == ANDROID) else None

            best_total, best_total_conf = (
                OCRScript_v3.choose_between_two_values(text1=self.usage_data.loc[date_index, f'{TOTAL}_{category}'],
                                                       conf1=self.usage_data_conf.loc[date_index, f'{TOTAL}_{category}'],
                                                       text2=ss.daily_total,
                                                       conf2=ss.daily_total_conf,
                                                       val_fmt=value_format))
            (self.usage_data.loc[date_index, f'{TOTAL}_{category}'],
             self.usage_data_conf.loc[date_index, f'{TOTAL}_{category}']) = (best_total, best_total_conf)
            if category == SCREENTIME and best_total == ss.daily_total and best_total_conf == ss.daily_total_conf:
                self.usage_data.loc[date_index, f'{TOTAL}_{category}_{MINUTES}'] = ss.daily_total_minutes
                self.usage_data_conf.loc[date_index, f'{TOTAL}_{category}_{MINUTES}'] = ss.daily_total_conf

            moe = 2
            existing_data_lineup_index = -1
            new_data_lineup_index = -1
            lineup_found = False
            existing_data_contains_apps = False
            for i in range(1, max_apps_per_category + 1):
                if self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'] == NO_TEXT:
                    continue
                existing_data_contains_apps = True
                for j in range(1, max_apps_per_category + 1):
                    # compare app name i (from existing data) to app name j (new screenshot data)
                    # if they're equal (and not NO_TEXT), then this is where the two datasets line up.
                    if ss.app_data[NAME][j] == NO_TEXT:
                        continue
                    elif edit_distance(self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'], ss.app_data[NAME][j]) <= moe:
                        existing_data_lineup_index = i
                        new_data_lineup_index = j
                        lineup_found = True
                        break
                    else:
                        continue
                if lineup_found:
                    break

            if not lineup_found:
                if category == SCREENTIME:
                    # List of column names
                    existing_values_columns = [f'screentime_app_{i}_{MINUTES}' for i in range(1, max_apps_per_category + 1)]
                    new_values = ss.app_data.loc[:, MINUTES]

                else:
                    existing_values_columns = [f'{category}_{APP}_{i}_{NUMBER}' for i in range(1, max_apps_per_category + 1)]
                    new_values = ss.app_data.loc[:, NUMBER]
                # Select the row with index 'date_index' for the specified columns
                existing_values_row = self.usage_data.loc[date_index, existing_values_columns]
                existing_values_row = existing_values_row.astype(int)

                # Exclude values equal to NO_NUMBER
                filtered_ex_values = existing_values_row[existing_values_row != NO_NUMBER]
                if filtered_ex_values.empty:
                    pass
                filtered_new_values = new_values[(new_values != NO_NUMBER) & new_values != NO_TEXT].astype(int)

                min_existing_value = filtered_ex_values.min()
                max_existing_value = filtered_ex_values.max()
                min_new_value = filtered_new_values.min()
                max_new_value = filtered_new_values.max()

                if (not pd.isna(min_existing_value) and not pd.isna(max_new_value) and
                        min_existing_value >= max_new_value):
                    for i in range(max_apps_per_category, 0, -1):
                        if existing_values_row[existing_values_columns[i - 1]] == min_existing_value:
                            existing_data_lineup_index = i
                            break
                    new_data_lineup_index = 0

                elif (not pd.isna(min_new_value) and not pd.isna(max_existing_value) and
                        min_new_value >= max_existing_value) or not existing_data_contains_apps:
                    existing_data_lineup_index = 0
                    new_data_lineup_index = new_values.idxmin()

            if existing_data_lineup_index == -1 or new_data_lineup_index == -1:
                print("Could not determine where existing data and new screenshot data line up. "
                      "Existing data remains unchanged:")
                return

            max_lineup = max([existing_data_lineup_index, new_data_lineup_index])
            compare_df = pd.DataFrame(columns=['ex_name', 'ex_name_conf', 'ex_number', 'ex_number_conf',
                                               'new_name', 'new_name_conf', 'new_number', 'new_number_conf'])

            for i in range(max_apps_per_category + 1):
                ex_index = i + existing_data_lineup_index - max_lineup
                if ex_index > 0:
                    compare_df.loc[i, 'ex_name'] = self.usage_data[
                        f'{category}_{APP}_{ex_index}_{NAME}'][date_index]
                    compare_df.loc[i, 'ex_name_conf'] = self.usage_data_conf[
                        f'{category}_{APP}_{ex_index}_{NAME}'][date_index]
                    compare_df.loc[i, 'ex_number'] = self.usage_data[
                        f'{category}_{APP}_{ex_index}_{NUMBER}'][date_index]
                    compare_df.loc[i, 'ex_number_conf'] = self.usage_data_conf[
                        f'{category}_{APP}_{ex_index}_{NUMBER}'][date_index]
                    if category == SCREENTIME:
                        compare_df.loc[i, 'ex_minutes'] = self.usage_data[
                            f'{category}_{APP}_{ex_index}_{MINUTES}'][date_index]
                else:
                    compare_df.loc[i, 'ex_name'] = NO_TEXT
                    compare_df.loc[i, 'ex_name_conf'] = NO_CONF
                    compare_df.loc[i, 'ex_number'] = NO_TEXT
                    compare_df.loc[i, 'ex_number_conf'] = NO_CONF
                    if category == SCREENTIME:
                        compare_df.loc[i, 'ex_minutes'] = NO_NUMBER

                new_index = i + new_data_lineup_index - max_lineup
                if new_index > 0:
                    compare_df.loc[i, 'new_name'] = ss.app_data.loc[new_index, NAME]
                    compare_df.loc[i, 'new_name_conf'] = ss.app_data.loc[new_index, NAME_CONF]
                    compare_df.loc[i, 'new_number'] = ss.app_data.loc[new_index, NUMBER]
                    compare_df.loc[i, 'new_number_conf'] = ss.app_data.loc[new_index, NUMBER_CONF]
                    if category == SCREENTIME:
                        compare_df.loc[i, 'new_minutes'] = ss.app_data.loc[new_index, MINUTES]
                else:
                    compare_df.loc[i, 'new_name'] = NO_TEXT
                    compare_df.loc[i, 'new_name_conf'] = NO_CONF
                    compare_df.loc[i, 'new_number'] = NO_TEXT
                    compare_df.loc[i, 'new_number_conf'] = NO_CONF
                    if category == SCREENTIME:
                        compare_df.loc[i, 'new_minutes'] = NO_NUMBER

            print("\nTable of app comparisons to be made:")
            print(compare_df[['ex_name', 'ex_number', 'new_name', 'new_number']][1:])
            print()
            # if compare_df['ex_name'].equals(compare_df['new_name']) and \
            #         compare_df['ex_number'].equals(compare_df['new_number']):
            #     print("Note: Current screenshot data matches existing data. Screenshot will be flagged.\n")
            #     ss.add_error(ERR_DUPLICATE_DATA)

            for i in range(1, max_apps_per_category + 1):
                updated_app_name = False  # Initialize
                existing_name = compare_df.loc[i, 'ex_name']
                existing_number = compare_df.loc[i, 'ex_number']
                new_name = compare_df.loc[i, 'new_name']
                new_number = compare_df.loc[i, 'new_number']

                if existing_name == NO_TEXT and new_name == NO_TEXT:
                    print(f"No existing app name or new app name in position {i}. App name remains N/A.")

                elif existing_name == NO_TEXT and new_name != NO_TEXT:
                    print(f"No existing app name in position {i}. Updating to {new_name}.")
                    updated_app_name = True
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}']) = (
                        new_name, compare_df.loc[i, 'new_name_conf'])

                elif existing_name != NO_TEXT and new_name == NO_TEXT:
                    print(f"No new app name in position {i}. Keeping {existing_name}.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}']) = (
                        existing_name, compare_df.loc[i, 'ex_name_conf'])

                else:
                    (best_app_name, best_app_number) = OCRScript_v3.choose_between_two_values(text1=compare_df.loc[i, 'ex_name'],
                                                                                              conf1=compare_df.loc[i, 'ex_name_conf'],
                                                                                              text2=compare_df.loc[i, 'new_name'],
                                                                                              conf2=compare_df.loc[i, 'new_name_conf'])
                    if best_app_name == new_name != existing_name:
                        updated_app_name = True

                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}']) = (best_app_name, best_app_number)

                if updated_app_name:
                    print(f"Existing app number '{existing_number if existing_number != NO_TEXT else 'N/A'}' will also be updated to '{new_number}'.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        new_number, compare_df.loc[i, 'new_number_conf'])

                elif str(existing_number) == NO_TEXT and str(new_number) == NO_TEXT:
                    print(f"No existing app number or new app number in position {i}. App number remains N/A.")

                elif str(existing_number) == NO_TEXT and str(new_name) != NO_TEXT:
                    print(f"No existing app number in position {i}. Updating to {new_number}.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        new_number, compare_df.loc[i, 'new_number_conf'])

                elif str(existing_number) != NO_TEXT and str(new_number) == NO_TEXT:
                    print(f"No new app number in position {i}. Keeping {existing_number}.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        existing_number, compare_df.loc[i, 'ex_number_conf'])

                else:
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        OCRScript_v3.choose_between_two_values(text1=compare_df.loc[i, 'ex_number'],
                                                               conf1=compare_df.loc[i, 'ex_number_conf'],
                                                               text2=compare_df.loc[i, 'new_number'],
                                                               conf2=compare_df.loc[i, 'new_number_conf'],
                                                               val_fmt=value_format))

                if category == SCREENTIME:
                    if self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'] == compare_df.loc[i, 'new_number']:
                        self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{MINUTES}'] = (
                            compare_df.loc)[i, 'new_minutes']

    def add_screentime_data(self):
        pass

    def add_pickups_data(self):
        pass

    def add_notifications_data(self):
        pass
