import pandas as pd

import ConvenienceVariables
import OCRScript_v3
import RuntimeValues
import iOSFunctions

SCREENTIME = ConvenienceVariables.SCREENTIME
PICKUPS = ConvenienceVariables.PICKUPS
NOTIFICATIONS = ConvenienceVariables.NOTIFICATIONS
UNLOCKS = ConvenienceVariables.UNLOCKS
MAX_APPS = RuntimeValues.max_apps_per_category
NO_TEXT = ConvenienceVariables.NO_TEXT
NO_NUMBER = ConvenienceVariables.NO_NUMBER
NO_CONF = ConvenienceVariables.NO_CONF

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
    df = pd.DataFrame(columns=['participant_id', 'language', 'date'])
    for cat in RuntimeValues.categories_included:
        df[f'total_{cat}'] = None
        if cat == SCREENTIME:
            df[f'total_{SCREENTIME}_minutes'] = None
        for i in range(1, RuntimeValues.max_apps_per_category + 1):
            df[f'{cat}_app_{i}_name'] = None
            df[f'{cat}_app_{i}_number'] = None
            if cat == SCREENTIME:
                df[f'{SCREENTIME}_app_{i}_minutes'] = None
    return df


class Participant:

    def __init__(self, user_id=None, device_id=None, device_os=None, lang=None):
        self.user_id = user_id
        self.device_id = device_id
        self.device_os = device_os
        self.language = lang
        self.screenshots = []
        self.usage_data = initialize_usage_df()
        self.usage_data_conf = initialize_usage_df()

    def __str__(self):
        return f"User ID: {self.user_id}    Language = {self.language}    Device_OS = {self.device_os}"

    def set_language(self, lang):
        self.language = lang

    def add_screenshot(self, ss):
        self.screenshots.append(ss)
        category = ss.category_detected
        if category in [PICKUPS, UNLOCKS]:
            category = PICKUPS

        if "Error reading data" in ss.errors:
            print("No data to add to participant's temporal data.")
            return
        if ss.date_detected is None:
            print("Date not detected. Screenshot data will not be added to participant's temporal data.")
            return
        elif ss.time_period not in [ConvenienceVariables.YESTERDAY, ConvenienceVariables.DAY_OF_THE_WEEK]:
            print("Screenshot does not contain data for a previous day. "
                  "Screenshot data will not be added to participant's temporal data.")
            return
        elif category is None:
            print("Category not detected. Screenshot data will not be added to participant's temporal data.")
            return

        try:
            date_index = self.usage_data[self.usage_data['date'] == ss.date_detected].index[0]
        except IndexError:
            date_index = len(self.usage_data)
            self.usage_data.loc[date_index] = EMPTY_CELL

        if self.usage_data[f'total_{category}'][date_index] == EMPTY_CELL:
            print(f"Adding data from current screenshot to participant {self.user_id}'s temporal data.")
            self.usage_data.loc[date_index, 'participant_id'] = self.user_id
            self.usage_data.loc[date_index, 'language'] = self.language
            self.usage_data.loc[date_index, 'date'] = ss.date_detected
            self.usage_data.loc[date_index, f'total_{category}'] = ss.daily_total

            self.usage_data_conf.loc[date_index, 'participant_id'] = self.user_id
            self.usage_data_conf.loc[date_index, 'language'] = self.language
            self.usage_data_conf.loc[date_index, 'date'] = ss.date_detected
            self.usage_data_conf.loc[date_index, f'total_{category}'] = ss.daily_total_conf

            if category == SCREENTIME:
                self.usage_data.loc[date_index, f'total_{category}_minutes'] = ss.daily_total_minutes

            for i in range(1, MAX_APPS + 1):
                self.usage_data.loc[date_index, f'{category}_app_{i}_name'] = ss.app_data['name'][i]
                self.usage_data.loc[date_index, f'{category}_app_{i}_number'] = ss.app_data['number'][i]
                self.usage_data_conf.loc[date_index, f'{category}_app_{i}_name'] = ss.app_data['name_conf'][i]
                self.usage_data_conf.loc[date_index, f'{category}_app_{i}_number'] = ss.app_data['number_conf'][i]
                if category == SCREENTIME:
                    self.usage_data.loc[date_index, f'{category}_app_{i}_minutes'] = \
                        iOSFunctions.convert_text_time_to_minutes(ss.app_data['number'][i])
                    self.usage_data_conf.loc[date_index, f'{category}_app_{i}_minutes'] = \
                        ss.app_data['number_conf'][i]

        else:
            print("Existing data found. Comparisons must be made.")
            (self.usage_data.loc[date_index, f'total_{category}'],
             self.usage_data_conf.loc[date_index, f'total_{category}']) = (
                OCRScript_v3.choose_between_two_values(text1=self.usage_data.loc[date_index, f'total_{category}'],
                                                       conf1=self.usage_data_conf.loc[date_index, f'total_{category}'],
                                                       text2=ss.daily_total,
                                                       conf2=ss.daily_total_conf))
            moe = 2
            existing_data_app_num = -1
            new_data_app_num = -1
            lineup_found = False
            existing_data_contains_apps = False
            for i in range(1, MAX_APPS + 1):
                if self.usage_data.loc[date_index, f'{category}_app_{i}_name'] == NO_TEXT:
                    continue
                existing_data_contains_apps = True
                for j in range(1, MAX_APPS + 1):
                    # compare app name i (from existing data) to app name j (new screenshot data)
                    # if they're equal (and not NO_TEXT), then this is where the two datasets line up.
                    if ss.app_data['name'][j] == NO_TEXT:
                        continue
                    elif edit_distance(self.usage_data.loc[date_index, f'{category}_app_{i}_name'], ss.app_data['name'][j]) <= moe:
                        existing_data_app_num = i
                        new_data_app_num = j
                        lineup_found = True
                        break
                    else:
                        continue
                if lineup_found:
                    break

            if not lineup_found:
                if category == SCREENTIME:
                    # List of column names
                    existing_values_columns = [f'screentime_app_{i}_minutes' for i in range(1, MAX_APPS + 1)]
                    new_values = ss.app_data.loc[:, 'minutes']

                else:
                    existing_values_columns = [f'{category}_app_{i}_number' for i in range(1, MAX_APPS + 1)]
                    new_values = ss.app_data.loc[:, 'number']
                # Select the row with index 'date_index' for the specified columns
                existing_values_row = self.usage_data.loc[date_index, existing_values_columns]

                # Exclude values equal to -99999
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
                    for i in range(MAX_APPS, 0, -1):
                        if existing_values_row[existing_values_columns[i - 1]] == min_existing_value:
                            existing_data_app_num = i
                            break
                    new_data_app_num = 0

                elif (not pd.isna(min_new_value) and not pd.isna(max_existing_value) and
                        min_new_value >= max_existing_value) or not existing_data_contains_apps:
                    existing_data_app_num = 0
                    new_data_app_num = new_values.idxmin()

            if existing_data_app_num == -1 or new_data_app_num == -1:
                print("Could not determine where existing data and new screenshot data line up. "
                      "Existing data remains unchanged.")
                return

            max_lineup = max([existing_data_app_num, new_data_app_num])
            compare_df = pd.DataFrame(columns=['ex_name', 'ex_name_conf', 'ex_number', 'ex_number_conf',
                                               'new_name', 'new_name_conf', 'new_number', 'new_number_conf'])
            for i in range(MAX_APPS + 1):
                ex_index = i + existing_data_app_num - max_lineup
                if ex_index > 0:
                    compare_df.loc[i, 'ex_name'] = self.usage_data[
                        f'{category}_app_{ex_index}_name'][date_index]
                    compare_df.loc[i, 'ex_name_conf'] = self.usage_data_conf[
                        f'{category}_app_{ex_index}_name'][date_index]
                    compare_df.loc[i, 'ex_number'] = self.usage_data[
                        f'{category}_app_{ex_index}_number'][date_index]
                    compare_df.loc[i, 'ex_number_conf'] = self.usage_data_conf[
                        f'{category}_app_{ex_index}_number'][date_index]
                    if category == SCREENTIME:
                        compare_df.loc[i, 'ex_minutes'] = self.usage_data[
                            f'{category}_app_{ex_index}_minutes'][date_index]
                else:
                    compare_df.loc[i, 'ex_name'] = NO_TEXT
                    compare_df.loc[i, 'ex_name_conf'] = NO_CONF
                    compare_df.loc[i, 'ex_number'] = NO_TEXT
                    compare_df.loc[i, 'ex_number_conf'] = NO_CONF
                    if category == SCREENTIME:
                        compare_df.loc[i, 'ex_minutes'] = NO_NUMBER

                new_index = i + new_data_app_num - max_lineup
                if new_index > 0:
                    compare_df.loc[i, 'new_name'] = ss.app_data.loc[new_index, 'name']
                    compare_df.loc[i, 'new_name_conf'] = ss.app_data.loc[new_index, 'name_conf']
                    compare_df.loc[i, 'new_number'] = ss.app_data.loc[new_index, 'number']
                    compare_df.loc[i, 'new_number_conf'] = ss.app_data.loc[new_index, 'number_conf']
                    if category == SCREENTIME:
                        compare_df.loc[i, 'new_minutes'] = ss.app_data.loc[new_index, 'minutes']
                else:
                    compare_df.loc[i, 'new_name'] = NO_TEXT
                    compare_df.loc[i, 'new_name_conf'] = NO_CONF
                    compare_df.loc[i, 'new_number'] = NO_TEXT
                    compare_df.loc[i, 'new_number_conf'] = NO_CONF
                    if category == SCREENTIME:
                        compare_df.loc[i, 'new_minutes'] = NO_NUMBER

            print("\nTable of app comparisons to be made:")
            print(compare_df[['ex_name', 'ex_number', 'new_name', 'new_number']][1:])

            for i in range(1, MAX_APPS + 1):
                if compare_df.loc[i, 'ex_name'] == NO_TEXT and compare_df.loc[i, 'new_name'] == NO_TEXT:
                    print(f"No existing app name or new app name in position {i}. App name remains N/A.")

                elif compare_df.loc[i, 'ex_name'] == NO_TEXT and compare_df.loc[i, 'new_name'] != NO_TEXT:
                    print(f"No existing app name in position {i}. Updating to {compare_df.loc[i, 'new_name']}.")
                    (self.usage_data.loc[date_index, f'{category}_app_{i}_name'],
                     self.usage_data_conf.loc[date_index, f'{category}_app_{i}_name']) = (
                        compare_df.loc[i, 'new_name'], compare_df.loc[i, 'new_name_conf'])

                elif compare_df.loc[i, 'ex_name'] != NO_TEXT and compare_df.loc[i, 'new_name'] == NO_TEXT:
                    print(f"No new app name in position {i}. Keeping {compare_df.loc[i, 'ex_name']}.")
                    (self.usage_data.loc[date_index, f'{category}_app_{i}_name'],
                     self.usage_data_conf.loc[date_index, f'{category}_app_{i}_name']) = (
                        compare_df.loc[i, 'ex_name'], compare_df.loc[i, 'ex_name_conf'])

                else:
                    (self.usage_data.loc[date_index, f'{category}_app_{i}_name'],
                     self.usage_data_conf.loc[date_index, f'{category}_app_{i}_name']) = (
                        OCRScript_v3.choose_between_two_values(text1=compare_df.loc[i, 'ex_name'],
                                                               conf1=compare_df.loc[i, 'ex_name_conf'],
                                                               text2=compare_df.loc[i, 'new_name'],
                                                               conf2=compare_df.loc[i, 'new_name_conf']))

                if str(compare_df.loc[i, 'ex_number']) == NO_TEXT and str(compare_df.loc[i, 'new_number']) == NO_TEXT:
                    print(f"No existing app number or new app number in position {i}. App number remains N/A.")

                elif str(compare_df.loc[i, 'ex_number']) == NO_TEXT and str(compare_df.loc[i, 'new_name']) != NO_TEXT:
                    print(f"No existing app number in position {i}. Updating to {compare_df.loc[i, 'new_number']}.")
                    (self.usage_data.loc[date_index, f'{category}_app_{i}_number'],
                     self.usage_data_conf.loc[date_index, f'{category}_app_{i}_number']) = (
                        compare_df.loc[i, 'new_number'], compare_df.loc[i, 'new_number_conf'])

                elif str(compare_df.loc[i, 'ex_number']) != NO_TEXT and str(compare_df.loc[i, 'new_number']) == NO_TEXT:
                    print(f"No new app number in position {i}. Keeping {compare_df.loc[i, 'ex_number']}.")
                    (self.usage_data.loc[date_index, f'{category}_app_{i}_number'],
                     self.usage_data_conf.loc[date_index, f'{category}_app_{i}_number']) = (
                        compare_df.loc[i, 'ex_number'], compare_df.loc[i, 'ex_number_conf'])

                else:
                    (self.usage_data.loc[date_index, f'{category}_app_{i}_number'],
                     self.usage_data_conf.loc[date_index, f'{category}_app_{i}_number']) = (
                        OCRScript_v3.choose_between_two_values(text1=compare_df.loc[i, 'ex_number'],
                                                               conf1=compare_df.loc[i, 'ex_number_conf'],
                                                               text2=compare_df.loc[i, 'new_number'],
                                                               conf2=compare_df.loc[i, 'new_number_conf']))

                if category == SCREENTIME:
                    if self.usage_data.loc[date_index, f'{category}_app_{i}_number'] == compare_df.loc[i, 'new_number']:
                        self.usage_data.loc[date_index, f'{category}_app_{i}_minutes'] = (
                            compare_df.loc)[i, 'new_minutes']

    def add_screentime_data(self):
        pass

    def add_pickups_data(self):
        pass

    def add_notifications_data(self):
        pass
