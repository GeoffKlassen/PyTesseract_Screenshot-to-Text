import pandas as pd

import ConvenienceVariables
import RuntimeValues
import iOSFunctions

SCREENTIME = ConvenienceVariables.SCREENTIME
PICKUPS = ConvenienceVariables.PICKUPS
NOTIFICATIONS = ConvenienceVariables.NOTIFICATIONS
UNLOCKS = ConvenienceVariables.UNLOCKS
MAX_APPS = RuntimeValues.max_apps_per_category
NO_TEXT = ConvenienceVariables.NO_TEXT

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
    df = pd.DataFrame(columns=['participant_id', 'date'])
    for cat in RuntimeValues.categories_included:
        df[f'{cat}_subheading_found'] = None
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
        if category == SCREENTIME:
            subheading_found_in_ss = ss.screentime_subheading_found
        elif category in [PICKUPS, UNLOCKS]:
            category = PICKUPS
            subheading_found_in_ss = ss.pickups_subheading_found
        elif category == NOTIFICATIONS:
            subheading_found_in_ss = ss.notifications_subheading_found
        else:
            subheading_found_in_ss = False

        if ss.date_detected is None:
            print("Date not detected. Screenshot data will not be added to participant's temporal data.")
            return
        elif ss.time_period in [ConvenienceVariables.WEEK, ConvenienceVariables.TODAY]:
            print("Screenshot does not contain data for 'yesterday'. "
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
        print(f'Was the subheading found? {subheading_found_in_ss}')

        if self.usage_data[f'total_{category}'][date_index] == EMPTY_CELL:
            print("Adding data from current screenshot to user data in participant.")
            self.usage_data.loc[date_index, 'participant_id'] = self.user_id
            self.usage_data.loc[date_index, 'date'] = ss.date_detected
            self.usage_data.loc[date_index, f'{category}_subheading_found'] = subheading_found_in_ss
            self.usage_data.loc[date_index, f'total_{category}'] = ss.daily_total

            self.usage_data_conf.loc[date_index, 'participant_id'] = self.user_id
            self.usage_data_conf.loc[date_index, 'date'] = ss.date_detected
            self.usage_data_conf.loc[date_index, f'{category}_subheading_found'] = subheading_found_in_ss
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
            # start with comparing the daily totals

            # then go on to this part
            moe = 2
            existing_data_app_num = -1
            new_data_index = -1
            lineup_found = False
            for i in range(1, MAX_APPS + 1):
                for j in range(1, MAX_APPS + 1):
                    # compare app name i (from existing data) to app @ index j (new screenshot data)
                    # if they're equal (and not NO_TEXT), then this is where the two datasets line up.
                    if ss.app_data['name'][j] == NO_TEXT:
                        continue
                    elif edit_distance(self.usage_data[f'{category}_app_{i}_name'], ss.app_data['name'][j]) <= moe:
                        existing_data_app_num = i
                        new_data_index = j
                        lineup_found = True
                        break
                    else:
                        continue
                if lineup_found:
                    break
            if lineup_found:
                # line up the data in a new dataframe
                if existing_data_app_num < new_data_index:
                    pass
                else:
                    pass
                pass
            else:
                # we need to see which dataset belongs first (in descending order of app usage)
                pass

        # self.usage_data.loc[len(self.usage_data)] = EMPTY_CELL
        # new_values_row = pd.DataFrame({'participant_id': [self.user_id],
        #                                'date': [ss.date_detected],
        #                                f'{category}_subheading_found': subheading_found_in_ss})
        # columns_to_make_blank = (x for x in self.usage_data.columns[:] if
        #                          x not in ['participant_id', 'date', f'{category}_subheading_found'])
        # for col in columns_to_make_blank:
        #     new_values_row[col] = EMPTY_CELL
        # new_conf_row = new_values_row.copy()
        #
        # # Put the daily total (and daily total conf) into the master df
        # new_values_row[f'total_{category}'] = ss.daily_total
        # new_conf_row[f'total_{category}'] = ss.daily_total_conf
        # # For screentime screenshots, put the total time converted to (int) minutes into the master df as well
        # if category == SCREENTIME:
        #     new_values_row[f'total_{category}_minutes'] = ss.daily_total_minutes
        #     new_conf_row[f'total_{category}_minutes'] = ss.daily_total_conf
        # # Put the data from the current screenshot into the master df
        # for n in range(RuntimeValues.max_apps_per_category):
        #     new_values_row[f'{category}_app_{n + 1}_name'] = ss.app_data['app'][n]
        #     new_values_row[f'{category}_app_{n + 1}_number'] = ss.app_data['number'][n]
        #     new_conf_row[f'{category}_app_{n + 1}_name'] = ss.app_data['app_conf'][n]
        #     new_conf_row[f'{category}_app_{n + 1}_number'] = ss.app_data['number_conf'][n]
        #     if category == SCREENTIME:
        #         new_values_row[f'{category}_app_{n + 1}_minutes'] = iOSFunctions.convert_text_time_to_minutes(
        #             ss.app_data['number'][n])
        #         new_conf_row[f'{category}_app_{n + 1}_minutes'] = ss.app_data['number_conf'][n]
        # self.usage_data = new_values_row if self.usage_data.shape[0] == 0 else (
        #     pd.concat([self.usage_data, new_values_row], ignore_index=True))
        # self.usage_data_conf = new_conf_row if self.usage_data.shape[0] == 0 else (
        #     pd.concat([self.usage_data_conf, new_conf_row], ignore_index=True))

    def add_screentime_data(self):
        pass

    def add_pickups_data(self):
        pass

    def add_notifications_data(self):
        pass
