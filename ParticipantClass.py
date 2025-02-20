import pandas as pd

import ConvenienceVariables
import RuntimeValues
import iOSFunctions

SCREENTIME = ConvenienceVariables.SCREENTIME
PICKUPS = ConvenienceVariables.PICKUPS
NOTIFICATIONS = ConvenienceVariables.NOTIFICATIONS

EMPTY_CELL = ''


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
        elif category == PICKUPS:
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

        if self.usage_data[f'{category}_subheading_found'][date_index] == EMPTY_CELL:
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

            for i in range(RuntimeValues.max_apps_per_category):
                self.usage_data.loc[date_index, f'{category}_app_{i + 1}_name'] = ss.app_data['name'][i]
                self.usage_data.loc[date_index, f'{category}_app_{i + 1}_number'] = ss.app_data['number'][i]
                self.usage_data_conf.loc[date_index, f'{category}_app_{i + 1}_name'] = ss.app_data['name_conf'][i]
                self.usage_data_conf.loc[date_index, f'{category}_app_{i + 1}_number'] = ss.app_data['number_conf'][i]
                if category == SCREENTIME:
                    self.usage_data.loc[date_index, f'{category}_app_{i + 1}_minutes'] = \
                        iOSFunctions.convert_text_time_to_minutes(ss.app_data['number'][i])
                    self.usage_data_conf.loc[date_index, f'{category}_app_{i + 1}_minutes'] = \
                        ss.app_data['number_conf'][i]

        else:
            print("Existing data found. Comparisons must be made.")

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
