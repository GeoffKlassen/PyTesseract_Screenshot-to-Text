import pandas as pd

import ConvenienceVariables
import RuntimeValues

SCREENTIME = ConvenienceVariables.SCREENTIME


def initialize_usage_data():
    df = pd.DataFrame(columns=['participant_id', 'date'])
    for cat in RuntimeValues.categories_included:
        df[f'{cat}_heading_found'] = None
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
        self.usage_data = initialize_usage_data()

    def __str__(self):
        return f"User ID: {self.user_id}    Language = {self.language}    Device_OS = {self.device_os}"

    def set_language(self, lang):
        self.language = lang

    def add_screenshot(self, ss):
        self.screenshots.append(ss)
        # here you also need to add the data to the usage_data DataFrame and make sure data doesn't get overwritten
        if ss.date_detected is None or ss.time_period == ConvenienceVariables.WEEK:
            return
        if ss.date_detected not in self.usage_data['date']:
            print(f"{ss.date_detected} is a new date to {self.user_id}")

    def add_screentime_data(self):
        pass

    def add_pickups_data(self):
        pass

    def add_notifications_data(self):
        pass
