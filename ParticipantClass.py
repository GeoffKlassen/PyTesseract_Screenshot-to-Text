"""
    This file contains the definition of the Participant object, and its variables and relevant functions.

    Functions:
        edit_distance:          A copy of the levenshtein_distance(s1, s2) function from OCRScript_v3.py.
                                This is used to calculate the character-distance between two strings.
        initialize_usage_df:    Initializes a dataframe of usage data for each Avicenna participant.
"""
import pandas as pd

from RuntimeValues import *
import OCRScript_v3
import iOSFunctions

EMPTY_CELL = ''


def edit_distance(s1, s2):
    """
    NOTE: This is an exact copy of levenshtein_distance from OCRScript_v3.py.

    Determines the number of character insertions/deletions/substitutions required to transform s1 into s2.
    :param s1: (String) One of the strings
    :param s2: (String) The other string
    :return: (int) The character-distance between s1 and s2.
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
    """
    Initialize a dataframe of phone and app usage for the Participant.
    Each row represents a single day of phone usage and contains columns for:
        For each category of data collected in the study (screentime, pickups, notifications):
            daily total,
            daily total minutes (for screentime data only),
            For the top 'n' apps, as requested from the study:
                app_n_name,
                app_n_number,
                app_n_minutes (for screentime data only)

    :return: The initialized usage dataframe with all the necessary columns.
    """
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
    """
    Each Participant object has several variables, including:
        user_id:        the User ID (Participant ID) for a given Avicenna account
        device_id:      the Device ID taken from the phone/tablet/etc. used by the Avicenna participant
        device_os:      the Operating System of the phone/tablet/etc. used by the Avicenna participant
        language:       the (most common) language of images used by the Avicenna participant
        user_data:      a dataframe of phone and app usage for the Avicenna participant (each row is a single day)
                        (data values include: daily total, app names, and app totals, for each of the 3 data categories)
        user_data_conf: the confidence values of the data values in user_data

    """
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
        """
        Adds the information from a provided Screenshot object to the Participant's usage_data. If usage_data already
                    has data for the participant on the given day, it lines up the existing data and the new data (by
                    app name) and makes comparisons to decide which datum to keep for each column.
        :param ss: The Screenshot object that contains usage data to add to the Participant.
        :return: None (Ensures the best data values are kept for each column)
        """

        # Initialize
        category = ss.category_detected
        device_os = ss.device_os_detected
        if category in [PICKUPS, UNLOCKS]:
            # The Android version of 'pickups' is 'unlocks', but for consistency, we rename it to 'pickups' so that it
            # lines up with the iOS name.
            category = PICKUPS

        if ERR_FILE_NOT_FOUND in ss.errors or ERR_UNREADABLE_DATA in ss.errors:
            print("\nNo data to add to participant's temporal data.")
            return

        if ss.relative_day not in [YESTERDAY, DAY_OF_THE_WEEK]:
            # If the daily usage data is not for a previous day (i.e. for 'today' or 'week'), then don't add it to the
            # usage data for that participant (since data from 'today' is incomplete and 'week' data is an average/sum
            # of the past week's usage, both of which are biased values).
            print("Screenshot does not contain data for a previous day. "
                  "Screenshot data will not be added to participant's temporal data.")
            return
        elif ss.date_detected is None:
            # If the date cannot be detected from the screenshot, we cannot guarantee which date the data are for.
            print("Date not detected. Screenshot data will not be added to participant's temporal data.")
            return
        elif category is None:
            # If the category is not detected, we cannot guarantee what category the data are for.
            print("Category not detected. Screenshot data will not be added to participant's temporal data.")
            return

        try:
            # Find the existing row of data (indexed by date), if it exists
            date_index = self.usage_data[self.usage_data[DATE_DETECTED] == ss.date_detected].index[0]
        except IndexError:
            # If it doesn't exist, create it
            date_index = len(self.usage_data)
            self.usage_data.loc[date_index] = EMPTY_CELL

        if self.usage_data[f'{TOTAL}_{category}'][date_index] == EMPTY_CELL:
            # If the existing date row has no values in the columns relevant to the provided Screenshot object, then
            # simply add the new data to usage_data.
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
                # Screentime screenshots also have a number of minutes to store.
                # This is more useful for data analysis than a string like '1 hour 23 minutes'.
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
            # If data already exists in usage_data at the same place as the data from the given Screenshot, then
            # the two sets of data must be lined up and compared to determine which datum is the best to keep for each
            # column.
            print(f"Existing {category} data found for participant {self.user_id} on {ss.date_detected}. "
                  f"Data will be compared.\n")
            value_format = ss.time_format_long if (category == SCREENTIME and device_os == ANDROID) else None

            # Choose between the existing daily total and the new daily total, and place it in the usage_data.
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

            moe = 2  # A margin of error (moe) for use with edit_distance, to determine if an app name from the existing
                     # data and an app name from the new data should be considered the same app name (and be compared).
            existing_data_lineup_index = -1
            new_data_lineup_index = -1
            lineup_found = False
            existing_data_contains_apps = False
            for i in range(1, max_apps_per_category + 1):
                # Loop through all the existing app names
                if self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'] == NO_TEXT:
                    continue
                existing_data_contains_apps = True
                for j in range(1, max_apps_per_category + 1):
                    # Loop through all the new app names.
                    if ss.app_data[NAME][j] == NO_TEXT:
                        continue
                    elif edit_distance(self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'], ss.app_data[NAME][j]) <= moe:
                        # Compare app name i (from existing data) to app name j (new screenshot data).
                        # If they're close enough (and not NO_TEXT), then this is where the two datasets will be lined up.
                        existing_data_lineup_index = i
                        new_data_lineup_index = j
                        # The existing data and new screenshot data will be placed in a comparison dataframe,
                        # such that existing app i lines up with new app j.
                        lineup_found = True
                        break
                    else:
                        continue
                if lineup_found:
                    break

            if not lineup_found:
                # An overlap between the existing data and new data could not be found. In this case, the numerical
                # values will be compared to see if the existing data all has larger numbers than the new data (or vice versa).

                # Set the columns of (numerical) values to compare
                if category == SCREENTIME:
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
                    # In this case, the new data (if any) all belongs at the top of the list (i.e. the first app in
                    # the new data can be considered 'app #1' for the current participant, etc.)
                    pass
                filtered_new_values = new_values[(new_values != NO_NUMBER) & new_values != NO_TEXT].astype(int)

                min_existing_value = filtered_ex_values.min()
                max_existing_value = filtered_ex_values.max()
                min_new_value = filtered_new_values.min()
                max_new_value = filtered_new_values.max()

                if (not pd.isna(min_existing_value) and not pd.isna(max_new_value) and
                        min_existing_value >= max_new_value):
                    # The smallest existing value is larger than the largest new value, so the existing data 'outranks'
                    # the new data (i.e. existing app #1 will still be app #1 for the current participant, etc.)
                    for i in range(max_apps_per_category, 0, -1):
                        # Work backwards through the existing data to find the last existing app row.
                        # This is done so we line up the new data right below the last existing app.
                        if existing_values_row[existing_values_columns[i - 1]] == min_existing_value:
                            existing_data_lineup_index = i  # The existing data will be placed in the comparison df
                                                            # starting in row i
                            break
                    new_data_lineup_index = 0  # The new data will be placed in the comparison df, starting at position 0

                elif (not pd.isna(min_new_value) and not pd.isna(max_existing_value) and
                        min_new_value >= max_existing_value) or not existing_data_contains_apps:
                    # The smallest new value is larger than the largest existing value, so the new data 'outranks'
                    # the existing data (i.e. new app #1 will be considered app #1 for the current participant, etc.)
                    existing_data_lineup_index = 0 # The existing data will be placed in the comparison df, starting at position 0
                    new_data_lineup_index = new_values.idxmin()  # The new data will be placed in the comparison df,
                                                                 # just below the existing data

            if existing_data_lineup_index == -1 or new_data_lineup_index == -1:
                print("Could not determine where existing data and new screenshot data line up. "
                      "Existing data remains unchanged:")
                return

            max_lineup = max(existing_data_lineup_index, new_data_lineup_index)
            compare_df = pd.DataFrame(columns=['ex_name', 'ex_name_conf', 'ex_number', 'ex_number_conf',
                                               'new_name', 'new_name_conf', 'new_number', 'new_number_conf'])

            # Build the dataframe of existing apps and new apps, lined up as calculated above, for comparison
            for i in range(1, max_apps_per_category + abs(new_data_lineup_index - existing_data_lineup_index) + 1):
                # i will range from 1 to 3, or from 1 to 8, e.g.
                """
                    Calculate the app index from the existing data that should go at position i in the comparison df.
                
                    For example, if it was determined that the existing app #5 matched with new app #3, then
                    existing_data_lineup_index = 5, new_data_lineup_index = 3, and max_lineup = max(5, 3) = 5.
                    Then, ex_index  =  i + 5 - 5  =  i.
                    This means the existing data will not be shifted up or down when placing it into the comparison df.
                    Also, new_index  =  i + 3 - 5  =  i - 2.  Since i - 2 > 0 when i > 2, the existing data will be
                    shifted down 2 places when inserting it into the comparison df.
                    
                    Example:
                         Existing           New
                             1               .          When i = 1, ex_index = 1 + 5 - 5 = 1 (existing app 1 goes here)
                             2               .
                             3               1          When i = 3, new_index = 3 + 3 - 5 = 1 (new app 1 goes here)
                             4               2
                             5 lines up with 3
                             6               4
                             7               5
                             8               6
                                             7
                                             8
                """
                ex_index = i + existing_data_lineup_index - max_lineup
                if 0 < ex_index < max_apps_per_category + 1:
                    # Place existing app # 'ex_index' into row 'i' of the comparison dataframe
                    compare_df.loc[i, 'ex_name'] = self.usage_data[f'{category}_{APP}_{ex_index}_{NAME}'][date_index]
                    compare_df.loc[i, 'ex_name_conf'] = self.usage_data_conf[f'{category}_{APP}_{ex_index}_{NAME}'][date_index]
                    compare_df.loc[i, 'ex_number'] = self.usage_data[f'{category}_{APP}_{ex_index}_{NUMBER}'][date_index]
                    compare_df.loc[i, 'ex_number_conf'] = self.usage_data_conf[f'{category}_{APP}_{ex_index}_{NUMBER}'][date_index]
                    if category == SCREENTIME:
                        compare_df.loc[i, 'ex_minutes'] = self.usage_data[f'{category}_{APP}_{ex_index}_{MINUTES}'][date_index]
                else:
                    # Insert an empty row of existing data
                    compare_df.loc[i, 'ex_name'] = NO_TEXT
                    compare_df.loc[i, 'ex_name_conf'] = NO_CONF
                    compare_df.loc[i, 'ex_number'] = NO_TEXT
                    compare_df.loc[i, 'ex_number_conf'] = NO_CONF
                    if category == SCREENTIME:
                        compare_df.loc[i, 'ex_minutes'] = NO_NUMBER

                """
                    Calculate the app index from the new data that should go at position i in the comparison df.
                    See example above. 
                """
                new_index = i + new_data_lineup_index - max_lineup
                if 0 < new_index < max_apps_per_category + 1:
                    # Place new app # 'new_index' into row 'i' of the comparison dataframe
                    compare_df.loc[i, 'new_name'] = ss.app_data.loc[new_index, NAME]
                    compare_df.loc[i, 'new_name_conf'] = ss.app_data.loc[new_index, NAME_CONF]
                    compare_df.loc[i, 'new_number'] = ss.app_data.loc[new_index, NUMBER]
                    compare_df.loc[i, 'new_number_conf'] = ss.app_data.loc[new_index, NUMBER_CONF]
                    if category == SCREENTIME:
                        compare_df.loc[i, 'new_minutes'] = ss.app_data.loc[new_index, MINUTES]
                else:
                    # Insert an empty row of new data
                    compare_df.loc[i, 'new_name'] = NO_TEXT
                    compare_df.loc[i, 'new_name_conf'] = NO_CONF
                    compare_df.loc[i, 'new_number'] = NO_TEXT
                    compare_df.loc[i, 'new_number_conf'] = NO_CONF
                    if category == SCREENTIME:
                        compare_df.loc[i, 'new_minutes'] = NO_NUMBER

                if all(comparison_value == NO_TEXT for comparison_value in
                       compare_df.loc[i, ['ex_name', 'ex_number', 'new_name', 'new_number']].tolist()) and \
                        i > max_apps_per_category:
                    compare_df.drop(compare_df.index[-1], inplace=True)
                    break

            print("\nTable of app comparisons to be made:")
            print(compare_df[['ex_name', 'ex_number', 'new_name', 'new_number']])
            print()

            for i in range(1, max_apps_per_category + 1):
                # Loop through the rows of the comparison dataframe and make comparisons of each row.

                updated_app_name = False  # Initialize
                existing_name = compare_df.loc[i, 'ex_name']
                existing_number = compare_df.loc[i, 'ex_number']
                new_name = compare_df.loc[i, 'new_name']
                new_number = compare_df.loc[i, 'new_number']

                """
                    Compare and update the app name in position i
                """
                if existing_name == NO_TEXT and new_name == NO_TEXT:
                    # Neither the existing data nor the new data have an app in position i
                    print(f"No existing app name or new app name in position {i}. App name remains N/A.")

                elif existing_name == NO_TEXT and new_name != NO_TEXT:
                    # Only the new data has an app in position i
                    print(f"No existing app name in position {i}. Updating to {new_name}.")
                    updated_app_name = True
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}']) = (
                        new_name, compare_df.loc[i, 'new_name_conf'])

                elif existing_name != NO_TEXT and new_name == NO_TEXT:
                    # Only the existing data has an app in position i
                    print(f"No new app name in position {i}. Keeping {existing_name}.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}']) = (
                        existing_name, compare_df.loc[i, 'ex_name_conf'])

                else:
                    # Both the existing data and new data have an app in position i; compare the two
                    (best_name, best_name_conf) = OCRScript_v3.choose_between_two_values(text1=compare_df.loc[i, 'ex_name'],
                                                                                         conf1=compare_df.loc[i, 'ex_name_conf'],
                                                                                         text2=compare_df.loc[i, 'new_name'],
                                                                                         conf2=compare_df.loc[i, 'new_name_conf'])
                    if best_name == new_name != existing_name:
                        updated_app_name = True

                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NAME}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NAME}']) = (best_name, best_name_conf)

                """
                    Compare and update the app number in position i
                """
                if updated_app_name:
                    # When replacing an existing app name with a new app name, the new app's number must also
                    # replace the existing app's number.
                    # If the existing app name and the new app name are the same, then their numbers will still be compared.
                    print(f"Existing app number '{existing_number if existing_number != NO_TEXT else 'N/A'}' will also be updated to '{new_number}'.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        new_number, compare_df.loc[i, 'new_number_conf'])

                elif str(existing_number) == NO_TEXT and str(new_number) == NO_TEXT:
                    # Neither the existing data nor the new data have a number in position i
                    print(f"No existing app number or new app number in position {i}. App number remains N/A.")

                elif str(existing_number) == NO_TEXT and str(new_name) != NO_TEXT:
                    # Only the new data has a number in position i
                    print(f"No existing app number in position {i}. Updating to {new_number}.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        new_number, compare_df.loc[i, 'new_number_conf'])

                elif str(existing_number) != NO_TEXT and str(new_number) == NO_TEXT:
                    # Only the existing data has a number in position i
                    print(f"No new app number in position {i}. Keeping {existing_number}.")
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (
                        existing_number, compare_df.loc[i, 'ex_number_conf'])

                else:
                    # Both the existing data and new data have a number in position i; compare the two
                    (best_number, best_number_conf) = OCRScript_v3.choose_between_two_values(text1=compare_df.loc[i, 'ex_number'],
                                                                                             conf1=compare_df.loc[i, 'ex_number_conf'],
                                                                                             text2=compare_df.loc[i, 'new_number'],
                                                                                             conf2=compare_df.loc[i, 'new_number_conf'],
                                                                                             val_fmt=value_format)
                    (self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}'],
                     self.usage_data_conf.loc[date_index, f'{category}_{APP}_{i}_{NUMBER}']) = (best_number, best_number_conf)

                    if category == SCREENTIME:
                        # Update the minutes as well (for screentime data)
                        if best_number == compare_df.loc[i, 'new_number']:
                            self.usage_data.loc[date_index, f'{category}_{APP}_{i}_{MINUTES}'] = compare_df.loc[i, 'new_minutes']

        return

    # def add_screentime_data(self):
    #     pass
    #
    # def add_pickups_data(self):
    #     pass
    #
    # def add_notifications_data(self):
    #     pass