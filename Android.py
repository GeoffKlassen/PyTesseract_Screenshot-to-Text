"""This file contains Android-specific functions and variables."""


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
