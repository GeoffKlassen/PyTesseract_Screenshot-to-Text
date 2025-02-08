"""This file contains iOS-specific functions and variables."""

misread_time_format = r'^[\d|t]+\s?[hn]$|^[\d|t]+\s?[hn]\s?[\d|tA]+\s?(min|m)$|^.{0,2}\s?[0-9AIt]+\s?(min|m)$|\d+\s?s$'
misread_number_format = r'^[0-9A]+$'
misread_time_or_number_format = '|'.join([misread_time_format, misread_number_format])


def main():
    print("I am now in iOS.py")