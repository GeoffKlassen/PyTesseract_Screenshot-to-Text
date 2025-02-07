class Participant:

    def __init__(self, user_id=None, lang=None, os=None):
        self.id = user_id
        self.language = lang
        self.phone_os = os
        self.screenshots = []

    def __str__(self):
        return f"ID={self.id}   Language={self.language}   Phone_OS={self.phone_os}"

    def set_language(self, lang):
        self.language = lang

    def add_screenshot(self, ss):
        self.screenshots.append(ss)