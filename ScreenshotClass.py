"""The Screenshot class will be used to store the raw image and the extracted data from a screenshot,
   as well as metadata about the screenshot as taken from the database of URLs."""


class Screenshot:

    """Initialize an object of type Screenshot
    url               : the full https://.../image.jpg URL as taken from the database of URLs
    filename          : the name of the file as stored locally (without the web address), taken from the above URL
    user_id           : the user_ID as taken from the database of URLs
    date_taken        : the date the screenshot was submitted, as taken from the database of URLs
    category_submitted: the category in which the screenshot was submitted (screentime, notifications, pickups/unlocks)
                        as taken from the database of URLs
    """
    def __init__(self, participant, url, device_os=None, date=None, category=None):
        self.participant = participant
        self.url = url
        self.filename = url[url.rfind('/') + 1:] if url is not None else None
        self.user_id = participant.user_id
        self.device_os = device_os
        self.date_submitted = date
        self.category_submitted = category
        self.height = 0
        self.width = 0
        self.language = None
        self.grey_image = None
        self.date_detected = None
        self.category_detected = None
        self.text = None
        self.time_period = None
        self.rows_with_day_type = None
        self.headings_df = None
        self.daily_total = None
        self.daily_total_conf = None

    def __str__(self):
        s_user_id = f"User ID: {self.user_id}".ljust(22)
        s_device_os = f"Device OS: {self.device_os}".ljust(23)
        s_date = f"Date submitted: {self.date_submitted}".ljust(33)
        s_cat = f"Category submitted: {self.category_submitted}"
        return f"URL: {self.url}\n{s_user_id}{s_device_os}{s_date}{s_cat}"

    def set_dimensions(self, dim):
        self.height, self.width = dim[0], dim[1]

    def set_language(self, lang):
        self.language = lang

    def set_image(self, img):
        self.grey_image = img

    def set_date_detected(self, date):
        self.date_detected = date

    def set_category_detected(self, cat):
        self.category_detected = cat

    def set_text(self, text):
        self.text = text

    def set_time_period(self, period):
        self.time_period = period

    def set_rows_with_day_type(self, df):
        self.rows_with_day_type = df

    def set_headings(self, df):
        self.headings_df = df

    def set_daily_total(self, value, conf):
        self.daily_total = value
        self.daily_total_conf = conf
