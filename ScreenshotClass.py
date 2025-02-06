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
    def __init__(self, url=None, user_id=None, date=None, category=None):
        self.url = url
        self.filename = ''  # call function to extract the file name from the url
        self.user_id = user_id
        self.date_submitted = date
        self.category_submitted = category
        self.image = ''  # call function to extract image from file/URL


    def __str__(self):
        s_user_id = f"User ID: {self.user_id}".ljust(25)
        s_date = f"Date submitted: {self.date_submitted}".ljust(35)
        s_cat = f"Category submitted: {self.category_submitted}"
        return f"URL: {self.url}\n{s_user_id}{s_date}{s_cat}"

    # def extract_filename(url):
    #    if the url doesn't contain a forward-slash, or a .jpg/.png extension, then the URL isn't valid,
    #    thus the filename cannot be extracted.
    #
    #    filename =  # the last portion of the string after the final forward-slash (/) character
