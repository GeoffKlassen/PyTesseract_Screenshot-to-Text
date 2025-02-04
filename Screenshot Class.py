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
    def __init__(self, url=None, user_id=0, date=0, category=''):
        self.url = url
        self.filename = ''  # call function to extract the file name from the url
        self.user_id = user_id
        self.date_taken = date
        self.category_submitted = category
        self.image = ''  # call function to extract image from file/URL
        

    def to_string(self):
        return f"My URL is {self.url if self.url is not None else 'None'}"

    # def extract_filename(url):
    #    if the url doesn't contain a forward-slash, or a .jpg/.png extension, then the URL isn't valid,
    #    thus the filename cannot be extracted.
    #
    #    filename =  # the last portion of the string after the final forward-slash (/) character

screenshot1 = Screenshot()
print(screenshot1.to_string())