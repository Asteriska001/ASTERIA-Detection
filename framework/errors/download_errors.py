
class DownloadFailed(IOError):
    """Error thrown if a download fails."""


class TooManyRequests(DownloadFailed):
    """Error thrown by HTTP 429 response."""