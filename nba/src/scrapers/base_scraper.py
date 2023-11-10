from abc import ABC, abstractmethod


class BaseScraper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def scrape(self, url):
        """
        Scrape data from url. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
