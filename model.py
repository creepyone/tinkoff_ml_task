from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, corpus):
        """

        :param corpus:
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self):
        raise NotImplementedError


class


class Model(AbstractModel):
    def __init__(self):
        ...

    def fit(self, corpus):
        ...

    def generate(self):
        ...
