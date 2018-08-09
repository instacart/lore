from abc import ABCMeta, abstractmethod


class Base(object):
    @abstractmethod
    def main(self, **kwargs):
        pass
