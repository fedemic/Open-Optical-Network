from Signal_information import *
from constants import *

# Lightpath object is a child class of SignalInformation that embeds channel information
class Lightpath(SignalInformation):
    def __init__(self, signal_power, path, channel):
        SignalInformation.__init__(self, signal_power, path)
        self._channel = channel
        self._rs = RS
        self._df = DF

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
