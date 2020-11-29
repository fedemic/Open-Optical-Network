from Signal_information import *

# Lightpath object is a child class of SignalInformation that embeds channel information
class Lightpath(SignalInformation):
    def __init__(self, signal_power, path, channel):
        SignalInformation.__init__(self, signal_power, path)
        self._channel = channel

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
