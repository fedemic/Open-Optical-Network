class Connection:
    def __init__(self, initial_data):
        self._input = initial_data['input']
        self._output = initial_data['output']
        self._signal_power = initial_data['signal_power']
        self._latency = 0
        self._snr = None

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @property
    def snr(self):
        return self._snr

    @signal_power.setter
    def signal_power(self, value):
        self._signal_power = value

    @latency.setter
    def latency(self, value):
        self._latency = value

    @snr.setter
    def snr(self, value):
        self._snr = value