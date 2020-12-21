class SignalInformation:
    def __init__(self, signal_power, path):
        self._signal_power = signal_power
        self._noise_power = 0
        self._latency = 0
        self._path = path
        self._inv_gsnr = 0

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @property
    def latency(self):
        return self._latency

    @property
    def path(self):
        return self._path

    @property
    def inv_gsnr(self):
        return self._inv_gsnr

    @signal_power.setter
    def signal_power(self, value):
        self._signal_power = value

    @noise_power.setter
    def noise_power(self, value):
        self._noise_power = value

    @latency.setter
    def latency(self, value):
        self._latency = value

    @path.setter
    def path(self, value):
        self._path = value

    @inv_gsnr.setter
    def inv_gsnr(self, value):
        self._inv_gsnr = value

    def update_signal_power(self, increment):
        self.signal_power += increment

    def update_noise_power(self, increment):
        self.noise_power += increment

    def update_latency(self, increment):
        self.latency += increment

    def update_path(self):
        self.path.pop(0)

    def update_inv_gsnr(self, increment):
        self.inv_gsnr += increment
