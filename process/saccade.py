import numpy as np


class SaccadeAnalyzer:
    def __init__(self, root, saccades):
        self.root = root
        self.saccades = saccades

    def number_of_saccades(self):
        return len(self.saccades)

    def duration_of_saccades(self):
        durs = [f["duration"] for f in self.saccades]
        if durs:
            return np.mean(durs), np.std(durs)
        else:
            return 0, 0

    def velocity_of_saccades(self):
        velocities = []
        for saccade in self.saccades:
            data = saccade["data"]
            velocity = [data[i + 1] - data[i] for i in range(len(data) - 1)]
            velocity = np.mean(velocity, axis=1)
            velocities.append(np.linalg.norm(velocity))
        return np.mean(velocities)
