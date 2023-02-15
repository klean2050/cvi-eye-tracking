import numpy as np


def fix_bounds(data):
    data = [d for d in data if (d[0] != 0) or (d[1] != 0)]
    for i, (x, y) in enumerate(data):
        x = 1279 if x >= 1280 else 0 if x < 0 else x
        y = 719 if y >= 720 else 0 if y < 0 else y
        data[i] = [int(x), int(y)]
    return data


class Subject:
    def __init__(self, root, subject_id):
        """
        Input: data path (str) and subject id (str)
        Function: locates and loads eye-tracking files
        Returns: <Subject> object
        """
        self.id = subject_id
        self.asc_file = root + self.id + ".asc"
        with open(self.asc_file, "r") as f:
            self.data = f.readlines()

    def __trial_list(self, trial_name, numeric=False):
        """
        Input: name of trial (str) and if only measures (bool)
        Function: iterates over the data to record this trial
        Returns: list of measurement rows (str) for <trial_name>
        """
        trial_active, trial = False, []
        for line in self.data:
            if trial_active:
                if "End Trial {}".format(trial_name) in line:
                    break
                elif "Start Trial {}".format(trial_name) in line:
                    # just consider it as End Trial
                    break
                elif line[0].isdigit():
                    trial.append(line)
                elif not numeric:
                    trial.append(line)
            elif "Start Trial {}".format(trial_name) in line:
                trial_active = True
            else:
                continue
        return trial

    def __fixation_list(self, left=True):
        """
        Input: whether to consider the left eye or not
        Function: iterates over the data to record [SFIX-EFIX] intervals
        Returns: fixation time-series where each element is a 4-item list
        """
        this_eye = "L" if left else "R"
        fixation_active, fixations = False, []
        for line in self.trial:
            if fixation_active:
                if f"EFIX {this_eye}" in line:
                    fixations.append(current_fixation)
                    fixation_active = False
                elif f"SFIX {this_eye}" in line:
                    fixations.append(current_fixation)
                    fixation_active = False
                else:
                    if line[0].isdigit():
                        current_fixation.append(line)
            elif f"SFIX {this_eye}" in line:
                fixation_active = True
                current_fixation = []
            else:
                continue
        return fixations

    def __saccade_list(self, left=True):
        """
        Input: whether to consider the left eye or not
        Function: iterates over the data to record [SSAC-ESAC] intervals
        Returns: saccade time-series where each element is a 4-item list
        """
        this_eye = "L" if left else "R"
        saccade_active, saccades = False, []
        for line in self.trial:
            if saccade_active:
                if f"ESACC {this_eye}" in line:
                    saccades.append(current_saccade)
                    saccade_active = False
                else:
                    if line[0].isdigit():
                        current_saccade.append(line)
            elif f"SSACC {this_eye}" in line:
                saccade_active = True
                current_saccade = []
            else:
                continue
        return saccades

    def __fuse_eyes(self, array, which="both"):
        """
        Input: list of rows (str) and which eye to consider
        Function: iterates over the data to fuse L and R (x,y) measures
        Returns: (num_coords, 2) array and fraction of non-zero data
        """
        out = []
        for line in array:
            l = line.split("\t")
            row = [l[1], l[2], l[4], l[5]]
            row = [0.0 if set(r.strip()) == {"."} else float(r) for r in row]
            out.append(row)

        out = np.array(out)
        out[out < 0] = 0.0

        fused_out = []
        for (xl, yl, xr, yr) in out:
            if ([xl, yl] == [0, 0]) or which == "R":
                fused_out.append([xr, yr])
            elif ([xr, yr] == [0, 0]) or which == "L":
                fused_out.append([xl, yl])
            else:
                fused_out.append([(xl + xr) / 2, (yl + yr) / 2])

        fused_out = np.array(fused_out)
        fraction = 1 - np.sum((fused_out[:, 0] == 0) & (fused_out[:, 1] == 0))
        fraction /= len(fused_out)

        return fused_out, fraction

    def extract_data(self, trial_name, vel=False):
        """
        Input: trial name (str) and whether to consider velocity (bool)
        Function: loads the trial, fuses eyes and differentiates for vel
        Returns: (num_coords, 2) array and fraction of non-zero data
        """
        self.trial = self.__trial_list(trial_name, numeric=True)
        data_fusion, fraction = self.__fuse_eyes(self.trial)

        if vel:
            velocity = [[0.0, 0.0]]
            velocity += [
                data_fusion[i + 1] - data_fusion[i] for i in range(len(data_fusion) - 1)
            ]
            return np.array(velocity), fraction
        else:
            return data_fusion, fraction

    def extract_saccades(self, trial_name):
        """
        Input: trial name (str)
        Function: loads the trial and finds common L, R saccades
        Returns: list of saccades, each a 4-item dictionary
        """
        self.trial = self.__trial_list(trial_name)
        numeric = self.__trial_list(trial_name, numeric=True)
        if len(self.trial) <= 2:
            return []

        # get both eye saccades
        saccades_l = self.__saccade_list(left=True)
        saccades_r = self.__saccade_list(left=False)
        self.saccades = [saccades_l, saccades_r]

        # compute the intersection between saccades
        if len(saccades_r) <= len(saccades_l):
            primary_saccades = saccades_r
            secondary_saccades = saccades_l
        else:
            primary_saccades = saccades_l
            secondary_saccades = saccades_r

        intersections = []
        for i, pri_sac in enumerate(primary_saccades):
            pri_sac_times = [d.split("\t")[0] for d in pri_sac]
            pri_sac_times = set(pri_sac_times)

            for j, sec_sac in enumerate(secondary_saccades):
                sec_sac_times = [d.split("\t")[0] for d in sec_sac]
                sec_sac_times = set(sec_sac_times)

                intersections.append(
                    [
                        i,
                        j,
                        len(list(pri_sac_times.intersection(sec_sac_times))),
                    ]
                )
        intersections.sort(key=lambda x: x[-1], reverse=True)

        saccade_idxes = intersections[: len(primary_saccades)]
        saccades_all = []
        for idx in saccade_idxes:
            a = [int(d.split("\t")[0]) for d in primary_saccades[idx[0]]]
            b = [int(d.split("\t")[0]) for d in secondary_saccades[idx[1]]]

            time_stamps = list(set(a).intersection(set(b)))
            if len(time_stamps) < 5:
                continue

            trial_start_time = int(numeric[0].split("\t")[0])
            latency = np.min(time_stamps) - trial_start_time
            duration = np.max(time_stamps) - np.min(time_stamps)
            saccade_raw = [
                d
                for d in primary_saccades[idx[0]]
                if int(d.split("\t")[0]) in time_stamps
            ]

            fused_saccade, fraction = self.__fuse_eyes(saccade_raw)
            saccades_all.append(
                {
                    "data": fix_bounds(fused_saccade),
                    "fraction": fraction,
                    "latency": latency,
                    "duration": duration,
                }
            )

        saccades_all.sort(key=lambda x: x["latency"])
        return saccades_all

    def extract_fixations(self, trial_name):
        """
        Input: trial name (str)
        Function: loads the trial and finds common L, R fixations
        Returns: list of fixations, each a 4-item dictionary
        """
        self.trial = self.__trial_list(trial_name, numeric=False)
        numeric = self.__trial_list(trial_name, numeric=True)
        if len(self.trial) <= 2:
            return []

        # get both eye fixations
        fixations_l = self.__fixation_list(left=True)
        fixations_r = self.__fixation_list(left=False)
        self.fixations = [fixations_l, fixations_r]

        # compute the intersection between fixations
        if len(fixations_r) <= len(fixations_l):
            primary_fixations = fixations_r
            secondary_fixations = fixations_l
        else:
            primary_fixations = fixations_l
            secondary_fixations = fixations_r

        intersections = []
        for i, pri_fix in enumerate(primary_fixations):
            pri_fix_times = [d.split("\t")[0] for d in pri_fix]
            pri_fix_times = set(pri_fix_times)

            for j, sec_fix in enumerate(secondary_fixations):
                sec_fix_times = [d.split("\t")[0] for d in sec_fix]
                sec_fix_times = set(sec_fix_times)

                intersections.append(
                    [
                        i,
                        j,
                        len(list(pri_fix_times.intersection(sec_fix_times))),
                    ]
                )
        intersections.sort(key=lambda x: x[-1], reverse=True)

        fixation_idxes = intersections[: len(primary_fixations)]
        fixations_all = []
        for idx in fixation_idxes:
            a = [int(d.split("\t")[0]) for d in primary_fixations[idx[0]]]
            b = [int(d.split("\t")[0]) for d in secondary_fixations[idx[1]]]
            time_stamps = list(set(a).intersection(set(b)))
            if len(time_stamps) < 10:
                break

            trial_start_time = int(numeric[0].split("\t")[0])
            latency = np.min(time_stamps) - trial_start_time
            duration = np.max(time_stamps) - np.min(time_stamps)
            fixation_raw = [
                d
                for d in primary_fixations[idx[0]]
                if int(d.split("\t")[0]) in time_stamps
            ]

            fused_fixation, fraction = self.__fuse_eyes(fixation_raw)
            fixations_all.append(
                {
                    "data": fix_bounds(fused_fixation),
                    "fraction": fraction,
                    "latency": latency,
                    "duration": duration,
                }
            )

        fixations_all.sort(key=lambda x: x["latency"])
        return fixations_all

    def extract_trace(self, trial_name, smap):
        data = self.extract_data(trial_name)
        data = fix_bounds([d["data"] for d in data])
        return [smap[d[1], d[0]] for d in data]
