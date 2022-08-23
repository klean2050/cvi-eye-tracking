from audioop import reverse
import numpy as np
import os
import sys
sys.path.append('./gazemae/gazemae')
from settings import *
from network import ModelManager
from argparse import Namespace
from torch import no_grad, Tensor
import glob
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tslearn.metrics import dtw
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
import pickle as pkl 

class Subject():
    def __init__(self, subject_id):
        self.id = subject_id
        self.asc_file = f'../asc_data_v1/{self.id}.asc'
        self.data = open(self.asc_file, 'r')
        # self.lines = [d for d in self.data]
        # print(self.lines)
    
    def readTrialData(self, trial_name, vel=False, verbose=False):
        # print(f'reading data for trial: {trial_name}')
        trial_active = False
        trial = []
        for line in self.data:
            if trial_active:
                if 'End Trial {}'.format(trial_name) in line:
                    # print(line)
                    break
                elif line[0].isdigit():
                    trial.append(line)
                    
            elif 'Start Trial {}'.format(trial_name) in line:
                # print('start trial encountered')
                # print (line)
                trial_active = True
            else:
                continue
        data = []
        self.trial = trial
        for line in trial:
            l = line.split('\t')
            data.append([l[1], l[2], l[4], l[5]])
        self.data_raw = data
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                try:
                    data[i][j] = float(element)
                except:
                    data[i][j] = 0.0
        # data = [[float(d) if d.isnumeric() else 0.0 for d in row] for row in data]
        
        data = np.array(data)
        # self.data = data
        if verbose:
            print(data)

        # combine the data from 2 eyes
        data[data<0] = 0.0
        data_fusion = []
        for data_ in data:
            xl, yl, xr, yr = data_
            if xl == 0 and yl == 0:
                data_fusion.append([xr, yr])
            elif xr == 0 and yr == 0:
                data_fusion.append([xl, yl])
            else:
                data_fusion.append([(xl + xr)/2, (yl + yr)/2])
        data_fusion = np.array(data_fusion)
        fraction = np.sum((data_fusion[:,0] == 0) & (data_fusion[:,1] == 0)) / len(data_fusion)
        if vel:
            velocity = np.array([data_fusion[i+1] - data_fusion[i] for i in range(len(data_fusion)-1)])
            return velocity, fraction
        # print(f'{self.id}: available data: {1 - fraction}')
        return data_fusion, fraction

    def extractFixations(self, trial_name):
        #read all the lines from a trial 
        trial_active = False
        trial = []
        self.trial_numeric= []
        for line in self.data:
            if trial_active:
                if ('End Trial {}'.format(trial_name) in line ) or ('Start Trial {}'.format(trial_name) in line):
                    # print(line)
                    break
                else:
                    if line[0].isnumeric():
                        self.trial_numeric.append(line)
                    trial.append(line)
                    
            elif 'Start Trial {}'.format(trial_name) in line:
                # print('start trial encountered')
                # print (line)
                trial_active = True
            else:
                continue
        
        self.trial = trial
        if len(trial) <=2 :
            return []
        trial_st = int(self.trial_numeric[0].split('\t')[0])
        trial_et = int(self.trial_numeric[-1].split('\t')[0])
        #get the left eye fixations
        fixation_active = False
        fixations_l = []
        for line in trial:
            if fixation_active:
                if 'EFIX L' in line:
                    fixations_l.append(current_fixation)
                    fixation_active = False
                else:
                    if line[0].isdigit():
                        current_fixation.append(line)
                        # print(line)
            elif 'SFIX L' in line:
                # print('got a start')
                fixation_active = True
                current_fixation = []
            else:
                continue

        #get the right eye fixations
        fixation_active = False
        fixations_r = []
        for line in trial:
            if fixation_active:
                if 'EFIX R' in line:
                    fixations_r.append(current_fixation)
                    fixation_active = False
                else:
                    if line[0].isdigit():
                        current_fixation.append(line)
            elif 'SFIX R' in line:
                fixation_active = True
                current_fixation = []
            else:
                continue
        self.fixations = [fixations_l, fixations_r]
        #compute the intersection between the left and right fixations
        if len(fixations_r) <= len(fixations_l):
            primary_fixations = fixations_r
            secondary_fixations = fixations_l
        else:
            primary_fixations = fixations_l
            secondary_fixations = fixations_r
        
        # print(len(primary_fixations), len(secondary_fixations))
        intersections = []
        for i, pri_fix in enumerate(primary_fixations):
            pri_fix_times = [d.split('\t')[0] for d in pri_fix]
            for j, sec_fix in enumerate(secondary_fixations):
                sec_fix_times = [d.split('\t')[0] for d in sec_fix]
                intersections.append([i, j, len(list(set(pri_fix_times).intersection(set(sec_fix_times))))])
        
        intersections.sort(key = lambda x: x[-1], reverse=True)
        # print(intersections)
        fixation_idxes = intersections[:len(primary_fixations)]
        fixations_all = []
        for idx in fixation_idxes:
            a = [int(d.split('\t')[0]) for d in primary_fixations[idx[0]]]
            b = [int(d.split('\t')[0]) for d in secondary_fixations[idx[1]]]
            time_stamps = list(set(a).intersection(set(b)))
            if len(time_stamps) < 10:
                break
            # print(time_stamps)
            fixation_st = np.min(time_stamps) - trial_st
            fixation_et = np.max(time_stamps) - trial_st
            fixation = []
            fixation_raw = [d for d in primary_fixations[idx[0]] if int(d.split('\t')[0]) in time_stamps]
            # print(fixation_raw)
            for line in fixation_raw:
                l = line.split('\t')
                fixation.append([l[1], l[2], l[4], l[5]])
            
            for i, row in enumerate(fixation):
                for j, element in enumerate(row):
                    try:
                        fixation[i][j] = float(element)
                    except:
                        fixation[i][j] = 0.0
            fixation = np.array(fixation)
            # print(fixation)
            fixation[fixation<0] = 0.0
            fused_fixation = []
            for fix_ in fixation:
                xl, yl, xr, yr = fix_
                if xl ==0 and yl ==0:
                    fused_fixation.append([xr, yr])
                elif xr == 0 and yr ==0:
                    fused_fixation.append([xl, yl])
                else:
                    fused_fixation.append([(xl + xr)/2, (yl + yr)/2])
            fused_fixation = np.array(fused_fixation)
            # print(fused_fixation.shape)
            self.fused_fixation = fused_fixation
            fraction = 1 - np.sum((fused_fixation[:,0] ==0) & (fused_fixation[:,1]==0)) / len(fused_fixation)
            fixations_all.append([fused_fixation, fraction, fixation_st, fixation_et])
        return fixations_all


class FeatureRepresentation():
    def __init__(self):
        args = {'log_to_file': False,
                'verbose': False,
                'autoencoder': 'temporal',
                'save_model': False,
                'tensorboard': False,
                'encoder': 'vanilla_tcn',
                'multiscale': False,
                'causal_encoder': False,
                'hierarchical': False,
                'hz': 0,
                'viewing_time': -1,
                'signal_type': 'pos',
                'slice_time_windows': None,
                'augment': False,
                'loss_type': '',
                'use_validation_set': False,
                'cuda': True,
                'rec_loss': 'mse',
                'batch_size': 64,
                'epochs': 200,
                'learning_rate': 0.0005,
                'model_pos': 'pos-i3738',
                'model_vel': 'vel-i8528',
                'pca_components': 0,
                'save_tsne_plot': True,
                'cv_folds': 5,
                'generate': False,
                'task': ''}
        args = Namespace(**args)
        self.model = ModelManager(args, training=False)
        self.vel = False
    
    def representForTrial(self, trial_data):
        # extract representation for a subject pertaining to a particular trial
        network = self.model.network['vel']
        network.eval()
        x = trial_data.swapaxes(0,1)
        with no_grad():
            x = Tensor(x).unsqueeze(0)
            rep = network.encode(x.cuda())[0].cpu().detach().numpy()
        return rep

    def representationAllSubjects(self, trial_name):
        self.trial_name = trial_name
        data_dir = '../asc_data_v1'
        subject_ids = glob.glob(os.path.join(data_dir, '*.asc'))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        self.rep_all_subjects = {}
        self.frac = {}
        for subject in subject_ids:
            # print(f'processing subject: {subject}') 
            sub = Subject(subject)
            trial_data, frac = sub.readTrialData(trial_name, vel=True)
            # trial_data = np.array([d for d in trial_data if (d[0]!=0) or (d[1]!=0)])
            self.rep_all_subjects[subject] = self.representForTrial(trial_data)
            self.frac[subject] = 1 - frac

    def plotDistanceMatrix(self, trial_name, vel=False):
        print(f'comuting distance matrix for {trial_name}')
        self.vel = vel
        self.representationAllSubjects(trial_name)
        subject_ids = [k for k in self.frac.keys() if self.frac[k] > 0.5]
        subject_ids.sort()
        distance_matrix = np.empty((len(subject_ids), len(subject_ids)))
        for i, keyi in enumerate(subject_ids):
            for j, keyj in enumerate(subject_ids):
                distance_matrix[i,j] = cdist(self.rep_all_subjects[keyi].reshape(1, -1),\
                    self.rep_all_subjects[keyj].reshape(1, -1), metric='cosine')
        
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(distance_matrix)
        ax.set_xticks(list(range(len(subject_ids))))
        ax.set_xticklabels(subject_ids, rotation=90)
        fig.colorbar(img)
        plt.tight_layout()
        if self.vel:
            plt.savefig(f'distance_vel_TCN_clean/{trial_name}.png')   
        else: 
            plt.savefig(f'distance_TCN_clean/{trial_name}.png')
        

    def plotTSNE(self):
        print(f'computing the tsne {self.trial_name}')
        x_y = [[self.rep_all_subjects[k], k] for k in self.frac.keys() if self.frac[k] > 0.5]
        x = np.array([d[0] for d in x_y])
        y = ['ctrl' if d[1].startswith('2') else 'cvi' for d in x_y]
        pca = PCA(n_components=20).fit_transform(x)
        tsne = TSNE().fit_transform(pca)
        cvi = np.array([t for i, t in enumerate(tsne) if y[i] == 'cvi'])
        ctrl = np.array([t for i, t in enumerate(tsne) if y[i] == 'ctrl'])
        plt.clf()
        plt.scatter(ctrl[:,0], ctrl[:,1], label='ctrl')
        plt.scatter(cvi[:,0], cvi[:,1], label='cvi')
        plt.legend()
        plt.savefig(f'tsnes_vel/tsne_{self.trial_name}.png')
        return

class DTWAnalysis():
    def readAllData(self, trial_name):
        data_dir = '../asc_data_v1'
        subject_ids = glob.glob(os.path.join(data_dir, '*.asc'))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        self.timeseries = {}
        self.data_frac = {}
        for subject in subject_ids:
            sub = Subject(subject)
            trial_data, frac = sub.readTrialData(trial_name, vel=self.vel)
            self.timeseries[subject] = trial_data
            self.data_frac[subject] = 1 - frac
        
    def plotDistanceMatrixTrialWise(self, trial_name, vel=False):
        print(f'distance matrix for  {trial_name}')
        self.vel = vel
        self.readAllData(trial_name)
        subject_ids = [k for k in self.data_frac.keys() if self.data_frac[k] > 0.5]
        subject_ids.sort()
        distance_matrix = np.empty((len(subject_ids), len(subject_ids)))
        for i, keyi in enumerate(subject_ids):
            for j, keyj in enumerate(subject_ids):
                distance_matrix[i,j] = dtw(self.timeseries[keyi][:1000], self.timeseries[keyj][:1000])
        plt.clf()
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(distance_matrix)
        ax.set_xticks(list(range(len(subject_ids))))
        ax.set_xticklabels(subject_ids, rotation=90)
        fig.colorbar(img)
        plt.tight_layout()
        if self.vel:
            plt.savefig(f'distance_matrix_vel/{trial_name}.png')   
        else: 
            plt.savefig(f'distance_matrix/{trial_name}.png')

class SaliencyTrace():
    def readAllData(self, trial_name):
        data_dir = '../asc_data_v1'
        subject_ids = glob.glob(os.path.join(data_dir, '*.asc'))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        self.timeseries = {}
        self.data_frac = {}
        for subject in subject_ids:
            sub = Subject(subject)
            trial_data, frac = sub.readTrialData(trial_name)
            self.timeseries[subject] = trial_data
            self.data_frac[subject] = 1 - frac
    
    def readAllFixations(self, trial_name):
        data_dir = '../asc_data_v1'
        subject_ids = glob.glob(os.path.join(data_dir, '*.asc'))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        self.fixation_timeseries = {}
        for subject in subject_ids:
            # print(subject, 'reading fixation')
            sub = Subject(subject)
            fixations = sub.extractFixations(trial_name)
            self.fixation_timeseries[subject] = fixations

    def computeTrace(self, trial_name, data):
        saliency_map = np.load(f'../smaps/gen/{trial_name}.npy').squeeze()
        saliency_map = (saliency_map - np.min(saliency_map[:]))/(np.max(saliency_map) - np.min(saliency_map))
        data = [d for d in data if (d[0] != 0) or (d[1]!=0)]
        for i, row in enumerate(data):
            x, y = data[i]
            if x >= 1280:
                x = 1279
            elif x <0:
                x = 0
            if y >= 720:
                y =719
            elif y<0:
                y=0
            data[i] = [int(x),int(y)] 
        trace = [saliency_map[d[1], d[0]] for d in data]
        return trace

    def computeFixationTraces(self, trial_name, fixations):
        self.readAllFixations(trial_name)
        saliency_map = np.load(f'../smaps/gen/{trial_name}.npy').squeeze()
        saliency_map = (saliency_map - np.min(saliency_map[:]))/(np.max(saliency_map) - np.min(saliency_map))
        avg_fixations = []
        for fixation_ in fixations:
            fixation, frac, st, et = fixation_
            fixation = [d for d in fixation if (d[0] != 0) or (d[1] != 0)]
            for i, row in enumerate(fixation):
                x, y = row
                if x>=1280:
                    x =1279
                elif x<0:
                    x=0
                if y>=720:
                    y=719
                elif y<0:
                    y=0
                fixation[i] = [int(x), int(y)]
            trace_ = [saliency_map[d[1], d[0]] for d in fixation]
            avg_fixations.append([np.mean(trace_), frac, st, et])
        return avg_fixations
    
    def computeFixationTraceForAll(self, trial_name):
        self.readAllFixations(trial_name)
        # self.avg_fixations = {subject: self.computeFixationTraces(trial_name, \
        #                       data) for subject, data in \
        #                       self.fixation_timeseries.items() if len(data)}
        self.avg_fixations = {}
        for sub, data in self.fixation_timeseries.items():
            # print(sub)
            if len(data):
                self.avg_fixations[sub] = self.computeFixationTraces(trial_name, data)

        
    
    def computeTraceForALL(self, trial_name):
        self.readAllData(trial_name)
        subject_ids = [k for k in self.data_frac.keys() if self.data_frac[k] > 0.5]
        subject_ids.sort()
        self.trace = {}
        for subject in subject_ids:
            self.trace[subject] = self.computeTrace(trial_name, self.timeseries[subject])
    
    def plotTrace(self, trial_name):
        self.computeTraceForALL(trial_name)
        plt.clf()
        for subject, trace in self.trace.items():
            x = [i for i in range(1, len(trace) + 1)]
            plt.plot(x, trace, label=subject)
        plt.grid()
        plt.legend()
        plt.savefig('trace_all.png', dpi=300)
    
    def computeDistance(self, trial_name):
        print(f'{trial_name}')
        self.computeTraceForALL(trial_name)
        subject_ids = list(self.trace.keys())
        subject_ids.sort()
        distance_matrix = np.empty((len(subject_ids), len(subject_ids)))
        for i, keyi in enumerate(subject_ids):
            for j, keyj in enumerate(subject_ids):
                distance_matrix[i,j] = dtw(self.trace[keyi][:1000], self.trace[keyj][:1000])
        plt.clf()
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(distance_matrix)
        ax.set_xticks(list(range(len(subject_ids))))
        ax.set_xticklabels(subject_ids, rotation=90)
        fig.colorbar(img)
        plt.tight_layout()
        plt.savefig(f'distance_dtw_trace/{trial_name}.png')

    def compareTrials(self):
        data_dir = '../asc_data_v1'
        subject_ids = glob.glob(os.path.join(data_dir, '*.asc'))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        compare_trials = [['Freeviewingstillimage_36.jpg', 'Freeviewingstillimage_36_cutout.tif'],
                            ['Freeviewingstillimage_28.jpg', 'Freeviewingstillimage_28_cutout.tif'],
                            ['Freeviewingstillimage_93.jpg', 'Freeviewingstillimage_93_cutout.tif'],
                            ['Freeviewingstillimage_36.jpg', 'Freeviewingstillimage_36_cutout.tif'],
                            ['Freeviewingstillimage_10.jpg', 'Freeviewingstillimage_10_cutout.tif']]
        for pair in compare_trials:
            print(pair)
            self.comparison = {}
            for subject in subject_ids:
                sub = Subject(subject)
                data0, frac0 = sub.readTrialData(pair[0])
                data1, frac1 = sub.readTrialData(pair[1])
                if (frac0 < 0.5) and  (frac1 < 0.5):
                    trace0 = self.computeTrace(pair[0], data0)
                    trace1 = self.computeTrace(pair[1], data1)
                    self.comparison[subject] = dtw(trace0, trace1)
            x_y = [[k, v] for k, v in self.comparison.items()]
            x_y.sort(key= lambda x: x[0])
            plt.clf()
            plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
            plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
            plt.grid()
            plt.tight_layout()
            name = pair[0][:-4].split('_')[1]
            plt.savefig(f'comparison_{name}.png')
            
class TraceAnalyzer():
    def __init__(self, trial_name) -> None:
        self.trial_name = trial_name
        self.st = SaliencyTrace()
        

    def representation(self, type ='avg'):
        self.st.computeTraceForALL(self.trial_name)
        self.traces = self.st.trace
        dict = {}
        for sub in self.traces.keys():
            dict[sub] = np.mean(self.traces[sub])
        return dict
        # return {sub: np.mean(self.traces[sub] for sub in self.traces.keys())}
        
    def metrics(self):
        rep = self.representation()
        x_y = [[k, v] for k,v in rep.items()]
        x_y.sort(key= lambda x: x[0])
        plt.clf()
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        name = os.path.join('trace_rep', self.trial_name[:-4] + '.png')
        plt.savefig(name, dpi=300)
        a = [d[1] for d in x_y if d[0].startswith('1')]
        b = [d[1] for d in x_y if d[0].startswith('2')]
        with open('ttest_trace_avg.txt', 'a') as f:
            print(self.trial_name[:-4], ttest_ind(a, b), file=f)
    
    def plotTraces(self):
        subjects = list(self.traces.keys())
        cvi = [sub for sub in subjects if sub.startswith('1')]
        ctrl = [sub for sub in subjects if sub.startswith('2')]
        plt.clf()
        for sub in ctrl:
            x = [i for i in range(1, len(self.traces[sub])+1)]
            plt.plot(x, self.traces[sub], label=sub)
        plt.savefig(f'trace_group_wise/{self.trial_name[:-4]}_ctrl.png', dpi=300)
        plt.clf()
        for sub in cvi:
            x = [i for i in range(1, len(self.traces[sub])+1)]
            plt.plot(x, self.traces[sub], label=sub)
        plt.savefig(f'trace_group_wise/{self.trial_name[:-4]}_cvi.png', dpi=300)

    # def plot_fixation_latency(self):
    #     self.st.computeFixationTraceForAll(self.trial_name)
    #     subjects

class FixationAnalyzer():
    def __init__(self, trial_name):
        self.trial_name = trial_name
        self.avg_fixation = pkl.load(open(os.path.join('avg_fixation_trial_wise', trial_name + '.pkl'), 'rb'))

    def latency_first_fixation(self):
        x_y = [[sub, data[0][2]] for sub, data in self.avg_fixation.items()]
        x_y.sort(key=lambda x: x[0])
        return x_y

    def saliency_first_fixation(self):
        x_y = [[sub, data[0][0]] for sub, data in self.avg_fixation.items()]
        x_y.sort(key=lambda x: x[0])
        return x_y
    
    def saliency_longest_fixation(self):
        x_y = []
        for sub, data in self.avg_fixation.items():
            data.sort(key = lambda x: x[-1] - x[-2])
            x_y.append([sub, data[-1][0]])
        x_y.sort(key=lambda x: x[0])
        return x_y
    
    def latency_longest_fixation(self):
        x_y = []
        for sub, data in self.avg_fixation.items():
            data.sort(key = lambda x: x[-1] - x[-2])
            x_y.append([sub, data[-1][2]])
        x_y.sort(key=lambda x: x[0])
        return x_y
    
    def latency_maximum_saliency(self):
        x_y = []
        for sub, data in self.avg_fixation.items():
            data.sort(key = lambda x: x[0])
            x_y.append([sub, data[-1][2]])
        x_y.sort(key=lambda x: x[0])
        return x_y
    
    def compute_metrics(self):
        stats = {}
        plt.clf()
        x_y = self.latency_first_fixation()
        plt.subplot(2,3,1)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title('latency_first_fixation')
        plt.ylabel('time')
        a = [d[1] for d in x_y if d[0].startswith('1')]
        b = [d[1] for d in x_y if d[0].startswith('2')]
        stat, p_value = ttest_ind(a, b)
        stats['latency_first_fixation'] = [stat, p_value]

        x_y = self.saliency_first_fixation()
        plt.subplot(2,3,2)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title('saliency_first_fixation')
        plt.ylabel('time')
        a = [d[1] for d in x_y if d[0].startswith('1')]
        b = [d[1] for d in x_y if d[0].startswith('2')]
        stat, p_value = ttest_ind(a, b)
        stats['saliency_first_fixation'] = [stat, p_value]

        x_y = self.saliency_longest_fixation()
        plt.subplot(2,3,3)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title('saliency_longest_fixation')
        plt.ylabel('time')
        a = [d[1] for d in x_y if d[0].startswith('1')]
        b = [d[1] for d in x_y if d[0].startswith('2')]
        stat, p_value = ttest_ind(a, b)
        stats['saliency_longest_fixation'] = [stat, p_value]

        x_y = self.latency_longest_fixation()
        plt.subplot(2,3,4)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title('latency_longest_fixation')
        plt.ylabel('time')
        a = [d[1] for d in x_y if d[0].startswith('1')]
        b = [d[1] for d in x_y if d[0].startswith('2')]
        stat, p_value = ttest_ind(a, b)
        stats['latency_longest_fixation'] = [stat, p_value]

        x_y = self.latency_maximum_saliency()
        plt.subplot(2,3,5)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title('latency_maximum_saliency')
        plt.ylabel('time')
        a = [d[1] for d in x_y if d[0].startswith('1')]
        b = [d[1] for d in x_y if d[0].startswith('2')]
        stat, p_value = ttest_ind(a, b)
        stats['latency_maximum_saliency'] = [stat, p_value]
        plt.savefig('fixation_stats/' + self.trial_name[:-4] + '.png', dpi=300)
        return stats





if __name__ == '__main__':
    # fr = FeatureRepresentation()
    # da = DTWAnalysis()
    # st = SaliencyTrace()
    # st.compareTrials()
    
    trials_images =['Freeviewingstillimage_1.jpg',
    'Freeviewingstillimage_2.jpg',
    'Freeviewingstillimage_4.jpg',
    'Freeviewingstillimage_5.jpg',
    'Freeviewingstillimage_93.jpg',
    'Freeviewingstillimage_7.jpg',
    'Freeviewingstillimage_8.jpg',
    'Freeviewingstillimage_9.jpg',
    'Freeviewingstillimage_10.jpg',
    'Freeviewingstillimage_11.jpg',
    'Freeviewingstillimage_12.jpg',
    'Freeviewingstillimage_13.jpg',
    'Freeviewingstillimage_15.jpg',
    'Freeviewingstillimage_16.jpg',
    'Freeviewingstillimage_17.jpg',
    'Freeviewingstillimage_18.jpg',
    'Freeviewingstillimage_19.jpg',
    'Freeviewingstillimage_20.jpg',
    'Freeviewingstillimage_21.jpg',
    'Freeviewingstillimage_22.jpg',
    'Freeviewingstillimage_23.jpg',
    'Freeviewingstillimage_24.jpg',
    'Freeviewingstillimage_25.jpg',
    'Freeviewingstillimage_26.jpg',
    'Freeviewingstillimage_27.jpg',
    'visual search orientation 4_1.jpg',
    'Freeviewingstillimage_28.jpg',
    'visual search orientation 8_1.jpg',
    'Freeviewingstillimage_29.jpg',
    'visual search orientation 16_1.jpg',
    'Freeviewingstillimage_10_cutout.tif',
    'visual search orientation 24_1.jpg',
    'Freeviewingstillimage_31.jpg',
    'visual search orientation 32_1.jpg',
    'visual search orientation 4_1.jpg',
    'Freeviewingstillimage_93_cutout.tif',
    'visual search orientation 8_1.jpg',
    'Freeviewingstillimage_33.jpg',
    'visual search orientation 16_1.jpg',
    'Moviestillimage_8.jpg',
    'visual search orientation 24_1.jpg',
    'Freeviewingstillimage_35.jpg',
    'visual search orientation 32_1.jpg',
    'visual search orientation 4_1.jpg',
    'Freeviewingstillimage_36.jpg',
    'visual search orientation 8_1.jpg',
    'Freeviewingstillimage_28_cutout.tif',
    'visual search orientation 16_1.jpg',
    'Moviestillimage_6.jpg',
    'visual search orientation 24_1.jpg',
    'Freeviewingstillimage_39.jpg',
    'visual search orientation 32_1.jpg',
    'visual search form 4_1.jpg',
    'Freeviewingstillimage_40.jpg',
    'visual search form 8_1.jpg',
    'Freeviewingstillimage_41.jpg',
    'visual search form 16_1.jpg',
    'Freeviewingstillimage_92.jpg',
    'visual search form 24_1.jpg',
    'Freeviewingstillimage_88.jpg',
    'visual search form 32_1.jpg',
    'visual search form 4_1.jpg',
    'Freeviewingstillimage_36_cutout.tif',
    'visual search form 8_1.jpg',
    'Freeviewingstillimage_45.jpg',
    'visual search form 16_1.jpg',
    'Freeviewingstillimage_46.jpg',
    'visual search form 24_1.jpg',
    'Freeviewingstillimage_47.jpg',
    'visual search form 32_1.jpg',
    'visual search form 4_1.jpg',
    'Moviestillimage_12.jpg',
    'visual search form 8_1.jpg',
    'Freeviewingstillimage_49.jpg',
    'visual search form 16_1.jpg',
    'Freeviewingstillimage_50.jpg',
    'visual search form 24_1.jpg',
    'Freeviewingstillimage_88_cutout.tif',
    'visual search form 32_1.jpg']
    trials = ['Opening Movie',
    'Movie 3.mp4',
    'Movie 6.mp4',
    'Freeviewingstillimage_1.jpg',
    'Freeviewingstillimage_2.jpg',
    'Moviestillimage_1.jpg',
    'Freeviewingstillimage_4.jpg',
    'Freeviewingstillimage_5.jpg',
    'Movie 7.mp4',
    'Freeviewingstillimage_93.jpg',
    'Freeviewingstillimage_7.jpg',
    'Freeviewingstillimage_8.jpg',
    'Freeviewingstillimage_9.jpg',
    'Freeviewingstillimage_10.jpg',
    'Movie 8.mp4',
    'Freeviewingstillimage_11.jpg',
    'Freeviewingstillimage_12.jpg',
    'Freeviewingstillimage_13.jpg',
    'Moviestillimage_3.jpg',
    'Freeviewingstillimage_15.jpg',
    'Movie 9.mp4',
    'Freeviewingstillimage_16.jpg',
    'Freeviewingstillimage_17.jpg',
    'Freeviewingstillimage_18.jpg',
    'Freeviewingstillimage_19.jpg',
    'Movie 11.mp4',
    'Freeviewingstillimage_20.jpg',
    'Freeviewingstillimage_21.jpg',
    'Freeviewingstillimage_22.jpg',
    'Freeviewingstillimage_23.jpg',
    'Movie 12.mp4',
    'Freeviewingstillimage_24.jpg',
    'Freeviewingstillimage_25.jpg',
    'Freeviewingstillimage_26.jpg',
    'Freeviewingstillimage_27.jpg',
    'Movie 13.mp4',
    'VF bear juggling.mp4',
    'VF bunny fruit.mp4',
    'VF Turtle with glasses.mp4',
    'VF watermelon shaking.mp4',
    'visual search orientation 4_1.jpg',
    'Freeviewingstillimage_28.jpg',
    'visual search orientation 8_1.jpg',
    'Freeviewingstillimage_29.jpg',
    'visual search orientation 16_1.jpg',
    'Freeviewingstillimage_10_cutout.tif',
    'visual search orientation 24_1.jpg',
    'Freeviewingstillimage_31.jpg',
    'visual search orientation 32_1.jpg',
    'Movie 1.mp4',
    'visual search orientation 4_1.jpg',
    'Freeviewingstillimage_93_cutout.tif',
    'visual search orientation 8_1.jpg',
    'Freeviewingstillimage_33.jpg',
    'visual search orientation 16_1.jpg',
    'Moviestillimage_8.jpg',
    'visual search orientation 24_1.jpg',
    'Freeviewingstillimage_35.jpg',
    'visual search orientation 32_1.jpg',
    'Movie 2.mp4',
    'visual search orientation 4_1.jpg',
    'Freeviewingstillimage_36.jpg',
    'visual search orientation 8_1.jpg',
    'Freeviewingstillimage_28_cutout.tif',
    'visual search orientation 16_1.jpg',
    'Moviestillimage_6.jpg',
    'visual search orientation 24_1.jpg',
    'Freeviewingstillimage_39.jpg',
    'visual search orientation 32_1.jpg',
    'Movie 10.mp4',
    'visual search form 4_1.jpg',
    'Freeviewingstillimage_40.jpg',
    'visual search form 8_1.jpg',
    'Freeviewingstillimage_41.jpg',
    'visual search form 16_1.jpg',
    'Freeviewingstillimage_92.jpg',
    'visual search form 24_1.jpg',
    'Freeviewingstillimage_88.jpg',
    'visual search form 32_1.jpg',
    'Movie 4.mp4',
    'visual search form 4_1.jpg',
    'Freeviewingstillimage_36_cutout.tif',
    'visual search form 8_1.jpg',
    'Freeviewingstillimage_45.jpg',
    'visual search form 16_1.jpg',
    'Freeviewingstillimage_46.jpg',
    'visual search form 24_1.jpg',
    'Freeviewingstillimage_47.jpg',
    'visual search form 32_1.jpg',
    'Movie 5.mp4',
    'visual search form 4_1.jpg',
    'Moviestillimage_12.jpg',
    'visual search form 8_1.jpg',
    'Freeviewingstillimage_49.jpg',
    'visual search form 16_1.jpg',
    'Freeviewingstillimage_50.jpg',
    'visual search form 24_1.jpg',
    'Freeviewingstillimage_88_cutout.tif',
    'visual search form 32_1.jpg',
    'Closing Movie']
    trials_images.sort(reverse=True)
    stats = {}

    for trial_name in trials_images:
        # fr.plotDistanceMatrix(trial_name, vel=True)
    #     # da.plotDistanceMatrixTrialWise(trial_name)
        # st.computeDistance(trial_name)
        # st.plotTrace(trial_name)
        # ta = TraceAnalyzer(trial_name)
        # # ta.metrics()
        # ta.plotTraces()'



        # save the avg fixation
        # print(trial_name)
        # file_name =os.path.join('avg_fixation_trial_wise', trial_name + '.pkl')
        # if os.path.isfile(file_name):
        #     continue
        # st = SaliencyTrace()
        # st.computeFixationTraceForAll(trial_name)
        # f = open(os.path.join('avg_fixation_trial_wise', trial_name + '.pkl'), 'wb')
        # pkl.dump(st.avg_fixations, f)
        # f.close()
        
        #analyze the saved fixation
        # stats = {}
        print(trial_name)
        fa =FixationAnalyzer(trial_name)
        stats[trial_name[:-4]] = fa.compute_metrics()
    
    f =open('avg_fixation_stats_all_trials.pkl', 'wb')
    pkl.dump(stats, f)
    f.close()


    
