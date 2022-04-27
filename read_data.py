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

class Subject():
    def __init__(self, subject_id):
        self.id = subject_id
        self.asc_file = f'../asc_data_v1/{self.id}.asc'
        self.data = open(self.asc_file, 'r')
    
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
        
    def computeTrace(self, trial_name, data):
        saliency_map = np.load(f'../smaps/gen/{trial_name}.npy').squeeze()
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

    def computeTraceForALL(self, trial_name):
        self.readAllData(trial_name)
        subject_ids = [k for k in self.data_frac.keys() if self.data_frac[k] > 0.5]
        subject_ids.sort()
        self.trace = {}
        for subject in subject_ids:
            self.trace[subject] = self.computeTrace(trial_name, self.timeseries[subject])
    
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
            


if __name__ == '__main__':
    fr = FeatureRepresentation()
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
    for trial_name in trials_images:
        fr.plotDistanceMatrix(trial_name, vel=True)
    #     # da.plotDistanceMatrixTrialWise(trial_name)
    #     st.computeDistance(trial_name)

    
