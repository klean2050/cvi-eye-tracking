# CVI eyetrack data representations 

## CVI problem formulation

### Visual saliency models:

[Image: Screen Shot 2022-03-02 at 11.51.42 AM.png]Using Itti’s visual saliency models, we can obtain a representation for the visual stimulus.
C, color; I, intensity; O, orientation; F, flicker; M, motion; J, line junction.

1. In case of videos, we can obtain the stimulus for the each frame. 



### Representation learning for eye-track data

#### 1.Unsupervised representations:

https://arxiv.org/pdf/2009.02437.pdf
Learning a stimulus agnostic representations for the eye track data.
Using an u-net type architecture for autoencoder to learn the representation for the eye tracks. Using the representations, classify for ctrl/CVI.
[Image: Screen Shot 2022-03-02 at 11.58.23 AM.png]This architecture can learn the representations at different scales, z1 and z2 being macro and micro scales. 

This type of learned representations are shown to work well for classification setups: classifying the stimuli type, age group, gender, even for biometrics.
Unsupervised:

* Using all the eyetracks irrespective of stimuli or the group to learn a model fro representing eyetracks in a unsupervised fashion.
* Since the representations are stimulus agnostic, get the representation for the two groups for particular trials to understand the importance of the particular trial. For ex: comparing the representations of the two groups (clustering the representations of all the subjects for the stimuli pertaining to the visual search/ visual field defects.)



#### Stimulus oriented representations (supervised):

1. Get the attention trace
    1. http://ilab.usc.edu/publications/doc/Tseng_etal13ideal.pdf

[Image: Screen Shot 2022-03-02 at 12.24.40 PM.png][Image: Screen Shot 2022-03-02 at 3.42.08 PM.png]
1. Generate spectograms for the trace.
    1. https://ieeexplore-ieee-org.libproxy2.usc.edu/stamp/stamp.jsp?tp=&arnumber=9581943
2. Use image like operations, using CNN/TCNs train a classifier for ctrl/CVI

#### Data set

* Represent an eyetrack using the attention trace, further computing a spectogram for the signal, thus the input depends on the stimulus as well. Now train the system in a supervised fashion for the task of classification into cvi/ctrl.
* Once trained, visualize the network for the important regions in the spectogram which enhance the differentiation between the two groups. These can give us information about the particular regions in the eyetracks, which are different among the two groups. For example it may capture the particular occurrences of saccades, or velocity of saccade which makes a difference between the two groups. 
* Advantage: can help us analyse more fundamentally, the difference between the groups. While the unsupervised method can help us validate the cinical hypothesis, using the  
* Limitations: 
    * Still accessing the amount of data that we have is sufficient?
    * The training data needs to be for the same stimulus.




## Related work

* 1. https://arxiv.org/pdf/2009.02437.pdf
    1. learning auto-encoders to represent the temporal signal.
    2. Uses dialated convolutions.
    3. Similar to u-net architecture.
2. https://ieeexplore-ieee-org.libproxy1.usc.edu/stamp/stamp.jsp?tp=&arnumber=9581943
3. [Deep learning on natural viewing behaviors to differentiate children with fetal alcohol spectrum disorder](http://ilab.usc.edu/publications/doc/Tseng_etal13ideal.pdf)
    1. Use the saliency models to compute the representation of the stimuli. 
    2. Compute the single dimension temporal representation of the signal. 
    3. Use the CNN (TCNs) to represent the time series
    4. Learn a classifier
4. https://github.com/chipbautista/gazemae


## Experiments

#### Setting

* We have 163 trials for each subject.
* Here we use just the still images.
* We compute the trace of the eyetracks with respect to the saliency map as shown:
* [Image: Screen Shot 2022-03-02 at 3.42.08 PM.png]
* Compare the trace for each subject for a particular trial against all other subjects.
* We use the DTW as the metric to compare the distance between two time series. 
* The objective is to understand which trials can distinguish two groups. 

#### Distinguishing behavior inherent to trials

[Image: cvi-distance-trace.png]

#### Interesting cases

[Image: ineresting-cases.png]
#### Comparing the trials

* We use the following comparable trials:
    * ['Freeviewingstillimage_36.jpg', 'Freeviewingstillimage_36_cutout.tif'],
        ['Freeviewingstillimage_28.jpg', 'Freeviewingstillimage_28_cutout.tif'],
        ['Freeviewingstillimage_93.jpg', 'Freeviewingstillimage_93_cutout.tif'],
        ['Freeviewingstillimage_36.jpg', 'Freeviewingstillimage_36_cutout.tif'],
        ['Freeviewingstillimage_10.jpg', 'Freeviewingstillimage_10_cutout.tif']
* Compare the trace for each trial in the pair for every subject.
* Objective is to see if the difference in more prominent in one group than other. 


### Next steps

* experiment with distance metrics to better quantify the distance. 
* Qualitative analysis of the distance matrices. 
* Constructive inference.

[Image: comparison_36.png][Image: comparison_93.png][Image: comparison_10.png][Image: comparison_28.png]