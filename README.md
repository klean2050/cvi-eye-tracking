# Visual Saliency in Pediatric CVI through Eye Tracking

## Cortical Visual Impairment

Cortical visual impairment (CVI) is a leading cause of visual impairment in developed and developing countries, yet the visual characteristics that are impacted are poorly defined, and no evidence-based treatment is available. CVI occurs in children with neurologic damage to visual pathways in the brain, rather than the eye. Common causes of CVI include prematurity with periventricular leukomalacia, hypoxic-ischemic encephalopathy, trauma, hydrocephalus, metabolic and genetic disorders, and seizures.
 
Traditionally, the diagnosis of CVI required decreased visual acuity and/or visual field defects. Most children diagnosed using this definition had profound visual impairment or blindness at diagnosis. The diversity of visual deficits reported in children with CVI translates to a broad range of visual function and functional vision. Several unifying characteristics include reduced contrast sensitivity, relatively intact color vision, selective sparing or limitation of motion processing, difficulties with visual crowding, and variability in visual function based on individual and environmental factors. Since most children with CVI have developmental delays that impair communication, identifying these deficits relies on assessment of visual behavior. There is currently no objective, validated method of quantifying the diversity of visual deficits.


## Eye Tracking Technology

A quantitative measure of CVI severity is important to enable accurate counseling of families, as well as prognostication and guidance for individualized therapies and accommodations. Here we develop an eye tracking-based measure of CVI severity that will satisfy the clinical and research needs described above. Eye tracking is an attractive option for measurement of visual function in CVI because in addition to being objective and quantitative, the eye tracking protocols may be designed to assess a multitude of lower- and higher-order visual characteristics. In assessing eye tracking as a potential measure of visual function in CVI, we will use machine learning methods to process the eye tracking data.


## Experiments & Contributions

We attempt to answer the following questions:

* Which image types & properties disproportionately impact CVI patients?
* What kind of oculomotor features are more discriminative of CVI?
* Can eye-tracking provide a signature that differentiates CVI patients?

### Dataset

We selected 64 children with diagnosed CVI along with 12 control subjects of matching age to take part in the experiment. Each subject watched a predetermined set of visual stimuli in a specfiic order. 163 trials in total were used for each subject. To perform this analysis, we excluded subjects with missing trials (21), trials in different order (4) and trials at higher resolution (11). As a result, our dataset consists of 30 CVI and 10 control cases. Out of the 163 trials, we restrict ourselves to those including still images (TBD).

### Visual Saliency Model

* We compute the trace of the eye tracks with respect to an extracted saliency map.
* We compare the trace for each subject for a particular trial against all other subjects (with DTW).

#### Comparing the trials

* We use the following comparable trials:
    * [‘Freeviewingstillimage_36.jpg’, ‘Freeviewingstillimage_36_cutout.tif’],
        [‘Freeviewingstillimage_28.jpg’, ‘Freeviewingstillimage_28_cutout.tif’],
        [‘Freeviewingstillimage_93.jpg’, ‘Freeviewingstillimage_93_cutout.tif’],
        [‘Freeviewingstillimage_36.jpg’, ‘Freeviewingstillimage_36_cutout.tif’],
        [‘Freeviewingstillimage_10.jpg’, ‘Freeviewingstillimage_10_cutout.tif’]
* We aim to see if the difference in more prominent in one group than other.

## Related Work

https://arxiv.org/pdf/2009.02437.pdf
Learning a stimulus agnostic representations for the eye track data.
Using an u-net type architecture for autoencoder to learn the representation for the eye tracks. Using the representations, classify for ctrl/CVI.
[Image: Screen Shot 2022-03-02 at 11.58.23 AM.png]This architecture can learn the representations at different scales, z1 and z2 being macro and micro scales.
This type of learned representations are shown to work well for classification setups: classifying the stimuli type, age group, gender, even for biometrics.
Unsupervised:
* Using all the eyetracks irrespective of stimuli or the group to learn a model fro representing eyetracks in a unsupervised fashion.
* Since the representations are stimulus agnostic, get the representation for the two groups for particular trials to understand the importance of the particular trial. For ex: comparing the representations of the two groups (clustering the representations of all the subjects for the stimuli pertaining to the visual search/ visual field defects.)
1. Get the attention trace
    1. http://ilab.usc.edu/publications/doc/Tseng_etal13ideal.pdf
[Image: Screen Shot 2022-03-02 at 12.24.40 PM.png][Image: Screen Shot 2022-03-02 at 3.42.08 PM.png]
1. Generate spectograms for the trace.
    1. https://ieeexplore-ieee-org.libproxy2.usc.edu/stamp/stamp.jsp?tp=&arnumber=9581943
2. Use image like operations, using CNN/TCNs train a classifier for ctrl/CVI
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
