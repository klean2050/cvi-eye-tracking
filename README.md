# Visual Saliency in Pediatric CVI through Eye Tracking

## Installation

We recommend using a conda environment with ``Python >= 3.9`` :
```
conda create -n cvi python=3.9
conda activate cvi
```
Clone the repository and install the dependencies:
```
git clone https://github.com/klean2050/cvi-eye-tracking
cd cvi-eye-tracking && pip install -r requirements.txt
```

We do not provide `.asc` files or image stimuli. If you have access to the data, modify the respective path parameters at `utils.py`. You can run a sample experiment by running:

```
python main.py
```

## Cortical Visual Impairment

Cortical visual impairment (CVI) is a leading cause of visual impairment in developed and developing countries, yet the visual characteristics that are impacted are poorly defined, and no evidence-based treatment is available. CVI occurs in children with neurologic damage to visual pathways in the brain, rather than the eye. Common causes of CVI in children include prematurity with periventricular leukomalacia, hypoxic-ischemic encephalopathy, trauma, hydrocephalus, metabolic and genetic disorders, and seizures.
 
Traditionally, the diagnosis of CVI required decreased visual acuity or field defects. Most children diagnosed using this definition had profound visual impairment or blindness at diagnosis. The diversity of visual deficits reported in children with CVI translates to a broad range of visual function and functional vision. Several unifying attributes include reduced contrast sensitivity, relatively intact color vision, selective sparing or limits in motion processing, difficulties with visual crowding, and variability in visual function based on individual and environmental factors. Since most children with CVI have developmental delays that impair communication, identifying these deficits relies on assessment of visual behavior. There is no objective measure to quantify the diversity of visual deficits.


## Eye Tracking Technology

A quantitative measure of CVI severity is important to enable accurate counseling of families, as well as prognosis and guidance for individualized therapies and accommodations. Here we develop an eye tracking-based measure of CVI severity that will satisfy the clinical and research needs described above. Eye tracking is an attractive option for measurement of visual function in CVI because in addition to being objective and quantitative, the eye tracking protocols may be designed to assess a multitude of lower- and higher-order visual characteristics. In assessing eye tracking as a potential measure of visual function in CVI, we will make use if machine learning methods.


## Experiments & Contributions

We attempt to answer the following questions:

* Which image types and properties disproportionately impact CVI patients?
* What kind of oculomotor features are more discriminative of CVI?
* Can eye-tracking provide a signature that differentiates CVI patients?

### Dataset

We selected 64 children with diagnosed CVI along with 12 controls of matching age to take part in the experiment. Each subject watched a predetermined set of visual stimuli in a specfiic order. 163 trials in total were used for each subject. To perform this analysis, we excluded subjects with missing trials (21), trials in different order (4) and slso trials at higher resolution (11). As a result, our dataset consists of 30 CVI and 10 control cases. Out of the total 163 trials, we restrict ourselves to those including still natural images.

### Visual Saliency Model

We consider various saliency modes to assess atypical visual saliency for CVI. Available modes:

* Pixel Intensity
* Spectral Residual
* Color - Opponency
* Edges - Orientation
* Perceived Depth
* Salient Objects
* Model Gradients (TBD)

TBD: Consider saliency in images through text prompts (e.g., CLIP model)

## Citation

* Rahul Sharma, PhD in Electrical and Computer Engineering
* Kleanthis Avramidis, PhD Student in Computer Science

University of Southern California, Citation TBD
