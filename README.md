# Visual Saliency in Pediatric CVI through Eye Tracking

## Installation

We recommend using a conda environment with ``Python >= 3.9`` :

```bash
conda create -n cvi python=3.9
conda activate cvi
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/klean2050/cvi-eye-tracking
cd cvi-eye-tracking && pip install -r requirements.txt
```

You will also need to clone [this repository](https://github.com/matthias-k/DeepGaze). We do not provide `.asc` files or image stimuli. If you have access to the data, modify the respective path parameters at `utils.py`. You can run a sample experiment using the following command:

```bash
python main.py
```

## Cortical Visual Impairment

Cortical visual impairment (CVI) is a leading cause of visual impairment in developed and developing countries, yet the visual characteristics that are impacted are poorly defined, and no evidence-based treatment is available. CVI occurs in children with neurologic damage to visual pathways in the brain, rather than the eye. Traditionally, the diagnosis of CVI required decreased visual acuity or field defects. Most children diagnosed using this definition had profound visual impairment or blindness at diagnosis. The diversity of visual deficits reported in children with CVI translates to a broad range of visual function and functional vision. Since most children with CVI have developmental delays that impair communication, identifying these deficits relies on assessment of visual behavior. There is no objective measure to quantify the diversity of visual deficits.

## Eye Tracking Technology

A quantitative measure of CVI severity is important to enable accurate counseling of families, as well as prognosis and guidance for individualized therapies and accommodations. Here we develop an eye tracking-based measure of CVI severity that will satisfy the clinical and research needs described above. Eye tracking is an attractive option for measurement of visual function in CVI because in addition to being objective and quantitative, the eye tracking protocols may be designed to assess a multitude of lower- and higher-order visual characteristics.

## Experiment

We selected children with diagnosed CVI along with controls of matching age to take part in the experiment. Each subject watched a predetermined set of naturalistic visual stimuli in a specfic order. This repository contains classes and functions to analyze the obtained eye tracking recordings and assess group differences in terms of visual saliency. We consider multiple saliency modes to assess atypical visual saliency in CVI. Available modes:

* Pixel Intensity
* Spectral Residual
* Color - Opponency
* Edges - Orientation
* Perceived Depth
* DeepGaze IIE Maps
* Ours: prompt-based visual saliency

## Citation

We hope that the shared implementation can assist in comprehending our documented method and also in informing similar studies on visual saliency. If you do use this work in your study, please consider citing:

```bibtex
@article{avramidis2024cvi,
  title={Evaluating Atypical Gaze Patterns through Vision Models: The Case of Cortical Visual Impairment},
  author={Avramidis, Kleanthis and Chang, Melinda Y and Sharma, Rahul and Borchert, Mark S and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2402.09655, currently under submission for IEEE EMBC 2024},
  year={2024}
}
```

* Rahul Sharma, PhD in Electrical and Computer Engineering, Amazon
* Kleanthis Avramidis, PhD Student in Computer Science, USC
