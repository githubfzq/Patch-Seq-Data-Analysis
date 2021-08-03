# Patch-Seq-Data-Analysis

This repository is the pipeline of **electrophysiological** and **morphological** analysis.

The electrophysiological analysis includes **downloading NWB data**, **plotting trace**, 
**extracting electrophysiological features** according to NWB files;
and morphological analysis includes **downloading SWC data**, **plotting morphology**, 
**extracting morphological features** from SWC files, and **converting formats**.

The electrophysiological analysis mainly relies on [dandi/dandi-cli][3] (downloading data), 
[AllenInstitute/ipfx][4] (extracting features),
and the morpholocial analysis is mainly based on [AllenInstitute/neuron_morphology][5] (extracting features),
[BlueBrain/NeuroM][6] (plotting).


# Sub-projects

## Cell

The electrophysiological and morphological analysis of article [Integrated Morphoelectric and Transcriptomic Classification of Cortical GABAergic Cells][1].

## Nature

The analysis pipeline of article [Phenotypic variation of transcriptomic cell types in mouse motor cortex][2].

## CJW

The morphological analysis pipeline of our own lab re-using the util methods of two articles above. 

# Files

- **notebooks**: Jupyter notebooks including electrophysiological (`electro.ipynb`) and morphological (`morpho.ipynb`)
are in each sub-projects.

- **utils**: The utility functions for each sub-projects, including electrophysiological (`electro_utils.py`) 
and morphological (`morpho_utils.py`) scripts. 

[1]: https://www.sciencedirect.com/science/article/abs/pii/S009286742031254X

[2]: https://www.nature.com/articles/s41586-020-2907-3

[3]: https://github.com/dandi/dandi-cli

[4]: https://github.com/AllenInstitute/ipfx

[5]: https://github.com/AllenInstitute/neuron_morphology

[6]: https://github.com/BlueBrain/NeuroM