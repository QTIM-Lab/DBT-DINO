# DBT-DINO

This repository contains the code necessary to run training and inference on the DBT-DINO model.

## Background
Digital Breast Tomosynthesis (DBT) is a pseudo-3D breast imaging modality approved by the FDA in 2011 for breast cancer screening. Compared to traditional 2D mammography, DBT can improve cancer detection rates and reduce false-positives, especially in women with dense breast tissue. 

DBT-DINO is a foundation model for DBT, pre-trained using self-supervised learning (DINOv2 methodology) on over 25 million 2D slices from 487,975 DBT volumes from 27,990 patients. This model can be fine-tuned for various downstream tasks including breast density classification, cancer risk prediction, and lesion detection. Read more about it in our pre-print: https://arxiv.org/pdf/2512.13608

## Paper Code
The code used to train the downstream models presented in the paper can be found in src. Bash scripts are provided that show how the training and evaluation was performed.

## Using DBT-DINO
### Embedding Extraction
The weights for DBT-DINO are provided on Zenodo: https://zenodo.org/records/17981813. 

The extract_embeddings.py script provides a code example that can be used to extract embeddings from a DBT volume.

### Downstream Tasks
Coming soon

### Example Notebooks
Coming soon

### Citation
TBD