# Fruit Quality Classifier using SVM
Ilias Alexandropoulos, mtn2302

Spyridon Georgoulas, mtn2309

Vasiliki Rentoula, mtn2317
## Description

This project is an SVM-based fruit quality classifier that categorizes fruits into 3 quality levels (Good, Bad, Mixed) based on features extracted from images ([Dataset](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality/data)). The project includes preprocessing of images, feature extraction, training an SVM classifier, and testing its performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Extract Features]($extract-features)
- [Training the model](#training-the-model)
- [Testing the model](#testing-the-model)


## Installation

```bash
git clone https://github.com/spygeorgoulas/fruit-quality.git
```
## Usage
```bash
cd fruit-quality
python3 -m vevn venv
pip install -r requirments.txt
```

## Data
[Dataset](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality/data) from kaggle.

## Extract features
Extract training and test features from the dataset folder and saves them at:
- `fruit-quality/src/train_pipeline/features_train` 
- `fruit-quality/src/train_pipeline/features_test` 


```bash
python3 -m src.train_pipeline.pipeline -i 'data'
```
`-i`: Path to dataset folder


## Training the model
```bash
python3 src/train_pipeline/train.py -train src/train_pipeline/features_train/ -test src/train_pipeline/features_test/
```

`-train`: Path to the train features folder

`-test`: Path to the test features folder

## Testing the model
