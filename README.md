# Hybrid Text & Behavioural Classification Engine

Machine learning system that combines natural language processing (NLP) features and structured survey-response data to classify artwork perceptions using a custom multiclass logistic regression implementation.

---

## Project Overview

This project investigates whether subjective emotional reactions and free-text survey responses can be used to identify which artwork a participant is viewing.

Participants were shown a series of paintings and asked to provide both structured ratings and open-ended textual responses describing their perceptions, emotions, and associations. These responses were transformed into machine learning features and used to train a multiclass classification system capable of predicting the underlying artwork.

The system classifies responses into one of three paintings:

* *The Persistence of Memory* — Salvador Dalí
* *The Starry Night* — Vincent van Gogh
* *The Water Lily Pond* — Claude Monet

---

## Key Features

* Custom multiclass logistic regression implementation using NumPy
* Softmax probability outputs and cross-entropy loss optimization
* L2 regularization and vectorized gradient descent
* Bag-of-Words NLP feature extraction pipeline
* Hybrid text and numeric feature representation
* Data cleaning, imputation, and normalization pipeline
* Model comparison against Random Forest and Decision Tree baselines
* Hyperparameter tuning experiments
* Saved model artifacts for reproducible inference
* Technical report documenting methodology and results

---

## Machine Learning Pipeline

```text
Raw Survey Responses
        ↓
Data Cleaning & Validation
        ↓
Numeric Features + Text Features
        ↓
Bag-of-Words Feature Extraction
        ↓
Hybrid Feature Matrix
        ↓
Feature Normalization
        ↓
Model Training & Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Artwork Classification
```

---

## Repository Structure

```text
hybrid-text-behavioural-classification/
│
├── data/
│   ├── raw/
│   │   └── ml_challenge_dataset.csv
│   └── processed/
│       └── ml_challenge_cleaned_dataset.csv
│
├── docs/
│   └── ml_challenge_survey_questions.txt
│
├── models/
│   └── saved/
│       ├── W.npy
│       ├── b.npy
│       ├── vocab.npy
│       ├── bow_mean.npy
│       ├── bow_std.npy
│       ├── num_mean.npy
│       └── num_std.npy
│
├── reports/
│   ├── figures/
│   │   ├── pipeline.png
│   │   ├── model_comparison.png
│   │   ├── learning_rate_tuning.png
│   │   ├── lambda_tuning.png
│   │   └── numeric_feature_distributions.png
│   │
│   ├── scripts/
│   │   ├── model_comparison.py
│   │   ├── hyperparameter_tuning.py
│   │   └── numeric_feature_distributions.py
│   │
│   └── Hybrid_Text_and_Behavioural_Classification_Engine_Technical_Report.pdf
│
├── src/
│   └── hybrid_text_classification/
│
├── tests/
│
├── pred.py
├── data_visualization.py
├── hyperparameter_tuning.py
├── model_comparison.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Dataset

The dataset consists of student survey responses associated with three artwork classes.

Each observation contains:

### Numeric Features

* Emotion intensity rating
* Sombre rating
* Content rating
* Calm rating
* Uneasy rating
* Number of prominent colours
* Number of noticed objects
* Willingness to pay

### Text Features

* Emotional descriptions
* Food analogies
* Seasonal associations
* Room placement preferences
* Social viewing preferences
* Imagined soundtracks

The combination of structured and unstructured data creates a heterogeneous classification problem that requires both NLP and traditional machine learning techniques.

---

## NLP Pipeline

All text-based survey fields are concatenated into a single document for each respondent.

The documents are then:

1. Tokenized using regex-based preprocessing
2. Converted into a Bag-of-Words representation
3. Transformed into sparse feature vectors
4. Combined with cleaned numeric survey features

The resulting feature space is high-dimensional and sparse, making logistic regression particularly effective for classification.

---

## Model

The final model is a custom implementation of multiclass logistic regression using NumPy.

Features include:

* Softmax activation
* Cross-entropy loss
* Full-batch gradient descent
* L2 regularization
* Vectorized gradient computation
* Probability-based prediction

For each sample:

[
Z = XW^T + b
]

Class probabilities are computed using the softmax function and optimized through cross-entropy minimization.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/shenjie9/Hybrid-NLP-Behavioral-Classification-Engine.git

cd Hybrid-NLP-Behavioral-Classification-Engine

pip install -r requirements.txt
pip install -e .
```

---

## Running Predictions

Using the package:

```python
from hybrid_text_classification import predict_all

predictions = predict_all(
    "data/raw/ml_challenge_dataset.csv"
)

print(predictions[:10])
```

For compatibility:

```python
from pred import predict_all

predictions = predict_all(
    "data/raw/ml_challenge_dataset.csv"
)

print(predictions[:10])
```

---

## Training the Model

Train the custom logistic regression model:

```bash
python -m hybrid_text_classification.train \
    --data data/raw/ml_challenge_dataset.csv \
    --model-dir models/saved
```

Example with custom hyperparameters:

```bash
python -m hybrid_text_classification.train \
    --data data/raw/ml_challenge_dataset.csv \
    --model-dir models/saved \
    --lr 0.05 \
    --n-iters 3000 \
    --l2 0.5 \
    --min-freq 5
```

---

## Model Comparison

Generate model comparison results:

```bash
python model_comparison.py
```

The project compares:

* Custom Logistic Regression
* Random Forest
* Random Forest + Naive Bayes Hybrid
* Decision Tree

Results show that logistic regression performs best on the sparse text-driven feature space.

---

## Hyperparameter Tuning

Hyperparameters explored include:

* Learning rate
* Number of training iterations
* Vocabulary frequency threshold
* L2 regularization strength

Generate tuning figures:

```bash
python hyperparameter_tuning.py
```

Generated outputs include:

* Learning rate validation curves
* L2 regularization validation curves
* Hyperparameter comparison figures

---

## Visualizations

Generate numeric feature distributions:

```bash
python reports/scripts/numeric_feature_distributions.py
```

Outputs:

```text
reports/figures/numeric_feature_distributions.png
```

The repository also contains:

* Pipeline diagram
* Model comparison plots
* Learning rate tuning plots
* Regularization tuning plots

---

## Testing

Run the test suite:

```bash
pytest
```

The tests verify:

* Softmax probability correctness
* Bag-of-Words feature construction
* Logistic regression training behaviour
* Core preprocessing functionality

---

## Results

The final logistic regression model achieved approximately:

* 89% training accuracy
* 91% validation accuracy

Key findings:

* Text features were substantially more informative than numeric features alone.
* Logistic regression outperformed tree-based approaches on sparse text data.
* Combining NLP-derived features with behavioural survey responses produced the strongest predictive performance.

---

## Technical Report

A complete technical report is included in:

```text
reports/Hybrid_Text_and_Behavioural_Classification_Engine_Technical_Report.pdf
```

The report covers:

* Data preparation
* Feature engineering
* NLP methodology
* Model implementation
* Hyperparameter tuning
* Experimental results
* Discussion and conclusions

---

## Skills Demonstrated

* Python
* NumPy
* Pandas
* Machine Learning
* Natural Language Processing (NLP)
* Logistic Regression
* Feature Engineering
* Data Cleaning
* Hyperparameter Tuning
* Data Visualization
* Software Design
* Scientific Reporting
