# Breast Cancer Diagnosis Prediction

This project focuses on developing machine learning models to predict breast cancer diagnosis (malignant or benign) based on cell nucleus characteristics extracted from fine needle aspirates of breast masses.

## Project Overview

The goal of this project is to build accurate prediction models that can assist medical professionals in diagnosing breast cancer. We use the Wisconsin Breast Cancer Diagnostic dataset, which contains features computed from digitized images of fine needle aspirates of breast masses.

## Dataset

The dataset contains features such as:
- Radius (mean of distances from center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness (perimeter² / area - 1.0)
- Concavity (severity of concave portions of the contour)
- Concave points (number of concave portions of the contour)
- Symmetry
- Fractal dimension ("coastline approximation" - 1)

Each feature is computed for the mean, standard error, and "worst" (mean of the three largest values) for each image, resulting in 30 features.

## Project Structure

```
breast_cancer/
├── data/                        # Data directory
│   ├── data.csv                 # Original dataset
│   └── breast_cancer_preprocessed_data.npy  # Preprocessed data
├── notebooks/                   # Jupyter notebooks
│   ├── 1_EDA.ipynb              # Exploratory Data Analysis
│   ├── 2_Data_Cleaning.ipynb    # Data Cleaning and Feature Engineering
│   ├── 3_Preprocessing.ipynb    # Data Preprocessing
│   └── 4_Modeling.ipynb         # Model Building and Evaluation
├── model/                       # Saved models
│   ├── breast_cancer_final_model.joblib
│   └── breast_cancer_model_metadata.joblib
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data preprocessing functions
│   ├── model_training.py        # Model training functions
│   └── prediction.py            # Prediction functions
├── requirements.txt             # Project dependencies
└── README.md                    # Project overview
```

## Notebooks

The analysis and modeling process is documented in Jupyter notebooks:

1. **1_EDA.ipynb**: Explores the dataset structure, distribution of features, and target classes.
2. **2_Data_Cleaning.ipynb**: Handles missing values, removes irrelevant features, and performs feature engineering.
3. **3_Preprocessing.ipynb**: Applies feature scaling, dimensionality reduction, and feature selection.
4. **4_Modeling.ipynb**: Builds, trains, and evaluates various machine learning models, including:
   - Traditional ML models (Logistic Regression, SVM, Random Forest, etc.)
   - Deep Learning models using TensorFlow/Keras

## Model Performance

We've evaluated multiple machine learning models using various metrics (accuracy, precision, recall, F1 score, ROC-AUC). The final model achieves high performance in predicting breast cancer diagnosis, with detailed metrics available in the modeling notebook.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/breast_cancer.git
   cd breast_cancer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Run the notebooks in order (1-4) to reproduce the analysis and modeling process.
2. Use the final model for predictions:

```python
import joblib
import pandas as pd

# Load the model and metadata
model = joblib.load('model/breast_cancer_final_model.joblib')
metadata = joblib.load('model/breast_cancer_model_metadata.joblib')

# Prepare input data (ensure it has all the required features)
input_data = pd.DataFrame({...})  # Your input data with the same features

# Make predictions
def predict_diagnosis(data, feature_names=metadata['feature_names'], model=model):
    # Ensure data has the required features
    if not all(feature in data.columns for feature in feature_names):
        missing = [f for f in feature_names if f not in data.columns]
        raise ValueError(f"Missing features in input data: {missing}")
    
    # Extract the required features in the correct order
    X = data[feature_names].values
    
    # Make predictions
    prediction = model.predict(X)[0]
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(X)[0, 1]
    
    # Return results
    result = {
        'prediction': 'Malignant' if prediction == 1 else 'Benign',
        'prediction_code': int(prediction)
    }
    
    if probability is not None:
        result['probability'] = float(probability)
    
    return result

# Example prediction
result = predict_diagnosis(input_data)
print(f"Prediction: {result['prediction']}")
print(f"Probability of malignancy: {result.get('probability', 'Not available')}")
```

## Results

Our final model achieves excellent performance in predicting breast cancer diagnosis. The most important features identified by the model align with medical knowledge about cellular characteristics indicative of cancerous cells.

## Future Work

- Validate the model on external datasets to ensure generalizability
- Develop an interpretable interface for medical professionals
- Incorporate additional clinical features for more comprehensive predictions
- Explore deep learning approaches for feature extraction from raw cell images

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original Wisconsin Breast Cancer Diagnostic dataset from the UCI Machine Learning Repository
- All contributors to the scikit-learn, TensorFlow, and other open-source libraries used in this project
