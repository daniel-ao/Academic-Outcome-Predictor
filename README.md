# Academic Success Classification Project

This project applies machine learning techniques to predict students' academic outcomes (Graduate, Dropout, or Enrolled) using a dataset of demographic, academic, and behavioral characteristics.

## Dataset Overview

The dataset comprises two files:
- `train.csv`: Contains features and a target column for training.
- `test.csv`: Contains features only for testing.

### Key Columns
- **Demographics**: Gender, Age, Nationality, Marital Status.
- **Academic Background**: Previous qualifications, Admission grade.
- **Family Information**: Parents' education and occupation.
- **Behavioral**: Number of curricular units enrolled, approved, etc.
- **Economic Context**: Unemployment rate, Inflation rate, GDP.
- **Target (train.csv only)**: Graduate, Dropout, or Enrolled.

The dataset has 38 columns in `train.csv` and 37 columns in `test.csv`.

---

## Methodology

### 1. **Data Preprocessing**
- Handled missing values using median (numeric) or "Unknown" (categorical).
- Applied one-hot encoding for categorical variables.
- Normalized numerical features.
- Converted the target into a binary format (Graduate = 1, Dropout = 0).

### 2. **Models**
- **Logistic Regression**: For simple binary classification.
- **Neural Network**: A more complex model with:
  - Input layer: Features as neurons.
  - Hidden layer: 32 neurons with ReLU activation.
  - Output layer: Single neuron with Sigmoid for binary classification.

### 3. **Loss Functions**
- **Logistic Regression**: Binary Cross-Entropy Loss.
- **Neural Network**: Cross-Entropy Loss for multi-class output.

### 4. **Optimization**
- **Logistic Regression**: Stochastic Gradient Descent (SGD).
- **Neural Network**: Adam optimizer with learning rate scheduling.

### 5. **Evaluation**
- Data split into training and validation sets (80/20 split).
- Metrics tracked:
  - **Loss**: Training and validation losses.
  - **Accuracy**: Validation accuracy for model performance comparison.
- Hyperparameter tuning conducted to optimize performance.

---

## Results

- **Logistic Regression**:
  - Fluctuating validation loss indicates underfitting.
- **Neural Network**:
  - Stable loss curve and higher accuracy (~88% after 20 epochs).
  - Demonstrates better generalization and predictive power.

---

## Submission to Kaggle

- Final predictions are saved in `submission.csv`.
- Successfully submitted with a positive leaderboard score.

---

## Running the Code

1. Set up a virtual environment with required libraries (`PyTorch`, `Pandas`, etc.).
2. Execute the Python script to:
   - Preprocess data.
   - Train models (logistic regression and neural network).
   - Evaluate performance.
   - Generate Kaggle submission file.

---

## Key Code Functions

- `load_data`: Load dataset from CSV.
- `preprocess_data`: Clean and encode data.
- `define_models`: Initialize models (logistic regression, neural network).
- `train_and_validate`: Train models and track performance.
- `hyperparameter_tuning`: Optimize hyperparameters.
- `generate_submission`: Create submission file for Kaggle.
