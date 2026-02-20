LOGISTIC REGRESSION FROM SCRATCH (PYTHON)

=====================================
PROJECT OVERVIEW
=====================================

This project implements Logistic Regression from scratch using Python and NumPy,
without using machine learning libraries such as scikit-learn for model training.

The implementation is designed to:
- Understand the mathematics behind Logistic Regression
- Implement Gradient Descent manually
- Build a clean ML pipeline using Object-Oriented Programming (OOP)
- Perform binary classification on a real-world medical dataset
- Evaluate the model using proper classification metrics


=====================================
WHAT IS LOGISTIC REGRESSION?
=====================================

Logistic Regression is a supervised learning algorithm used for
binary classification problems.

Instead of predicting continuous values, it predicts the
probability that an input belongs to a particular class.

The output is mapped to a probability using the sigmoid function.

Hypothesis:
ŷ = σ(Xw)

Where:
σ(z) = 1 / (1 + e⁻ᶻ)


=====================================
DATASET
=====================================

Dataset Used:
Breast Cancer Wisconsin Dataset

Target Column:
- diagnosis
  • M (Malignant) → 1
  • B (Benign) → 0

Dropped Columns:
- id
- Unnamed: 32

The dataset contains only numerical features after preprocessing,
making it suitable for Logistic Regression.


=====================================
PROJECT FEATURES
=====================================

- Logistic Regression implemented from scratch
- Supports:
  • Batch Gradient Descent (BGD)
  • Stochastic Gradient Descent (SGD)
  • Mini-Batch Gradient Descent (MBGD)
- Standardization using training data statistics
- Explicit bias term handling
- Train-test split (70/30)
- Shuffling to avoid sampling bias
- Binary classification output
- Probability-based prediction
- Evaluation using:
  • Confusion Matrix
  • Precision, Recall, F1-score
  • Accuracy


=====================================
PROJECT STRUCTURE
=====================================

├── main.py
│   - Entry point of the program
│   - Handles training, evaluation, and user interaction
│
├── model.py
│   - LogisticRegressionGD class
│   - Contains sigmoid, gradient descent logic, and prediction
│
├── pipeline.py
│   - DataPipeline class
│   - Handles loading, preprocessing, standardization
│
├── data.csv
│   - Breast Cancer dataset
│
└── README.md / README.txt
    - Project documentation


=====================================
MODEL DETAILS
=====================================

Sigmoid Function:
σ(z) = 1 / (1 + e⁻ᶻ)

Loss Function:
Binary Cross-Entropy (Log Loss)

Gradient:
∇J = (1/m) · Xᵀ(ŷ − y)

This loss function is derived from Maximum Likelihood Estimation
and is well-suited for probabilistic classification.


=====================================
GRADIENT DESCENT METHODS
=====================================

1. Batch Gradient Descent (BGD)
   - Uses entire dataset per update
   - Stable but slower

2. Stochastic Gradient Descent (SGD)
   - Updates weights using one sample
   - Faster but noisy updates

3. Mini-Batch Gradient Descent (MBGD)
   - Uses small batches
   - Best balance between speed and stability


=====================================
FEATURE SCALING
=====================================

Standardization is used:

x_scaled = (x − mean) / std

Why standardization?
- Faster convergence
- Improves numerical stability
- Essential for gradient-based optimization

Mean and standard deviation are computed
ONLY on training data to prevent data leakage.


=====================================
EVALUATION METRICS
=====================================

Confusion Matrix:
- TP: True Positive
- TN: True Negative
- FP: False Positive
- FN: False Negative

Metrics:
- Accuracy
- Precision
- Recall
- F1-score

These metrics provide a better evaluation
than accuracy alone, especially for medical datasets.


=====================================
HOW TO RUN THE PROJECT
=====================================

1. Clone the repository:

   git clone <your-repo-url>
   cd <repo-folder>

2. Install dependencies

3. Run the program:

   python main.py

4. Select gradient descent type:
   bgd / sgd / mbgd

5. Enter feature values when prompted
   to get a prediction for a new sample


=====================================
REQUIREMENTS
=====================================

Python Version:
- Python 3.8 or higher

Required Libraries:
- numpy
- pandas

Optional (for evaluation metrics only):
- scikit-learn

Install dependencies using:

pip install numpy pandas scikit-learn


=====================================
LEARNING OUTCOMES
=====================================

By completing this project, you will understand:
- Logistic Regression mathematics
- Sigmoid activation and log-loss
- Gradient Descent optimization
- Difference between regression and classification
- Proper ML pipeline design
- Avoiding data leakage
- Model evaluation for classification problems


=====================================
FUTURE EXTENSIONS
=====================================

- Add L2 Regularization (Ridge Logistic Regression)
- Add ROC-AUC and threshold tuning
- Implement Early Stopping
- Compare with scikit-learn LogisticRegression
- Extend to Multiclass Classification (Softmax)
- Add feature importance analysis


=====================================
AUTHOR
=====================================

Developed as a learning project to deeply understand
Machine Learning classification algorithms and optimization
by implementing them from scratch.

=====================================
