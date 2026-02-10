LINEAR REGRESSION FROM SCRATCH (PYTHON)

==================================
PROJECT OVERVIEW
==================================

This project implements Linear Regression from scratch using Python and NumPy,
without relying on machine learning libraries like scikit-learn.

The goal of this project is to:
- Understand the mathematics behind Linear Regression
- Implement Gradient Descent manually
- Learn proper ML code structure using Object-Oriented Programming (OOP)
- Handle data preprocessing, normalization, training, evaluation, and prediction


==================================
WHAT IS LINEAR REGRESSION?
==================================

Linear Regression is a supervised learning algorithm used to predict
continuous values by modeling a linear relationship between input features
(X) and output variable (y).

Mathematical Model:
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

In matrix form:
ŷ = Xw


==================================
PROJECT FEATURES
==================================

- Implemented from scratch using NumPy
- Supports:
  • Batch Gradient Descent (BGD)
  • Stochastic Gradient Descent (SGD)
  • Mini-Batch Gradient Descent (MBGD)
- Min-Max Feature Normalization
- Explicit bias handling
- Train-test split (70/30)
- Evaluation using MSE and RMSE
- User input prediction support
- Clean separation of:
  • Data pipeline
  • Model logic
  • Training & evaluation logic


==================================
PROJECT STRUCTURE
==================================

├── main.py
│   - Entry point of the program
│   - Handles training, testing, and user interaction
│
├── model.py
│   - LinearRegressionGD class
│   - Contains gradient descent logic and prediction
│
├── pipeline.py
│   - DataPipeline class
│   - Handles loading, cleaning, encoding, normalization
│
├── test_energy_data.csv
│   - Sample dataset
│
└── README.txt
    - Project documentation


==================================
GRADIENT DESCENT METHODS
==================================

1. Batch Gradient Descent (BGD)
   - Uses the entire dataset to compute gradients
   - Stable but slower for large datasets

2. Stochastic Gradient Descent (SGD)
   - Updates weights using one sample at a time
   - Faster but noisy updates

3. Mini-Batch Gradient Descent (MBGD)
   - Uses small batches of data
   - Best trade-off between speed and stability


==================================
FEATURE SCALING
==================================

Min-Max Normalization is used:

x_scaled = (x - min) / (max - min)

Why scaling is important:
- Faster convergence
- Prevents large-valued features from dominating
- Improves numerical stability


==================================
EVALUATION METRICS
==================================

1. Mean Squared Error (MSE)
   MSE = (1/m) Σ (ŷ - y)²

2. Root Mean Squared Error (RMSE)
   RMSE = √MSE

Lower values indicate better model performance.


==================================
HOW TO RUN THE PROJECT
==================================

1. Clone the repository:

   git clone <your-repo-url>
   cd <repo-folder>

2. Install requirements (see below)

3. Run the program:

   python main.py

4. Choose gradient descent type when prompted:
   bgd / sgd / mbgd

5. Enter feature values when asked to get a prediction


==================================
REQUIREMENTS
==================================

Python Version:
- Python 3.8 or higher

Python Libraries:
- numpy
- pandas

Optional (for extended evaluation):
- scikit-learn (only for metrics, not for model training)


Install dependencies using:

pip install numpy pandas scikit-learn


==================================
DATASET DETAILS
==================================

The dataset should be a CSV file containing:
- Input features (numerical or categorical)
- Target column named: "Energy Consumption"

Categorical columns are one-hot encoded using pandas.


==================================
LEARNING OUTCOMES
==================================

By working on this project, you will understand:
- Linear Regression mathematics
- Gradient Descent optimization
- Feature scaling and bias handling
- Proper ML code organization
- Difference between regression and classification pipelines


==================================
FUTURE EXTENSIONS
==================================

- Add L2 Regularization (Ridge Regression)
- Add Early Stopping
- Convert to Logistic Regression
- Add visualization of loss vs epochs
- Compare with scikit-learn implementation
- Extend to multivariate regression problems


==================================
AUTHOR
==================================

Developed as a learning project to understand Machine Learning fundamentals
and implement algorithms from scratch.

==================================
