# Customer Churn Prediction

A machine learning–based web application that predicts whether a bank customer is likely to churn based on demographic, financial, and account activity attributes.

The application uses a trained neural network model with serialized preprocessing components and provides an interactive **Streamlit** interface for real-time churn prediction.

---

## Overview

Customer churn prediction is a binary classification problem where the objective is to determine whether a customer is likely to leave a banking service.

This project implements an end-to-end machine learning inference pipeline:

```text
User Input
    ↓
Feature Construction
    ↓
Categorical Encoding
    ↓
Feature Alignment
    ↓
Feature Scaling
    ↓
Neural Network Model
    ↓
Churn Probability
    ↓
Churn / No Churn
```

The application accepts customer information such as credit score, geography, age, balance, tenure, number of products, credit card ownership, activity status, and estimated salary.

---

## Features

* Interactive customer input interface using Streamlit
* Binary churn prediction
* One-hot encoding for geographical features
* Label encoding for gender
* Standardized numerical features using `StandardScaler`
* TensorFlow/Keras neural network inference
* Serialized preprocessing artifacts using Pickle
* Real-time prediction through a web interface
* Probability-based classification using a configurable threshold

---

## Tech Stack

### Programming Language

* Python

### Machine Learning

* TensorFlow / Keras
* Scikit-learn
* NumPy
* Pandas

### Data Processing

* `OneHotEncoder`
* `LabelEncoder`
* `StandardScaler`

### Web Application

* Streamlit

### Model & Artifact Serialization

* Pickle
* Keras `.h5` model format

---

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── model.h5
├── scaler.pkl
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                       | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `app.py`                   | Streamlit application and inference pipeline       |
| `model.h5`                 | Trained TensorFlow/Keras neural network            |
| `scaler.pkl`               | Fitted `StandardScaler` used during model training |
| `onehot_encoder_geo.pkl`   | Fitted encoder for Geography                       |
| `label_encoder_gender.pkl` | Fitted encoder for Gender                          |
| `requirements.txt`         | Python dependencies                                |
| `README.md`                | Project documentation                              |

---

## Input Features

The application accepts the following customer attributes:

| Feature           | Description                                         |
| ----------------- | --------------------------------------------------- |
| `CreditScore`     | Customer's credit score                             |
| `Geography`       | Customer's geographical region                      |
| `Gender`          | Customer gender                                     |
| `Age`             | Customer age                                        |
| `Tenure`          | Number of years the customer has been with the bank |
| `Balance`         | Customer's account balance                          |
| `NumOfProducts`   | Number of banking products used                     |
| `HasCrCard`       | Whether the customer owns a credit card             |
| `IsActiveMember`  | Whether the customer is an active bank member       |
| `EstimatedSalary` | Estimated annual salary                             |

---

## Data Preprocessing

The preprocessing pipeline is designed to replicate the transformations applied during model training.

### 1. Geography Encoding

`Geography` is a categorical feature and is transformed using a fitted `OneHotEncoder`.

For example:

```text
Germany
```

may be transformed into a representation such as:

```text
Geography_France
Geography_Germany
Geography_Spain
```

The encoder is loaded from:

```text
onehot_encoder_geo.pkl
```

This ensures that inference uses the same categorical representation as training.

---

### 2. Gender Encoding

The `Gender` feature is transformed using a previously fitted `LabelEncoder`.

```python
label_encoder_gender.transform([gender])[0]
```

The serialized encoder is loaded from:

```text
label_encoder_gender.pkl
```

---

### 3. Feature Alignment

The final inference DataFrame must contain exactly the same features used when fitting the scaler.

```python
input_data = input_data[scalar.feature_names_in_]
```

This prevents feature-name and feature-order mismatches between training and inference.

---

### 4. Feature Scaling

The processed input is standardized using the fitted `StandardScaler`.

```python
input_data_scaled = scalar.transform(input_data)
```

The scaler is loaded from:

```text
scaler.pkl
```

Using the same fitted scaler is important because the neural network expects inputs transformed using the same feature distributions used during training.

---

## Model Architecture

The prediction model is implemented using **TensorFlow/Keras**.

The model receives the preprocessed feature vector and outputs a probability representing the likelihood of customer churn.

Conceptually:

```text
Preprocessed Features
        ↓
Dense Neural Network
        ↓
Output Layer
        ↓
Churn Probability
```

The model is loaded during application startup:

```python
model = tf.keras.models.load_model("model.h5")
```

---

## Prediction

The model generates a probability:

```python
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
```

A threshold of `0.5` is used to convert the probability into a binary prediction:

```python
if prediction_proba > 0.5:
    st.write("person is likely to churn")
else:
    st.write("person is not likely to churn")
```

Therefore:

```text
Probability > 0.5  → Likely to Churn
Probability ≤ 0.5 → Not Likely to Churn
```

The threshold can be adjusted depending on the desired trade-off between false positives and false negatives.

---

## Application Workflow

When a user submits customer information, the application performs the following operations:

### Step 1 — Collect User Input

Streamlit widgets collect customer attributes:

```python
geography = st.selectbox(...)
gender = st.selectbox(...)
age = st.slider(...)
balance = st.number_input(...)
```

### Step 2 — Construct Feature DataFrame

The numerical and encoded categorical features are combined into a Pandas DataFrame.

### Step 3 — Encode Categorical Features

Geography is one-hot encoded and Gender is label encoded.

### Step 4 — Align Features

The feature columns are reordered to match the columns used during scaler fitting.

### Step 5 — Scale Features

The input is transformed using the serialized `StandardScaler`.

### Step 6 — Generate Prediction

The processed feature vector is passed to the TensorFlow model.

### Step 7 — Display Result

The predicted churn probability is converted into a binary churn classification and displayed through Streamlit.

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will start a local Streamlit server and provide a browser-based interface for entering customer information.

---

## Example Prediction Scenario

A potentially high-risk customer profile can be tested using characteristics such as:

```text
Geography          : Germany
Gender             : Female
Age                : 45
Credit Score       : 450
Tenure             : 2
Balance            : 120000
Number of Products : 1
Has Credit Card    : 1
Active Member      : 0
Estimated Salary   : 100000
```

The model then calculates a churn probability based on the patterns learned during training.

---

## Key Machine Learning Considerations

### Consistent Preprocessing

The preprocessing artifacts are persisted separately from the model to ensure inference uses the same transformations as training.

```text
Training                         Inference
────────                         ─────────
Raw Data                         User Input
   ↓                                ↓
Encoding                         Encoding
   ↓                                ↓
Scaling                          Scaling
   ↓                                ↓
Neural Network                  Neural Network
```

Changing the encoding or scaling strategy after training can result in incorrect predictions.

### Feature Ordering

The order of features is important when using a trained scaler and neural network.

The application therefore explicitly aligns the DataFrame with:

```python
scalar.feature_names_in_
```

before performing scaling.

---

## Future Improvements

Potential improvements to the application include:

* Displaying the exact churn probability
* Adding model confidence indicators
* Implementing SHAP/LIME-based explainability
* Adding feature importance visualization
* Introducing probability calibration
* Experimenting with alternative classification algorithms
* Hyperparameter optimization
* Model versioning using MLflow
* Containerizing the application with Docker
* Deploying the application to a cloud platform
* Adding automated model monitoring
* Implementing CI/CD for model and application deployment

---

## Limitations

* Prediction quality depends on the training dataset and model performance.
* A probability threshold of `0.5` may not be optimal for every business use case.
* The model should not be interpreted as a deterministic prediction of customer behavior.
* Model performance may degrade when applied to customer populations significantly different from the training data.
* Preprocessing artifacts must remain compatible with the trained model.

---

## Conclusion

This project demonstrates an end-to-end **machine learning inference system** for customer churn prediction.

Rather than directly passing raw user input to the neural network, the application reproduces the complete preprocessing pipeline used during training, including categorical encoding, feature alignment, and numerical standardization.

The resulting system combines **TensorFlow, Scikit-learn, Pandas, and Streamlit** to provide an interactive interface for real-time customer churn prediction.

---

## Author

**Bhavana**

Artificial Intelligence & Machine Learning Student

Technologies: Python · TensorFlow · Scikit-learn · Pandas · Streamlit · Machine Learning
