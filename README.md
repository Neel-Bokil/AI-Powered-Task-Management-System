# AI Powered Task Management System

This project is an AI-based Task Management System built using Apache JIRA issue data.  
It predicts the **Issue/Task Type** and **Priority** of a given task description and automatically suggests the best assignee using workload balancing.

---

## 🚀 Features

- **Issue Type Prediction** using fine-tuned **BERT**
- **Priority Prediction** using **XGBoost + TF-IDF + Class Weighting**
- **Workload Balancing** using real user workload summary data
- **Automatic Task Assignment** to the least loaded available employee
- **Duplicate Task Detection** (prevents assigning same task multiple times)
- Interactive **Streamlit Web Interface**

---

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Transformers (HuggingFace)
- PyTorch
- Joblib

---

## 📂 Project Structure

```bash
AI Powered Task Management System/
│── app.py
│── jira_users_with_workload.csv
│── priority_xgboost_pipeline_v2.pkl
│── priority_label_encoder_v2.pkl
│── issue_type_label_encoder.pkl
│── requirements.txt
│── README.md
│── notebooks/
│ ├── Task_management_with_BERT.ipynb
│ ├── Priority_Model_XGBoost_TFIDF.ipynb
```

---

## 📊 Models Used

### 1. Issue Type Classification
- Model: Fine-tuned **BERT**
- Output Classes: Bug, Task, Improvement, New Feature, Story, Sub-task

### 2. Priority Prediction
- Model: **XGBoost**
- Features Used:
  - TF-IDF text features
  - Numeric engineered features (summary length, votes, comments, etc.)
  - BERT predicted issue type
  - Soft class weights for handling class imbalance
- Output Classes: Blocker, Critical, Major, Minor, Trivial

### 3. Workload Balancing
- Uses user workload dataset (`jira_users_with_workload.csv`)
- Assigns tasks to active employees with the least workload
- Updates availability dynamically during runtime

---

## 🖥️ Running the Streamlit App Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/Neel-Bokil/AI-Powered-Task-Management-System.git
cd AI-Powered-Task-Management-System
```
### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```
### Step 3: Run the App
```bash
streamlit run app.py
```

---

## 🤗 HuggingFace Model

The fine-tuned BERT model is hosted on HuggingFace Hub and is loaded directly in the Streamlit app.

---

## 📌 Example Output

Input Task:

- "Security vulnerability found in login module."

Predicted Output:

- Issue Type: Bug
- Priority: Critical
- Suggested Assignee: Least loaded active employee

---

## 🔮 Future Improvements

- Add Jira API integration for real-time issue creation and assignment
- Improve priority prediction using advanced embeddings (Sentence Transformers)
- Add dashboard analytics for workload monitoring

---

## 👨‍💻 Author

Neel Bokil

---
