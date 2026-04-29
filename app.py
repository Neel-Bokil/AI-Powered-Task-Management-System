import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Task Assignment System", layout="wide")


# -----------------------------
# CUSTOM CSS (DARK THEME UI)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding-top: 1rem;
}
.main-title {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 0px 15px rgba(147, 51, 234, 0.4);
}
.sub-title {
    text-align: center;
    font-size: 15px;
    margin-top: -15px;
    color: #d1d5db;
    margin-bottom: 25px;
}
.card {
    background-color: #151a24;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
    height: 140px;
}
.card-label {
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 10px;
}
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 18px;
    font-size: 16px;
    font-weight: 700;
    color: white;
}
.badge-priority {
    background-color: #f59e0b;
}
.badge-type {
    background-color: #ef4444;
}
.assignee-box {
    background-color: #151a24;
    border-radius: 18px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
}
.assignee-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}
.assignee-name {
    font-size: 22px;
    font-weight: 800;
    color: #a78bfa;
}
.assignee-id {
    font-size: 14px;
    color: #9ca3af;
}
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}
.sidebar-user {
    font-size: 15px;
    font-weight: 700;
    color: #e5e7eb;
}
.sidebar-role {
    font-size: 13px;
    color: #9ca3af;
    margin-bottom: 5px;
}
.stProgress > div > div > div {
    border-radius: 20px;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 800;
    margin-top: 3px;
}
.status-available {
    background-color: #22c55e;
    color: black;
}
.status-busy {
    background-color: #ef4444;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# -----------------------------
# LOAD BERT MODEL (HF HUB)
# -----------------------------
@st.cache_resource
def load_bert_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("Neel-Bokil/jira-issue-type-bert")
    bert_model = AutoModelForSequenceClassification.from_pretrained("Neel-Bokil/jira-issue-type-bert")

    bert_model.to(device)
    bert_model.eval()

    issue_type_encoder = joblib.load("issue_type_label_encoder.pkl")

    return tokenizer, bert_model, issue_type_encoder, device


tokenizer, bert_model, issue_type_encoder, device = load_bert_model()


# -----------------------------
# LOAD PRIORITY MODEL
# -----------------------------
@st.cache_resource
def load_priority_model():
    priority_pipeline = joblib.load("priority_xgboost_pipeline_v2.pkl")
    priority_encoder = joblib.load("priority_label_encoder_v2.pkl")
    return priority_pipeline, priority_encoder


priority_pipeline, priority_encoder = load_priority_model()


# -----------------------------
# LOAD USERS DATASET (SESSION SAFE)
# -----------------------------
if "users_df" not in st.session_state:
    users_df = pd.read_csv("jira_users_with_workload.csv")
    users_df["active"] = users_df["active"].fillna(False)
    users_df["total_user_workload_seconds"] = users_df["total_user_workload_seconds"].fillna(0)

    st.session_state["users_df"] = users_df

users_df = st.session_state["users_df"]


# -----------------------------
# STORE ALREADY ASSIGNED TASKS (DUPLICATE DETECTION)
# -----------------------------
if "assigned_tasks" not in st.session_state:
    st.session_state["assigned_tasks"] = set()


# -----------------------------
# ISSUE TYPE PREDICTION FUNCTION
# -----------------------------
def predict_issue_type_bert(text):
    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    pred_class = torch.argmax(outputs.logits, dim=1).item()
    return issue_type_encoder.inverse_transform([pred_class])[0]


# -----------------------------
# PRIORITY PREDICTION FUNCTION
# -----------------------------
def predict_priority(new_task_text, predicted_issue_type):
    summary_len = len(new_task_text)
    desc_len = len(new_task_text)

    sample = pd.DataFrame([{
        "summary_len": summary_len,
        "desc_len": desc_len,
        "commentcount": 0,
        "attachmentcount": 0,
        "votes": 0,
        "watch_count": 0,
        "age_days": 0,
        "bert_predicted_issue_type": predicted_issue_type,
        "clean_text": clean_text(new_task_text)                #newly added column for v2 models
    }])

    pred_encoded = priority_pipeline.predict(sample)[0]
    pred_label = priority_encoder.inverse_transform([pred_encoded])[0]

    return pred_label


# -----------------------------
# WORKLOAD ASSIGNMENT FUNCTION
# -----------------------------
def suggest_assignee_and_update(users_df):
    active_users = users_df[users_df["active"] == True]

    if active_users.empty:
        users_df["active"] = True
        active_users = users_df.copy()

    least_loaded_user = active_users.sort_values("total_user_workload_seconds").iloc[0]

    selected_user = least_loaded_user["user"]
    selected_display = least_loaded_user["display_name"]

    # mark user inactive after assignment
    users_df.loc[users_df["user"] == selected_user, "active"] = False

    return selected_user, selected_display


# -----------------------------
# UI HEADER
# -----------------------------
st.markdown('<div class="main-title">🤖 AI-Powered Task Assignment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicts task priority & type, and assigns the optimal team member</div>', unsafe_allow_html=True)


# -----------------------------
# MAIN LAYOUT (SIDEBAR + MAIN)
# -----------------------------
left_col, right_col = st.columns([1, 3])


# -----------------------------
# LEFT PANEL: TEAM DASHBOARD
# -----------------------------
with left_col:
    st.markdown("<div class='sidebar-title'>Team Dashboard</div>", unsafe_allow_html=True)


    # show top 10 least loaded users
    top_users = (
        users_df[~users_df["user"].str.contains("bot", case=False, na=False)]
        .sort_values("total_user_workload_seconds")
        .head(10)
        .sort_values("display_name")
    )

    # Workload normalization for progress bar
    max_load = top_users["total_user_workload_seconds"].max()
    if max_load == 0:
        max_load = 1

    for _, row in top_users.iterrows():
        name = row["display_name"]
        user_id = row["user"]
        active_status = row["active"]

        workload = row["total_user_workload_seconds"]
        percent = int((workload / max_load) * 100)

        status_text = "Available" if active_status else "Busy"
        status_class = "status-available" if active_status else "status-busy"

        st.markdown(f"<div class='sidebar-user'>{name}</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='sidebar-role'>
            {user_id}
            <span class='status-badge {status_class}'>{status_text}</span>
        </div>
        """, unsafe_allow_html=True)

        st.progress(percent / 100)


# -----------------------------
# RIGHT PANEL: TASK INPUT + RESULTS
# -----------------------------
with right_col:
    st.subheader("📄 Describe Your Task")

    task_text = st.text_area("", height=150, placeholder="Enter task description here...")

    if st.button("🚀 Analyze & Assign"):

        if task_text.strip() == "":
            st.warning("⚠️ Please enter a task description.")

        else:
            # -----------------------------
            # DUPLICATE TASK CHECK
            # -----------------------------
            task_cleaned = clean_text(task_text)

            if task_cleaned in st.session_state["assigned_tasks"]:
                st.warning("⚠️ This task has already been assigned earlier.")
                st.stop()
            else:
                st.session_state["assigned_tasks"].add(task_cleaned)

            # -----------------------------
            # PREDICTIONS
            # -----------------------------
            predicted_issue_type = predict_issue_type_bert(task_text)
            predicted_priority = predict_priority(task_text, predicted_issue_type)

            # -----------------------------
            # ASSIGN USER
            # -----------------------------
            user_id, display_name = suggest_assignee_and_update(users_df)

            # Save updated dataframe back
            st.session_state["users_df"] = users_df

            st.success("✅ Prediction Completed!")

            # Output cards row
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">Priority</div>
                    <span class="badge badge-priority">{predicted_priority}</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">Task Type</div>
                    <span class="badge badge-type">{predicted_issue_type}</span>
                </div>
                """, unsafe_allow_html=True)

            # Assigned member section
            st.markdown(f"""
            <div class="assignee-box">
                <div class="assignee-title">Assigned Member</div>
                <div class="assignee-name">{display_name}</div>
                <div class="assignee-id">UserID: {user_id}</div>
            </div>
            """, unsafe_allow_html=True)

            # Show updated availability table (without email column)
            # st.subheader("👥 Updated Availability Table (Top 10)")
            # st.dataframe(
            #     users_df.drop(columns=["email"], errors="ignore")
            #             .sort_values("total_user_workload_seconds")
            #             .head(10),
            #     use_container_width=True
            # )