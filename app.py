
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Personal Loan Propensity • Universal Bank", layout="wide")

# ---------------- Utility Functions ----------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace('.', '').replace(' ', '_') for c in df.columns]
    return df

def prepare_xy(df: pd.DataFrame):
    drop_cols = [c for c in df.columns if c.lower() in ('id','zip','zipcode','zip_code','zip__code')]
    if 'Personal_Loan' not in df.columns and 'Personal Loan' in df.columns:
        df = df.rename(columns={'Personal Loan':'Personal_Loan'})
    X = df.drop(columns=drop_cols + ['Personal_Loan'])
    y = df['Personal_Loan']
    return X, y, drop_cols

def train_models(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=y)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Gradient Boosted": GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    }
    metrics, rocs, cms, importances = [], {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_tr_pred, y_te_pred = mdl.predict(X_train), mdl.predict(X_test)
        y_tr_proba, y_te_proba = mdl.predict_proba(X_train)[:,1], mdl.predict_proba(X_test)[:,1]
        row = {
            "Algorithm": name,
            "Train Accuracy": accuracy_score(y_train, y_tr_pred),
            "Test Accuracy": accuracy_score(y_test, y_te_pred),
            "Precision": precision_score(y_test, y_te_pred, zero_division=0),
            "Recall": recall_score(y_test, y_te_pred, zero_division=0),
            "F1": f1_score(y_test, y_te_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_te_proba)
        }
        metrics.append(row)
        fpr, tpr, _ = roc_curve(y_test, y_te_proba)
        rocs[name] = (fpr, tpr, row["AUC"])
        cms[name] = {"train": confusion_matrix(y_train, y_tr_pred), "test": confusion_matrix(y_test, y_te_pred)}
        importances[name] = mdl.feature_importances_ if hasattr(mdl, "feature_importances_") else np.zeros(X.shape[1])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_add = {}
    for name, base in models.items():
        cv_acc = cross_val_score(base, X, y, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(base, X, y, cv=cv, scoring='roc_auc')
        cv_add[name] = (cv_acc.mean(), cv_acc.std(), cv_auc.mean(), cv_auc.std())

    metrics_df = pd.DataFrame(metrics).set_index("Algorithm").round(4)
    metrics_df["CV5_Acc_Mean"] = [cv_add[n][0] for n in metrics_df.index]
    metrics_df["CV5_Acc_SD"] = [cv_add[n][1] for n in metrics_df.index]
    metrics_df["CV5_AUC_Mean"] = [cv_add[n][2] for n in metrics_df.index]
    metrics_df["CV5_AUC_SD"] = [cv_add[n][3] for n in metrics_df.index]
    return models, metrics_df, rocs, cms, importances, (X_train, X_test, y_train, y_test)

def render_confusion_matrix(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Loan","Loan"]); ax.set_yticklabels(["No Loan","Loan"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center')
    st.pyplot(fig, use_container_width=True)

# ---------------- Streamlit App ----------------
st.sidebar.title("Universal Bank — Loan Propensity")
st.sidebar.write("Upload a dataset or use the included sample to explore, model and predict.")

uploaded = st.sidebar.file_uploader("Upload CSV (same schema as sample)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("UniversalBank.csv")
df = clean_columns(df)

st.title("Personal Loan Propensity Dashboard")
tab1, tab2, tab3 = st.tabs(["Customer Insights", "Model Training", "Predict New Data"])

with tab1:
    st.subheader("Customer Insights Dashboard")
    df['IncomeDecile'] = pd.qcut(df['Income'], 10, labels=False, duplicates='drop')
    rate = df.groupby('IncomeDecile')['Personal_Loan'].mean()
    fig, ax = plt.subplots()
    ax.bar(rate.index.astype(str), rate.values)
    ax.set_title("Loan Acceptance Rate by Income Decile")
    st.pyplot(fig)

with tab2:
    st.subheader("Train and Evaluate Models")
    if st.button("Run Models"):
        X, y, _ = prepare_xy(df)
        models, metrics_df, rocs, cms, importances, _ = train_models(X, y)
        st.dataframe(metrics_df)

        st.write("### Combined ROC Curves")
        fig, ax = plt.subplots()
        colors = {"Decision Tree":"blue","Random Forest":"green","Gradient Boosted":"red"}
        for name, (fpr,tpr,auc_val) in rocs.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=colors[name])  # <-- Fixed line
        ax.plot([0,1],[0,1],'--')
        ax.legend(); st.pyplot(fig)

with tab3:
    st.subheader("Upload New Data for Prediction")
    new_file = st.file_uploader("Upload new CSV", type=["csv"], key="scorecsv")
    if new_file:
        new_df = pd.read_csv(new_file)
        new_df = clean_columns(new_df)
        st.write(new_df.head())
