"""
=============================================================
Teen Mental Health & Social Media  ─  Streamlit Dashboard
=============================================================
Sections
  1. 🏠 Overview       – dataset summary statistics
  2. 📊 EDA            – interactive charts
  3. 🤖 AI Models      – train 4 classifiers and compare them
  4. 🔮 Prediction     – predict depression risk for a new student
  5. 📋 About          – project info
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

# ── optional imbalanced-learn ──────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teen Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8eaf6;
}

/* Main content area */
section.main > div {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(106,90,205,0.3);
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(106,90,205,0.7) 0%, rgba(72,52,212,0.7) 100%);
    border: 1px solid rgba(106,90,205,0.5);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.hero-banner h1 { font-size: 2.8rem; font-weight: 700; margin: 0; color: #fff; }
.hero-banner p  { font-size: 1.15rem; color: #c5cae9; margin-top: 0.6rem; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(106,90,205,0.4);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(8px);
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(106,90,205,0.4);
}
.metric-card .value { font-size: 2.2rem; font-weight: 700; color: #7c83fd; }
.metric-card .label { font-size: 0.85rem; color: #9fa8da; margin-top: 0.3rem; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, rgba(106,90,205,0.5), rgba(72,52,212,0));
    border-left: 4px solid #7c83fd;
    padding: 0.8rem 1.2rem;
    border-radius: 0 12px 12px 0;
    margin: 2rem 0 1.2rem 0;
    color: #e8eaf6;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Model result cards */
.model-card {
    background: rgba(20,20,50,0.7);
    border: 1px solid rgba(124,131,253,0.35);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
    backdrop-filter: blur(6px);
}
.model-card:hover { border-color: #7c83fd; }
.model-card .model-name { font-size: 1.05rem; font-weight: 600; color: #b39ddb; }
.model-card .metrics { display: flex; gap: 2rem; margin-top: 0.6rem; flex-wrap: wrap; }
.model-card .metric  { font-size: 0.88rem; color: #c5cae9; }
.model-card .metric span { font-weight: 700; color: #7c83fd; }

/* Prediction result */
.pred-safe {
    background: linear-gradient(135deg, rgba(56,142,60,0.3), rgba(46,125,50,0.2));
    border: 1px solid #66bb6a;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: #a5d6a7;
}
.pred-risk {
    background: linear-gradient(135deg, rgba(198,40,40,0.3), rgba(183,28,28,0.2));
    border: 1px solid #ef5350;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ef9a9a;
}

/* Streamlit widget overrides */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input {
    background: rgba(30,30,70,0.8) !important;
    border-color: rgba(106,90,205,0.5) !important;
    color: #e8eaf6 !important;
}
.stSlider > div > div > div { color: #7c83fd !important; }
.stButton > button {
    background: linear-gradient(135deg, #6a5acd, #4834d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(20,20,50,0.6);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #9fa8da !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(106,90,205,0.4) !important;
    color: #fff !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helper: load & preprocess
# ─────────────────────────────────────────────────────────────
DATA_PATH = "Teen_Mental_Health_Dataset.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df['age'] = df['age'].astype(int)
    df['academic_performance'] = df['academic_performance'].astype(float)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df[(df['age'] >= 13) & (df['age'] <= 19)]
    df['gender'] = df['gender'].str.strip().str.lower()
    df['platform_usage'] = df['platform_usage'].str.strip().str.capitalize()
    return df

@st.cache_resource(show_spinner=False)
def prepare_and_train(use_smote: bool = True):
    """Full preprocessing + model training. Returns models, metrics, test sets."""
    df = load_data()

    le = LabelEncoder()
    df['depression_label_encoded'] = le.fit_transform(df['depression_label'])

    cat_features = [c for c in ['gender', 'platform_usage'] if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_features, drop_first=True)

    drop_cols = [c for c in [
        'depression_label', 'depression_label_encoded',
        'daily_social_media_hours_is_outlier',
        'academic_performance_is_outlier',
    ] if c in df_enc.columns]

    X = df_enc.drop(columns=drop_cols)
    y = df_enc['depression_label_encoded']
    X = X.select_dtypes(include=[np.number])

    scaler_pre = StandardScaler()
    X_scaled = pd.DataFrame(scaler_pre.fit_transform(X), columns=X.columns)

    def cap_iqr(df_, col):
        Q1, Q3 = df_[col].quantile(0.25), df_[col].quantile(0.75)
        IQR = Q3 - Q1
        df_[col] = df_[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df_

    for col in X_scaled.columns:
        X_scaled = cap_iqr(X_scaled, col)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )

    if use_smote and SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42, C=0.5, gamma="scale"))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))
        ]),
    }

    results = []
    trained = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba[:, 1])
        cm  = confusion_matrix(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "F1-Score": f1, "ROC-AUC": auc, "CM": cm})
        trained[name] = pipe

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    feature_cols = list(X.columns)
    return trained, results_df, X_test, y_test, feature_cols, scaler_pre, X.columns.tolist(), le

# ─────────────────────────────────────────────────────────────
# Sidebar / Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0.5rem 0.5rem;'>
        <div style='font-size:3rem;'>🧠</div>
        <div style='font-size:1.1rem; font-weight:700; color:#b39ddb; margin-top:0.3rem;'>
            Teen Mental Health
        </div>
        <div style='font-size:0.8rem; color:#7986cb; margin-bottom:1rem;'>
            Social Media & Depression Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA", "🤖 AI Models", "🔮 Prediction", "📋 About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#7986cb; padding: 0 0.5rem;'>
    <b style='color:#9fa8da;'>Dataset</b><br>
    • 1,200 teen records<br>
    • Ages 13–19<br>
    • 13 features<br>
    • Binary depression label<br><br>
    <b style='color:#9fa8da;'>Models</b><br>
    • Logistic Regression<br>
    • Random Forest<br>
    • SVM (RBF)<br>
    • Gradient Boosting
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load data (always)
# ─────────────────────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"❌ Dataset not found at `{DATA_PATH}`. Make sure `Teen_Mental_Health_Dataset.csv` is in the same folder as this app.")
    st.stop()

# ═════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero-banner'>
        <h1>🧠 Teen Mental Health Dashboard</h1>
        <p>Exploring the impact of social media on adolescent mental health · Ages 13–19</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, df.shape[0], "Total Students"),
        (c2, df['age'].nunique(), "Age Groups"),
        (c3, f"{df['daily_social_media_hours'].mean():.1f}h", "Avg Social Media / Day"),
        (c4, f"{df['sleep_hours'].mean():.1f}h", "Avg Sleep / Night"),
        (c5, f"{(df['depression_label']==1).sum()}", "Depression Cases"),
    ]
    for col, val, lbl in kpi_data:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>{val}</div>
                <div class='label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns: distribution + correlation
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-header'>📈 Feature Distributions</div>", unsafe_allow_html=True)
        num_feat = [
            "daily_social_media_hours","sleep_hours","stress_level",
            "anxiety_level","addiction_level","academic_performance",
            "physical_activity","screen_time_before_sleep"
        ]
        feat_choice = st.selectbox("Select feature", num_feat, key="ov_feat")
        fig = px.histogram(
            df, x=feat_choice, color="depression_label",
            color_discrete_map={0: "#5c6bc0", 1: "#ef5350"},
            barmode="overlay", opacity=0.75,
            labels={"depression_label": "Depression", feat_choice: feat_choice.replace("_"," ").title()},
            template="plotly_dark",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(title="Depression", x=0.78, y=0.98),
            height=340, margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
        num_cols_corr = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols_corr].corr()
        fig2 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, template="plotly_dark",
            aspect="auto",
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=340, margin=dict(l=0,r=0,t=20,b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Depression by gender / platform
    st.markdown("<div class='section-header'>📊 Depression Across Groups</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        fig_g = px.histogram(
            df, x="gender", color="depression_label",
            barmode="group", text_auto=True,
            color_discrete_map={0:"#5c6bc0",1:"#ef5350"},
            labels={"depression_label":"Depression","gender":"Gender"},
            template="plotly_dark",
        )
        fig_g.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=30,b=0),
            title="By Gender",
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with g2:
        fig_p = px.histogram(
            df, x="platform_usage", color="depression_label",
            barmode="group", text_auto=True,
            color_discrete_map={0:"#5c6bc0",1:"#ef5350"},
            labels={"depression_label":"Depression","platform_usage":"Platform"},
            template="plotly_dark",
        )
        fig_p.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=30,b=0),
            title="By Platform",
        )
        st.plotly_chart(fig_p, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: EDA
# ═════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Box Plots", "🌀 Scatter", "🎯 Age Analysis", "📋 Raw Stats"])

    with tab1:
        feat_box = st.selectbox("Feature for box plot", [
            "daily_social_media_hours","sleep_hours","stress_level",
            "anxiety_level","addiction_level","academic_performance",
            "physical_activity","screen_time_before_sleep"
        ], key="box_feat")
        fig = px.box(
            df, x="depression_label", y=feat_box,
            color="depression_label",
            color_discrete_map={0:"#5c6bc0",1:"#ef5350"},
            points="outliers",
            labels={"depression_label":"Depression"},
            template="plotly_dark",
            title=f"{feat_box.replace('_',' ').title()} by Depression Label",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_feats = [
            "daily_social_media_hours","sleep_hours","stress_level",
            "anxiety_level","addiction_level","academic_performance",
            "physical_activity","screen_time_before_sleep","age"
        ]
        sc1, sc2 = st.columns(2)
        with sc1:
            x_feat = st.selectbox("X axis", num_feats, index=0, key="sc_x")
        with sc2:
            y_feat = st.selectbox("Y axis", num_feats, index=1, key="sc_y")

        fig_s = px.scatter(
            df, x=x_feat, y=y_feat,
            color=df["depression_label"].map({0:"No Depression",1:"Depression"}),
            color_discrete_map={"No Depression":"#5c6bc0","Depression":"#ef5350"},
            opacity=0.6, template="plotly_dark",
            labels={x_feat: x_feat.replace("_"," ").title(), y_feat: y_feat.replace("_"," ").title()},
            title=f"{x_feat.replace('_',' ').title()} vs {y_feat.replace('_',' ').title()}",
        )
        fig_s.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with tab3:
        agg = df.groupby("age").agg(
            Depression_Rate=("depression_label","mean"),
            Avg_Social_Media=("daily_social_media_hours","mean"),
            Avg_Sleep=("sleep_hours","mean"),
            Avg_Stress=("stress_level","mean"),
        ).reset_index()

        fig_a1 = px.bar(
            agg, x="age", y="Depression_Rate",
            template="plotly_dark",
            color="Depression_Rate",
            color_continuous_scale="Reds",
            title="Depression Rate by Age",
        )
        fig_a1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_a1, use_container_width=True)

        fig_a2 = go.Figure()
        for col_, name_, color_ in [
            ("Avg_Social_Media","Avg Social Media Hours","#7c83fd"),
            ("Avg_Sleep","Avg Sleep Hours","#66bb6a"),
            ("Avg_Stress","Avg Stress Level","#ef5350"),
        ]:
            fig_a2.add_trace(go.Scatter(
                x=agg["age"], y=agg[col_], mode="lines+markers",
                name=name_, line=dict(color=color_, width=2),
                marker=dict(size=7),
            ))
        fig_a2.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title="Key Metrics by Age",
            xaxis_title="Age", yaxis_title="Value",
        )
        st.plotly_chart(fig_a2, use_container_width=True)

    with tab4:
        st.subheader("Statistical Summary")
        st.dataframe(
            df.describe().T.style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )
        st.subheader("Class Balance")
        bal = df['depression_label'].value_counts(normalize=True).mul(100).round(2)
        st.dataframe(bal.rename("Percentage (%)").reset_index(), use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: AI Models
# ═════════════════════════════════════════════════════════════
elif page == "🤖 AI Models":
    st.markdown("<div class='section-header'>🤖 AI Model Training & Evaluation</div>", unsafe_allow_html=True)

    use_smote = st.checkbox(
        "Apply SMOTE oversampling (handles class imbalance)",
        value=True,
        disabled=(not SMOTE_AVAILABLE),
        help="SMOTE requires the `imbalanced-learn` package." if not SMOTE_AVAILABLE else "",
    )

    with st.spinner("Training 4 classifiers… this may take a moment ⚙️"):
        trained, results_df, X_test, y_test, feature_cols, scaler_pre, orig_cols, le = \
            prepare_and_train(use_smote=use_smote and SMOTE_AVAILABLE)

    st.success("✅ All models trained successfully!")

    # ── Model comparison bar chart ──
    st.markdown("<div class='section-header'>📊 Model Comparison</div>", unsafe_allow_html=True)
    melted = results_df.melt(
        id_vars="Model", value_vars=["Accuracy","F1-Score","ROC-AUC"],
        var_name="Metric", value_name="Score",
    )
    fig_cmp = px.bar(
        melted, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_map={"Accuracy":"#5c6bc0","F1-Score":"#ff7043","ROC-AUC":"#26a69a"},
        template="plotly_dark", title="Accuracy / F1-Score / ROC-AUC per Model",
    )
    fig_cmp.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,1.1]),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Model cards ──
    st.markdown("<div class='section-header'>🏆 Individual Results</div>", unsafe_allow_html=True)
    for _, row in results_df.iterrows():
        medal = "🥇" if _ == 0 else ("🥈" if _ == 1 else ("🥉" if _ == 2 else ""))
        st.markdown(f"""
        <div class='model-card'>
            <div class='model-name'>{medal} {row['Model']}</div>
            <div class='metrics'>
                <div class='metric'>Accuracy <span>{row['Accuracy']:.4f}</span></div>
                <div class='metric'>F1-Score <span>{row['F1-Score']:.4f}</span></div>
                <div class='metric'>ROC-AUC <span>{row['ROC-AUC']:.4f}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confusion matrices ──
    st.markdown("<div class='section-header'>🧩 Confusion Matrices</div>", unsafe_allow_html=True)
    model_names = results_df["Model"].tolist()
    cols_cm = st.columns(2)
    for idx, name in enumerate(model_names):
        pipe = trained[name]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale="Blues",
            template="plotly_dark",
            title=name,
        )
        fig_cm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=40,b=0),
            coloraxis_showscale=False,
        )
        with cols_cm[idx % 2]:
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC curves ──
    st.markdown("<div class='section-header'>📈 ROC Curves</div>", unsafe_allow_html=True)
    from sklearn.metrics import roc_curve
    colors_roc = ["#7c83fd","#ef5350","#66bb6a","#ffa726"]
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="gray", dash="dash"))
    for i, name in enumerate(model_names):
        pipe = trained[name]
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = results_df.loc[results_df["Model"]==name,"ROC-AUC"].values[0]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=colors_roc[i], width=2.5),
        ))
    fig_roc.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        title="ROC Curves — All Models",
        legend=dict(x=0.55, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: Prediction
# ═════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("<div class='section-header'>🔮 Depression Risk Predictor</div>", unsafe_allow_html=True)

    st.info("Fill in the student's details below, then click **Predict** to estimate depression risk.")

    with st.spinner("Loading models…"):
        trained, results_df, X_test, y_test, feature_cols, scaler_pre, orig_cols, le = \
            prepare_and_train(use_smote=SMOTE_AVAILABLE)

    # ── Input form ──
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", 13, 19, 16)
            gender = st.selectbox("Gender", ["female", "male"])
            social_media_hours = st.slider("Daily Social Media Hours", 1.0, 8.0, 4.5, 0.1)

        with col2:
            st.markdown("**😴 Lifestyle**")
            sleep_hours = st.slider("Sleep Hours", 4.0, 9.0, 6.5, 0.1)
            screen_time = st.slider("Screen Time Before Sleep (hrs)", 0.5, 3.0, 1.5, 0.1)
            physical_activity = st.slider("Physical Activity (hrs/day)", 0.0, 2.0, 1.0, 0.1)

        with col3:
            st.markdown("**🏫 Academic & Social**")
            academic_perf = st.slider("Academic Performance (GPA 2.0–4.0)", 2.0, 4.0, 3.0, 0.01)
            social_interaction = st.selectbox("Social Interaction Level", ["low", "medium", "high"])
            platform = st.selectbox("Platform Usage", ["Instagram", "Tiktok", "Both"])

        st.markdown("**😰 Mental Wellness Scales (1–10)**")
        ms1, ms2, ms3 = st.columns(3)
        with ms1:
            stress = st.slider("Stress Level", 1, 10, 5)
        with ms2:
            anxiety = st.slider("Anxiety Level", 1, 10, 5)
        with ms3:
            addiction = st.slider("Addiction Level", 1, 10, 5)

        chosen_model = st.selectbox(
            "Model to use for prediction",
            results_df["Model"].tolist(),
        )

        submitted = st.form_submit_button("🔮 Predict Depression Risk", use_container_width=True)

    if submitted:
        # Build raw input row that mirrors the original df columns
        gen_female = 1 if gender == "female" else 0
        plat_instagram = 1 if platform == "Instagram" else 0
        plat_tiktok    = 1 if platform == "Tiktok"    else 0

        # social_interaction_level encoding (one-hot, but we used get_dummies in preprocessing)
        soc_low    = 1 if social_interaction == "low"    else 0
        soc_medium = 1 if social_interaction == "medium" else 0

        # Reconstruct a feature vector by replaying preprocessing
        # We need to match exactly orig_cols (X.columns from training)
        raw = {
            "age": age,
            "daily_social_media_hours": social_media_hours,
            "sleep_hours": sleep_hours,
            "screen_time_before_sleep": screen_time,
            "academic_performance": academic_perf,
            "physical_activity": physical_activity,
            "social_interaction_level": social_interaction,
            "stress_level": stress,
            "anxiety_level": anxiety,
            "addiction_level": addiction,
        }
        raw_df = pd.DataFrame([raw])
        # Re-apply get_dummies to match training schema
        raw_df["gender"] = gender
        raw_df["platform_usage"] = platform
        raw_encoded = pd.get_dummies(raw_df, columns=["gender","platform_usage","social_interaction_level"], drop_first=False)

        # Align columns with training data (add missing cols as 0)
        for c in orig_cols:
            if c not in raw_encoded.columns:
                raw_encoded[c] = 0
        raw_encoded = raw_encoded[orig_cols]

        pipe = trained[chosen_model]
        prediction = pipe.predict(raw_encoded)[0]
        prob = pipe.predict_proba(raw_encoded)[0]
        risk_prob = prob[1]

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class='pred-risk'>
                ⚠️ High Depression Risk Detected<br>
                <span style='font-size:1rem; font-weight:400;'>Model confidence: {risk_prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-safe'>
                ✅ Low Depression Risk<br>
                <span style='font-size:1rem; font-weight:400;'>Model confidence: {(1-risk_prob)*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob * 100,
            title={"text": "Depression Risk %", "font": {"color": "#e8eaf6"}},
            number={"suffix": "%", "font": {"color": "#e8eaf6", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9fa8da"},
                "bar": {"color": "#ef5350" if risk_prob > 0.5 else "#66bb6a"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(102,187,106,0.2)"},
                    {"range": [30, 60], "color": "rgba(255,167,38,0.2)"},
                    {"range": [60, 100], "color": "rgba(239,83,80,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=40,b=0),
            font=dict(color="#e8eaf6"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance (if RF or GB)
        if chosen_model in ("Random Forest", "Gradient Boosting"):
            clf = trained[chosen_model].named_steps["clf"]
            importances = clf.feature_importances_
            fi_df = pd.DataFrame({"Feature": orig_cols, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                template="plotly_dark", title=f"Feature Importances ({chosen_model})",
                color="Importance", color_continuous_scale="Purples",
            )
            fig_fi.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE: About
# ═════════════════════════════════════════════════════════════
elif page == "📋 About":
    st.markdown("""
    <div class='hero-banner'>
        <h1>📋 About This Project</h1>
        <p>Teen Mental Health & Social Media Impact Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='model-card'>
            <div class='model-name'>🎯 Project Goal</div>
            <p style='color:#c5cae9; margin-top:0.8rem;'>
            Investigate the relationship between adolescent social media usage patterns and mental health 
            outcomes (specifically depression), and build machine learning models that can predict 
            depression risk from behavioral and lifestyle features.
            </p>
        </div>

        <div class='model-card'>
            <div class='model-name'>📂 Dataset</div>
            <div class='metrics' style='flex-direction:column; gap:0.4rem; margin-top:0.6rem;'>
                <div class='metric'>Records: <span>1,200 teen students</span></div>
                <div class='metric'>Age range: <span>13–19 years</span></div>
                <div class='metric'>Features: <span>13 columns</span></div>
                <div class='metric'>Target: <span>Depression label (binary)</span></div>
                <div class='metric'>Class balance: <span>~97.4% no depression, ~2.6% depression</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='model-card'>
            <div class='model-name'>🔬 Key Features</div>
            <div style='color:#c5cae9; margin-top:0.8rem; line-height:1.8;'>
            • <b>Daily Social Media Hours</b> – time spent on social platforms<br>
            • <b>Platform Usage</b> – Instagram / TikTok / Both<br>
            • <b>Sleep Hours</b> – nightly sleep duration<br>
            • <b>Screen Time Before Sleep</b> – late-night device use<br>
            • <b>Academic Performance</b> – GPA equivalent<br>
            • <b>Physical Activity</b> – daily exercise in hours<br>
            • <b>Social Interaction Level</b> – low / medium / high<br>
            • <b>Stress / Anxiety / Addiction Level</b> – self-reported scales (1–10)
            </div>
        </div>

        <div class='model-card'>
            <div class='model-name'>🤖 ML Pipeline</div>
            <div style='color:#c5cae9; margin-top:0.8rem; line-height:1.8;'>
            1. Data cleaning & type conversion<br>
            2. Categorical encoding (Label + One-Hot)<br>
            3. Feature scaling (StandardScaler)<br>
            4. Outlier capping (IQR method)<br>
            5. SMOTE oversampling (class imbalance)<br>
            6. 80/20 stratified train-test split<br>
            7. Train 4 classifiers (LR, RF, SVM, GB)<br>
            8. Evaluate: Accuracy, F1, ROC-AUC
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='model-card' style='margin-top:1rem;'>
        <div class='model-name'>⚙️ Technology Stack</div>
        <div class='metrics' style='margin-top:0.6rem;'>
            <div class='metric'>Frontend: <span>Streamlit</span></div>
            <div class='metric'>ML: <span>Scikit-learn</span></div>
            <div class='metric'>Imbalance: <span>imbalanced-learn / SMOTE</span></div>
            <div class='metric'>Visualization: <span>Plotly Express</span></div>
            <div class='metric'>Data: <span>Pandas / NumPy</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
