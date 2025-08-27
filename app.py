import streamlit as st
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

import pandas as pd
import joblib
import json
import plotly.express as px

# --------- Cached loaders ---------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/titanic.csv")

# Load with spinners (nice UX)
with st.spinner("Loading model..."):
    model = load_model()
with st.spinner("Loading data & metrics..."):
    df = load_data()
    metrics = load_metrics()

# --------- Sidebar ---------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Explore Data", "Visualizations", "Predict", "Model Performance"]
)

# --------- Overview ---------
if menu == "Overview":
    st.title("🚢 Titanic Survival Prediction App")
    st.write("""
    This app predicts passenger survival on the Titanic using a trained **Logistic Regression model**.

    **Features included:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, IsAlone
    """)
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head(), use_container_width=True)

# --------- Explore Data ---------
elif menu == "Explore Data":
    st.title("🔎 Explore Dataset")
    st.write(df.describe())
    col1, col2 = st.columns(2)
    with col1:
        sex_filter = st.multiselect("Select Sex", options=df["Sex"].unique(), default=list(df["Sex"].unique()))
    with col2:
        pclass_filter = st.multiselect("Select Pclass", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))
    filtered = df[(df["Sex"].isin(sex_filter)) & (df["Pclass"].isin(pclass_filter))]
    st.caption(f"Filtered rows: {len(filtered)}")
    st.dataframe(filtered.head(50), use_container_width=True)

# --------- Visualizations ---------
elif menu == "Visualizations":
    st.title("📊 Visualizations")
    fig1 = px.histogram(df, x="Age", color="Survived", nbins=30, title="Age distribution by Survival")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(df, x="Sex", color="Survived", title="Survival counts by Sex")
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.bar(df, x="Pclass", color="Survived", title="Survival counts by Passenger Class")
    st.plotly_chart(fig3, use_container_width=True)

# --------- Predict ---------
elif menu == "Predict":
    st.title("🧮 Predict Survival")
    st.write("Enter passenger details below:")

    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("SibSp (# of siblings/spouses)", 0, 10, 0)
    parch = st.number_input("Parch (# of parents/children)", 0, 10, 0)
    fare = st.slider("Fare", 0, 500, 50)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])

    family_size = int(sibsp + parch + 1)
    is_alone = 1 if family_size == 1 else 0
    st.caption(f"Computed: FamilySize={family_size}, IsAlone={is_alone}")

    if st.button("Predict Survival"):
        input_df = pd.DataFrame([{
            "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch,
            "Fare": fare, "Embarked": embarked, "Title": title,
            "FamilySize": family_size, "IsAlone": is_alone
        }])
        try:
            pred = model.predict(input_df)[0]
            proba = float(model.predict_proba(input_df)[0][1])
            if pred == 1:
                st.success(f"✅ Survived (Probability: {proba:.2f})")
            else:
                st.error(f"❌ Did Not Survive (Probability: {proba:.2f})")
        except Exception as e:
            st.warning(f"Prediction failed: {e}")

# --------- Model Performance ---------
elif menu == "Model Performance":
    st.title("📈 Model Performance")

    # Show saved metrics
    st.subheader("Test Metrics (from metrics.json)")
    st.json(metrics)

    # Confusion Matrix on the full dataset using the trained pipeline
    st.subheader("Confusion Matrix (on dataset)")

    # Recreate engineered features used in training:
    tmp = df.copy()
    tmp["Title"] = tmp["Name"].str.extract(r',\s*([^\.]+)\.')
    tmp["Title"] = tmp["Title"].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Other', 'Countess': 'Other',
        'Capt': 'Other', 'Col': 'Other', 'Don': 'Other', 'Dr': 'Other', 'Major': 'Other',
        'Rev': 'Other', 'Sir': 'Other', 'Jonkheer': 'Other', 'Dona': 'Other'
    })
    tmp["Title"] = tmp["Title"].where(tmp["Title"].isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Other')
    tmp["FamilySize"] = tmp["SibSp"] + tmp["Parch"] + 1
    tmp["IsAlone"] = (tmp["FamilySize"] == 1).astype(int)

    feats = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "IsAlone"]
    X_all = tmp[feats]
    y_all = tmp["Survived"]

    # Predict with pipeline (it includes preprocessing)
    y_hat = model.predict(X_all)

    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff
    cm = confusion_matrix(y_all, y_hat)
    fig = ff.create_annotated_heatmap(
        z=cm.astype(int),
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        showscale=False
    )
    fig.update_layout(width=500, height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=False)
