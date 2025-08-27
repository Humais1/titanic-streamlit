# Titanic Survival Prediction (Streamlit + scikit-learn)

**Live app:** https://<your-live-subdomain>.streamlit.app  
**Repo:** https://github.com/Humais1/titanic-streamlit

This app predicts passenger survival on the Titanic using a Logistic Regression pipeline (with preprocessing + engineered features).

## Features
- Pages: Overview, Explore Data, Visualizations, Predict, Model Performance (metrics + confusion matrix)
- Inputs: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title
- Engineered: FamilySize, IsAlone

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py

**Live app:** https://<your-live-subdomain>.streamlit.app  
**Repo:** https://github.com/Humais1/titanic-streamlit
