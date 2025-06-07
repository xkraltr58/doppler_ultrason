import shap
import pandas as pd
import matplotlib.pyplot as plt
import pickle


model = pickle.load(open("rf_advanced_model.pkl", "rb"))
data = pd.read_csv("selected_features_advanced_cleaned.csv")
X = data.drop(["Label", "Filename"], axis=1)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("shap_bar_rf.png")
