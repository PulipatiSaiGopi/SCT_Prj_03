# SCT_Prj_03
SCT_DS_03-Task3

Here's a comprehensive overview of the datasets:


---

📌 Objective

To predict whether a client will subscribe to a term deposit (y column: "yes"/"no") based on socio-demographic and campaign-related attributes using data mining and machine learning techniques.


---

🧠 Skills Applied

Data Preprocessing: Encoding categorical variables, handling class imbalance, scaling.

Exploratory Data Analysis (EDA): Descriptive statistics, correlation analysis, visualizations.

Modeling: Logistic Regression, Decision Trees, Random Forest, SVM, Gradient Boosting, etc.

Model Evaluation: Accuracy, Precision, Recall, F1-score, ROC AUC.

Feature Engineering: One-hot encoding, interaction terms.

Pipeline Creation: Scikit-learn Pipelines for reproducibility.

Cross-validation: Stratified K-Fold.

Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV.



---

🧰 Tools & Libraries

Language: Python 3.x

Libraries:

pandas, numpy – Data manipulation

matplotlib, seaborn – Visualization

scikit-learn – Machine learning models & preprocessing

xgboost, lightgbm (optional) – Advanced modeling

joblib or pickle – Model persistence

jupyter notebook or VSCode – Development environment




---

📁 Dataset Details

Files:

bank.csv: 4,521 samples (10% of full dataset)

bank-full.csv: 45,211 samples (full dataset)

bank-names.txt: Metadata and description


Target Variable: y (binary: yes / no)

Features: 16 input variables (categorical + numeric)

No Missing Values



---

📄 Output Files (Typical Results)

After running a pipeline, you typically produce:

model.pkl or model.joblib: Trained model

results.csv: Model predictions on test set

metrics.txt or report.json: Accuracy, F1-score, Confusion Matrix, ROC-AUC

feature_importances.csv: Ranking of features

plots/: Visualization outputs like ROC curves, confusion matrices, etc.
