🧠 Mental Tiredness Score Prediction

An end-to-end machine learning project that predicts a person's mental tiredness score (a regression problem) from daily work, health, lifestyle and environment features.

The project covers the full MLOps flow: a reproducible DVC pipeline, hyperparameter tuning with Optuna, experiment tracking with MLflow, a Streamlit web app, and Docker deployment.

📌 Highlights
10 regression algorithms compared: Ridge, Lasso, ElasticNet, KNN, Decision Tree, SVR, Random Forest, Extra Trees, Gradient Boosting and Hist Gradient Boosting
Optuna tuning with 20 trials per algorithm, scored by cross-validated RMSE
MLflow tracking: each algorithm gets its own experiment with metrics, plots and the saved model
DVC pipeline (prepare → train → evaluate) with versioned data, models and metrics
Streamlit app with a styled UI for live predictions
Docker image published on Docker Hub
🗂️ Project Structure
.
├── data/
│   └── raw/
│       └── mental_tiredness.csv      # raw dataset (tracked by DVC)
├── src/
│   ├── prepare.py                    # clean, feature engineering, train/test split
│   ├── train.py                      # Optuna + MLflow, 10 models, saves best model
│   └── evaluate.py                   # test-set metrics -> metrics.json
├── models/
│   └── model.pkl                     # best pipeline (tracked by DVC)
├── app.py                            # Streamlit app
├── params.yaml                       # pipeline settings
├── dvc.yaml                          # pipeline definition
├── dvc.lock                          # exact versions of pipeline inputs/outputs
├── metrics.json                      # evaluation metrics (tracked by Git)
├── requirements.txt                  # dependencies for the app / Docker image
├── Dockerfile
└── .dockerignore
📊 Dataset

The dataset contains daily activity, health and environment information for each record.

Group	Features
Work activity	decisions made, context switches, notifications, screen time, deep work minutes, task complexity, workload score, breaks
Health & lifestyle	sleep hours, deep sleep %, caffeine, hydration, mood
Environment	work type, work environment, noise level, temperature

Target: mental_tiredness_score

Engineered features (created in prepare.py):

deep_sleep_hours = sleep_hours × deep_sleep_pct / 100
workload_intensity = workload_score × context_switch_count
⚙️ Pipeline
data/raw/mental_tiredness.csv
        │
        ▼
   [ prepare ]  →  data/processed/train.csv, test.csv
        │
        ▼
   [  train  ]  →  models/model.pkl, models/leaderboard.csv, models/best_model_info.json
        │
        ▼
   [ evaluate ] →  metrics.json
prepare: cleans column names, removes duplicates, adds the engineered features and splits the data 80/20.
train: tunes all 10 algorithms with Optuna, logs each one to MLflow, and saves the model with the lowest cross-validated RMSE. The best model is chosen on cross-validation, not on the test set, to avoid test leakage.
evaluate: scores the best model on the held-out test set and writes metrics.json.

Every model is a full scikit-learn pipeline (imputation, scaling, one-hot encoding, optional polynomial features, regressor), so raw input goes straight in.

Settings live in params.yaml:

yaml
prepare:
  data_path: data/raw/mental_tiredness.csv
  test_size: 0.2
  random_state: 42

train:
  n_trials: 20
  cv_folds: 5
  random_state: 42
📈 Results

Run dvc metrics show to see the latest numbers, or open metrics.json. The full ranking of all 10 algorithms is in models/leaderboard.csv.

Metric	Value
Best model	(see models/best_model_info.json)
Test RMSE	(see metrics.json)
Test MAE	(see metrics.json)
Test R²	(see metrics.json)
🚀 Getting Started
1. Clone the repository
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Install dependencies
bash
pip install dvc optuna mlflow scikit-learn pandas numpy matplotlib joblib cloudpickle pyyaml streamlit
3. Get the data and model

The data and model are stored with DVC, not in Git:

bash
dvc pull

The DVC remote in this project is a local folder, so dvc pull only works on the machine that created it. To use another machine, point the remote to shared storage (DagsHub, Google Drive or S3) with dvc remote modify and run dvc push once from the original machine.

4. Run the pipeline
bash
dvc repro            # runs only the stages whose inputs changed
dvc metrics show     # prints the evaluation metrics
dvc dag              # draws the pipeline graph

Changing a value in params.yaml (for example n_trials) and running dvc repro again re-runs only the affected stages.

5. View MLflow experiments
bash
mlflow ui

Open http://127.0.0.1:5000. Each of the 10 algorithms appears as its own experiment.

6. Run the Streamlit app
bash
streamlit run app.py

Open http://localhost:8501, enter the values and click Predict.

🐳 Docker
Run the published image

Anyone with Docker installed can run the app:

bash
docker run -p 8501:8501 nithindev01/mental-tiredness-app:v1

Then open http://localhost:8501.

Build it yourself
bash
dvc pull models/model.pkl      # make sure the model file exists
docker build -t mental-tiredness-app .
docker run -p 8501:8501 mental-tiredness-app
Publish a new version
bash
docker tag mental-tiredness-app <dockerhub-username>/mental-tiredness-app:v2
docker push <dockerhub-username>/mental-tiredness-app:v2

Docker Hub usernames and image names must be lowercase.

🔁 Retraining Workflow
Edit params.yaml (for example change n_trials).
dvc repro then dvc push.
Rebuild and push the Docker image with a new version tag.
git add the changed files (dvc.lock, metrics.json, params.yaml), commit and push.
🛠️ Tech Stack

Python 3.11 · scikit-learn · Optuna · MLflow · DVC · Streamlit · Docker · Pandas · NumPy · Matplotlib

📝 Notes
Model pickles depend on the scikit-learn version. requirements.txt pins the same version used for training, so keep them in sync.
The prediction score is floored at 0, and the Low / Moderate / High labels in the app (below 20, 20 to 40, above 40) are rough guide bands, not clinical thresholds.
This project predicts a score from lifestyle data for learning purposes. It is not a medical tool.
👤 Author

Nithin Dev GitHub: nithindev-01
