from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error

import mlflow
import mlflow.sklearn
import joblib
import os
import pandas as pd
import warnings
import numpy as np

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_terminal()

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="mlflow")
warnings.filterwarnings("ignore", module="joblib")
warnings.filterwarnings("ignore", module="scikit")
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="mlflow.sklearn")
warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle"
)

from evaluate import evaluate_model
from data_ingestion import ingest_data

#Make artifacts directory for mlflow
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

#Direktori Definition
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "mlflow.db"))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "B.csv")

#Create and connect mlflow database
mlflow_db = os.path.join(BASE_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("UTS_Model_Deployment")


PIPE_PATH = ['model_regression_salary.pkl', 'model_classification_placement.pkl']

def auto_delete(): #auto delete the model in directory to train from scratch
    print(f"MANUAL TUNING IS ON!, AUTO-DELETING current model.pkl in directory")
    if os.path.exists(os.path.join(BASE_DIR, PIPE_PATH[0]) or os.path.join(BASE_DIR, PIPE_PATH[1])):
        os.remove(os.path.join(BASE_DIR, 'artifacts', PIPE_PATH[0]))
        os.remove(os.path.join(BASE_DIR, 'artifacts', PIPE_PATH[1]))
        print(f"\t - .pkl model has been deleted.")
    else:
        print(f"\t - There is no current model.pkl in {BASE_DIR}")
    print('\n')

auto_delete() #turn it off or on here

def train():

    ingest_data("B.csv") 
    target = ["placement_status", "salary_package_lpa"]

    print("1. Data loaded successfully for training.")
    df_final_class = pd.read_csv(os.path.join(BASE_DIR, "data", "ingested", "data_cleaned.csv")) 
    # print(df_final_class.columns)

    num_cols = df_final_class.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df_final_class.select_dtypes(include=["object"]).columns

    cat_cols = [col for col in cat_cols if col not in target]
    num_cols = [col for col in num_cols if col not in target]

    print(f"2. Preparing Preprocessor with {len(num_cols)} numerical columns and {len(cat_cols)} categorical columns.")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    ) 

    print("3. Building Pipelines for Classification and Regression.")
    pipeline_1 = Pipeline([
        ("preprocessor", preprocessor), #Best Pipeline from EDA.ipynb
        ("classifier", XGBClassifier(n_jobs = -1, random_state = 42, learning_rate = 0.05, 
                                    max_depth = 3, n_estimators = 100, scale_pos_weight = 4, 
                                    tree_method = "hist"))
    ])

    pipeline_2 = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBRegressor(n_jobs = -1, random_state = 42, learning_rate=0.01, 
                                    max_depth = 3, n_estimators = 100, subsample = 0.8, 
                                    objective = "reg:squarederror"))
    ])

    #Configuration Experiments:
    experiments = [
        {
            "name": "Classification_Placement",
            "pipeline": pipeline_1,
            "target": "placement_status",
            "type": "classification"
        },
        {
            "name": "Regression_Salary",
            "pipeline": pipeline_2,
            "target": "salary_package_lpa",
            "type": "regression"
        }
    ]

    for exp in experiments:
        print(f"\nStarting Experiment: {exp['name']}")

        if exp['type'] == 'classification':
            X = df_final_class.drop(columns=["placement_status", "salary_package_lpa"])
            y = df_final_class[exp['target']]

            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object"]).columns

            cat_cols = [col for col in cat_cols if col not in target]
            num_cols = [col for col in num_cols if col not in target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        elif exp['type'] == 'regression':
            df_final_reg = df_final_class[df_final_class['placement_status'] > 0]

            X = df_final_reg.drop(columns=["placement_status", "salary_package_lpa"])
            y = np.log1p(df_final_reg[exp['target']])

            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object"]).columns

            cat_cols = [col for col in cat_cols if col not in target]
            num_cols = [col for col in num_cols if col not in target]

            X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

        else:
            print(f"Type model invalid!")

        # mlflow.sklearn.autolog()
        with mlflow.start_run(run_name = exp['name']):
            print("4. Training the model pipeline.")
            if exp['type'] == "classification":

                exp['pipeline'].fit(X_train, y_train)

                model_step_name = "classifier" 
                model_params = exp['pipeline'].named_steps[model_step_name].get_params()

                # # Custom Threshold
                # custom_threshold = 0.8
                # y_probs = exp['pipeline'].predict_proba(X_test)[:, 1]
                # y_preds = (y_probs >= custom_threshold).astype(int)

                #No Custom Threshold
                y_preds = exp['pipeline'].predict(X_test)
                
                # We filter for the most important ones to keep MLflow clean
                important_params = {
                    "random_state": model_params.get("random_state"),
                    "learning_rate": model_params.get("learning_rate"),
                    "max_depth": model_params.get("max_depth"),
                    "n_estimators": model_params.get("n_estimators"),
                    "scale_po_weight": model_params.get("scale_post_weight"),
                    "tree_method": model_params.get("tree_method"),
                    "model_type": exp['type']
                    # "threshold": custom_threshold if exp['type'] == 'classification' else "N/A"
                }
                mlflow.log_params(important_params)

                print("5. Evaluating the model performance - Classification")
                cv_mean, cv_std = evaluate_model(exp['pipeline'], X_train, y_train, X_test, y_test, target, 0, "clf", X, y)

                accuracy = accuracy_score(y_test, y_preds)
                precision = precision_score(y_test, y_preds)
                recall = recall_score(y_test, y_preds)
                F1_score = f1_score(y_test, y_preds)
                F1_weighted = f1_score(y_test, y_preds, average = "weighted")
                ROC_AUC = roc_auc_score(y_test, exp['pipeline'].predict_proba(X_test)[:, 1])

                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1 Score", F1_score)
                mlflow.log_metric("F1 Weighted", F1_weighted)
                mlflow.log_metric("ROC-AUC", ROC_AUC)
            
                mlflow.log_metric("Cross Validation F1-Mean", cv_mean)
                mlflow.log_metric("Cross Validation F1-Std", cv_std)

            elif exp['type'] == 'regression':
                exp['pipeline'].fit(X_train, y_train_log)

                model_params = exp['pipeline'].named_steps["classifier"].get_params()    
                important_params = {
                    "random_state": model_params.get("random_state"),
                    "learning_rate": model_params.get("learning_rate"),
                    "max_depth": model_params.get("max_depth"),
                    "n_estimators": model_params.get("n_estimators"),
                    "subsample": model_params.get('subsample'),
                    "objective": model_params.get('objective'),
                    "model_type": exp['type'],
                    "threshold": custom_threshold if exp['type'] == 'classification' else "N/A"
                }
                mlflow.log_params(important_params)

                y_preds = exp['pipeline'].predict(X_test)

                y_train_norm = np.expm1(y_train_log)
                y_test_norm = np.expm1(y_test_log)
                y_preds_norm = np.expm1(y_preds)
                
                print("6. Evaluating the model performance.")

                smape_mean, smape_std = evaluate_model(exp['pipeline'], X_train, y_train_norm, X_test, y_test_norm, target, 1, "reg", X, y)

                rsq_score = r2_score(y_test_norm, y_preds_norm)
                mae_score = mean_absolute_error(y_test_norm, y_preds_norm)
                mape_score = mean_absolute_percentage_error(y_test_norm, y_preds_norm)
                mse_score = mean_squared_error(y_test_norm, y_preds_norm)
                rmse_score = root_mean_squared_error(y_test_norm, y_preds_norm)

                
                mlflow.log_metric("R2 Score", rsq_score)
                mlflow.log_metric("MAE", mae_score)
                mlflow.log_metric("MAPE", mape_score)
                mlflow.log_metric("MSE", mse_score)
                mlflow.log_metric("RMSE", rmse_score)

                mlflow.log_metric("Cross Validation SMAPE-mean", smape_mean)
                mlflow.log_metric("Cross Validation SMAPE-std", smape_std)
            else:
                print(f"Invalid Experiment Set!")
                return
            

            input_example = X_test.iloc[[0]]
            mlflow.sklearn.log_model(
                sk_model = exp['pipeline'],
                name = "model",
                input_example = input_example
            )

            model_filename = f"model_{exp['name'].lower()}.pkl"
            save_path = os.path.join(BASE_DIR, 'artifacts', model_filename)
            joblib.dump(exp['pipeline'], save_path)

            print(f"\n8. Saving the trained model pipeline to {ARTIFACTS_DIR}.")
            print(f"9. Training Pipeline succesfully done!")
    

if __name__ == "__main__":
    train()
