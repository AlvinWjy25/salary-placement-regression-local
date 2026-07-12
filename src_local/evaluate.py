from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
import joblib
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

BASE_DIR = os.getcwd()
MODEL_PATH_1 = os.path.join(BASE_DIR, 'artifacts', "model_classification_placement.pkl")
MODEL_PATH_2 = os.path.join(BASE_DIR, 'artifacts', "model_regression_salary.pkl")

def smape_calculate(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    ) * 100

def evaluate_model(model_pipeline, X_train, y_train, X_test, y_test, target, which, mode, X, y):
    if not os.path.exists(MODEL_PATH_1 and MODEL_PATH_2):
        print(f"Model file not found at {MODEL_PATH_1}.\nUsing model from train_pipeline.py for evaluation instead.")
        model = model_pipeline
    else:
        if mode == "clf":
            print(f"Loading model from {MODEL_PATH_1} for evaluation.")
            model = joblib.load(MODEL_PATH_1)
        elif mode == "reg":
            print(f"Loading model from {MODEL_PATH_2} for evaluation.")
            model = joblib.load(MODEL_PATH_2)
        else:
            print(f"Unable to find mode of model training: {mode}")
            return KeyError
    
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "ingested", "data_cleaned.csv"))

    if mode == 'clf':
        X = df.drop(target, axis=1)
        y = df[target[which]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv = 5, scoring = 'f1_macro')

        y_preds = model.predict(X_test)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
        f1_macro = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
        f1_weighted = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

        print(f"=" * 24, "CV - Classfication Evaluation", "=" * 24)
        print(f"Cross-Validated Accuracy: {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
        print(f"Cross-Validated Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
        print(f"Cross-Validated Recall: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
        print(f"Cross-Validated F1-macro: {f1_macro.mean():.4f} ± {f1_macro.std():.4f}")
        print(f"Cross-Validated F1-weighted: {f1_weighted.mean():.4f} ± {f1_weighted.std():.4f}")
        print(f"=" * 80)

        return cv_scores.mean(), cv_scores.std()
    
    elif mode == 'reg':
        X = X.reset_index(drop = True)
        y = y.reset_index(drop = True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_list = []
        mae_list = []
        mse_list = []
        rmse_list = []
        mape_list = []
        smape_list = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]


            model.fit(X_train_cv, y_train_cv)

            y_pred_log = model.predict(X_val_cv)
            y_pred = np.expm1(y_pred_log) # Back to normal units for metrics
            
            # y_val_cv is already in normal units because we passed y_train_norm
            y_val = np.expm1(y_val_cv)

            mask = y_val > 0

            y_val_safe = y_val[mask]
            y_pred_safe = y_pred[mask]

            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_val_safe - y_pred_safe) / y_val_safe)) * 100
            smape = smape_calculate(y_val, y_pred)

            r2_list.append(r2)
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            mape_list.append(mape)
            smape_list.append(smape)

            comparison_df = pd.DataFrame({
                'Actual Salary (LPA)': y_val[:10].values,
                'Predicted Salary (LPA)': y_pred[:10],
                'Difference': y_val[:10].values - y_pred[:10]
            })
        
        print(f"=" * 24, "CV - Regression Evaluation", "=" * 25)
        print(f"R^2: {np.mean(r2_list):.4f} ± {np.std(mae_list):.4f}")
        print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
        print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
        print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")        
        print(f"MAPE: {np.mean(mape_list):.4f}% ± {np.std(mape_list):.4f}")
        print(f"SMAPE: {np.mean(smape_list):.4f}% ± {np.std(smape_list):.4f}")
        print(f"=" * 80)

        print(comparison_df.round(2).to_string(index=False))
        print(f"=" * 80)
        
        return np.mean(smape_list), np.std(smape_list)
    
    else:
        print(f"Unknown model type!")

if __name__ == "__main__":
    evaluate_model()
