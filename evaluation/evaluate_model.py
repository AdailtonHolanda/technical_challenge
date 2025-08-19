import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import numpy as np

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
TARGET_COLUMN = "price"

# Features to use from the home sale data
SALES_COLUMNS = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
    'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]

# Additional features to use from the demographics data
DEMOGRAPHICS_COLUMNS = [
    'ppltn_qty', 'urbn_ppltn_qty', 'sbrbn_ppltn_qty',
    'farm_ppltn_qty', 'non_farm_qty', 'medn_hshld_incm_amt',
    'medn_incm_per_prsn_amt', 'hous_val_amt', 'edctn_less_than_9_qty',
    'edctn_9_12_qty', 'edctn_high_schl_qty', 'edctn_some_clg_qty',
    'edctn_assoc_dgre_qty', 'edctn_bchlr_dgre_qty', 'edctn_prfsnl_qty',
    'per_urbn', 'per_sbrbn', 'per_farm', 'per_non_farm',
    'per_less_than_9', 'per_9_to_12', 'per_hsd',
    'per_some_clg', 'per_assoc', 'per_bchlr', 'per_prfsnl'
]


def load_data(sales_path: str, demographics_path: str) -> pd.DataFrame:
    """
    Loads data by merging sales and demographics datasets.
    """
    sales_data = pd.read_csv(sales_path, usecols=SALES_COLUMNS)
    demographics_data = pd.read_csv(demographics_path)

    demographics_data.rename(columns={'zipcode': 'zipcode'}, inplace=True)
    merged_data = pd.merge(sales_data, demographics_data, on='zipcode')

    return merged_data


def evaluate_model():
    """
    Loads data, trains a model, and evaluates its performance on a test set.
    """
    print("Starting model evaluation...")

    df = load_data(SALES_PATH, DEMOGRAPHICS_PATH)
    if df is None:
        return

    print(f"Successfully loaded and merged data. Total number of examples: {len(df)}.")

    FEATURES_TO_USE = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[FEATURES_TO_USE]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        RobustScaler(),
        KNeighborsRegressor(n_neighbors=10, weights='uniform', p=2)
    )
    model.fit(X_train, y_train)
    print("\nModel trained successfully on the training set.")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "="*50)
    print("Model Performance Metrics")
    print("="*50)
    print(f"Mean Squared Error (MSE) on Training Set: {train_mse:.2f}")
    print(f"Mean Squared Error (MSE) on Test Set: {test_mse:.2f}")
    print("-" * 50)
    print(f"R-squared ($R^2$) on Training Set: {train_r2:.4f}")
    print(f"R-squared ($R^2$) on Test Set: {test_r2:.4f}")
    print("="*50)

    if not np.isinf(train_mse) and not np.isinf(test_mse):
        print("\nGenerating error metrics plot...")
        labels = ['MSE (Train)', 'MSE (Test)']
        metrics = [train_mse, test_mse]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, metrics, color=['blue', 'red'])
        plt.title('Mean Squared Error (MSE)', fontsize=16)
        plt.ylabel('Metric Value', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, v in enumerate(metrics):
            plt.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('mse_metrics.png')
        print("MSE plot saved as 'mse_metrics.png'.")

    if not np.isinf(train_r2) and not np.isinf(test_r2):
        print("\nGenerating goodness-of-fit plot...")
        labels = ['R² (Train)', 'R² (Test)']
        metrics = [train_r2, test_r2]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, metrics, color=['green', 'orange'])
        plt.title('R-squared ($R^2$)', fontsize=16)
        plt.ylabel('Metric Value', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)

        for i, v in enumerate(metrics):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('r2_metrics.png')
        print("R² plot saved as 'r2_metrics.png'.")


if __name__ == "__main__":
    evaluate_model()