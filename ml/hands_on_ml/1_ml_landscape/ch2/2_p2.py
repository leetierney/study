import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from scipy.stats import expon, reciprocal

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def preprocess_housing_data(housing):
    # Create a categorical feature for the income categories
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

    # Split up the training set    
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    # Create a numerical feature
    housing_num = housing.drop('ocean_proximity', axis=1)

    rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

    # Define function to create additional features
    def add_extra_features(X, add_bedrooms_per_room = True):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]

        if add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
    # Create numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # Create a full pipeline, running the numerical pipeline and a OneHotEncoder for the categorical features
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    housing_prepared = full_pipeline.fit_transform(housing)

    return housing_prepared, housing_labels

def run_randomized_search(housing_prepared, housing_labels):
    param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

    svm_reg = SVR()
    rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                    n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                    verbose=2, n_jobs=4, random_state=42)
    
    rnd_search.fit(housing_prepared, housing_labels)

    negative_mse = rnd_search.best_score_
    rmse = np.sqrt(-negative_mse)

    print("RMSE: ", rmse)
    print("Best Parameters: ", rnd_search.best_params_)

def main():
    # Load in the data
    housing_path = os.path.join("datasets", "housing")
    housing = load_housing_data(housing_path)

    # Run through the preprocessing function
    housing_prepared, housing_labels = preprocess_housing_data(housing)

    # Run grid search
    run_randomized_search(housing_prepared, housing_labels)

if __name__ == '__main__':
    main()
