import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor

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

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

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

    num_attribs = list(housing.drop('ocean_proximity', axis=1))
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

    attributes = num_attribs + extra_attribs + cat_one_hot_attribs

    return housing_prepared, housing_labels, attributes

def return_feature_importances(housing_prepared, housing_labels):
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=  42)

    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring='neg_mean_squared_error', return_train_score = True)

    grid_search.fit(housing_prepared, housing_labels)

    feature_importances = grid_search.best_estimator_.feature_importances_

    return feature_importances

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

def main():
    # Load in the data
    housing_path = os.path.join("datasets", "housing")
    housing = load_housing_data(housing_path)

    # Run through the preprocessing function
    housing_prepared, housing_labels, attributes = preprocess_housing_data(housing)

    k = 5

    feature_importances = return_feature_importances(housing_prepared, housing_labels)

    top_k_feature_indices = indices_of_top_k(feature_importances, k)
    
    print("Indices of the top k features: ", top_k_feature_indices)
    print("Top k features: ", np.array(attributes)[top_k_feature_indices])
    print("Actual top k features: ", sorted(zip(feature_importances, attributes), reverse=True)[:k])


if __name__ == '__main__':
    main()