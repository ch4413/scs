from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

import pandas as pd
import numpy as np
from . import logger


@logger.logger
def gen_feat_var(df, target="Availability", features=["Faults", "Totes"]):
    """
    Summary
    -------
    Splits dataframe into features and target variable
    ----------
    df: pandas DataFrame
        dataframe of features and target variables
    target: string
        sets target variable (y)
    features: list
        set what to include in features (X)
    Returns
    -------
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    Example
    --------
    X, y = gen_feat_var(df,
                        target="Availability",
                        features=["Faults", "Totes"])
    """
    # Removes any rows where there is no target variable data
    df = df[~df[target].isnull()]

    # Selects Features DataFrame
    if "Faults" in features and "Totes" in features:
        X = df.drop(['Availability', 'timestamp'], axis=1)
    elif "Faults" in features:
        X = df.drop(['Availability', 'timestamp', 'log_totes'], axis=1)
    elif "Totes" in features:
        X = pd.DataFrame(df['log_totes'])
    else:
        X = df.drop(['Availability', 'timestamp'], axis=1)
        print('Features not valid, returning all')

    # Selects Target Variable
    y = df[target]

    return X, y


@logger.logger
def split(X, y, split_options={'test_size': 0.3, 'random_state': None}):
    """
    Summary
    -------
    Performs test train split on features and target variable
    ----------
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    test_size: Numeric
        proportion of data in test
    random_state: Integer
        sets seed if required
    Returns
    -------
    X_train: pandas DataFrame
        dataframe of training features
    X_test: pandas Series
        dataframe of test features
    y_train: pandas Series
        series of training target variables
    y_test: pandas Series
        series of test target variables
    Example
    --------
    X_train, X_test, y_train, y_test = split(X,y,test_size=0.3,random_state=101)
    """
    # train_test_split
    X_train, X_test, y_train, y_test =\
        train_test_split(X,
                         y,
                         test_size=split_options['test_size'],
                         random_state=split_options['random_state'])

    return X_train, X_test, y_train, y_test


@logger.logger
def cross_validate_r2(model, X, y, n_folds=5, shuffle=True, random_state=None):
    """
    Summary
    -------
    Generates cross-validated R2 scores for a given number of folds
    ----------
    model: sklearn model object
        fitted linear regression model
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    n_folds: numeric
        number of splits of the data
    shuffle: Boolean
        choose to shuffle data before splitting
    random_state: Integer
        sets seed if required
    Returns
    -------
    scores.mean(): float
        mean of cross validated R2 scores
    Example
    --------
    cv_R2 = cross_validate_r2(model, X, y, n_folds = 10, shuffle = True,
    random_state = 101)
    """
    # Creates dataframe to be filled
    df_cross_val = \
        pd.DataFrame(index=[str(i) for i in range(1, n_folds + 1)]
                     + ['Mean', 'STD'])

    # Set number of fold to be create in the dataframe
    folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    # Performs cross validation to generate R2 scores
    scores = cross_val_score(estimator=model, X=X, y=y, scoring='r2', cv=folds)
    # Append dataframe with scores and suammary, then prints
    df_cross_val[' R2 Scores'] = np.append(scores, [scores.mean(), scores.std()])
    print('\nCross Validation Scores: \n \n', df_cross_val)

    return scores.mean()


@logger.logger
def find_features(X_train, y_train, n):
    """
    Summary
    -------
    Performs backwards elimination to remove features that
    don't have significant relationship the target
    variable
    ----------
    X_train: pandas DataFrame
        dataframe of training features
    y_train: pandas Series
        series of training target variables
    n: numeric
        number of features to remove each iteration
    Returns
    -------
    X_train.columns: list
        list of selected features
    Example
    --------
    selected_feat = find_features(X_train, y_train, n)
    """
    X_train = X_train.copy()
    # Set max_p to run first loop
    max_p = 1
    # Loops until max_p drops below threshold
    while max_p > 0.1:
        # Create model
        model = sm.OLS(y_train, X_train)
        # Extract results
        results = model.fit()
        # Find features the top n p_values
        top_n = results.pvalues.sort_values(ascending=False).head(n)
        # Save the highest p
        max_p = top_n.tail(1).values[0]
        # Remove top n p_values from features
        rm_col = list(results.pvalues.sort_values(ascending=False).head(n).index)
        X_train = X_train.drop(rm_col, axis=1)
        #print(X_train.shape[1])
    return X_train.columns


@logger.logger
def run_OLS(X_train, y_train, X_test, y_test, n, n_folds=5):
    """
    Summary
    -------
    Runs ordinary linear regression model
    ----------
    X_train: pandas DataFrame
        dataframe of training features
    X_test: pandas Series
        dataframe of test features
    y_train: pandas Series
        series of training target variables
    y_test: pandas Series
        series of test target variables
    n: numeric
        number of features to remove each iteration
    n_folds: integer
        number of splits of the data
    Returns
    -------
    r2_oos: numeric
        Out of sample R2 score
    coefficients: pandas DataFrame
        dataframe of select assets and their coefficients
    num_assets: numeric
        number of selected assets
    Example
    --------
    r2_oos, coefficients, num_assets = run_OLS(X_train,
                                               y_train,
                                               X_test,
                                               y_test,
                                               n=30,
                                               n_folds=5)
    """
    # Create wrapper so stats model can be used with sklearn cross_val
    class SMWrapper(BaseEstimator, RegressorMixin):
        """ A universal sklearn-style wrapper for statsmodels regressors """
        def __init__(self, model_class, fit_intercept=True):
            self.model_class = model_class
            self.fit_intercept = fit_intercept

        def fit(self, X, y):
            if self.fit_intercept:
                X = sm.add_constant(X)
            self.model_ = self.model_class(y, X)
            self.results_ = self.model_.fit()

        def predict(self, X):
            if self.fit_intercept:
                X = sm.add_constant(X)
            return self.results_.predict(X)

    # Find selected features
    keep_features = find_features(X_train=X_train, y_train=y_train, n=n)

    # Run OLS model with selected features
    model = sm.OLS(y_train, X_train[keep_features])
    results = model.fit()

    # Perform cross validation
    _ = cross_validate_r2(model=SMWrapper(sm.OLS),
                          X=X_train[keep_features],
                          y=y_train,
                          n_folds=n_folds)
    print(results.summary())

    # Extract only the coefficients that are negative
    # with respect to availabiltiy
    negs = results.params[results.params < 0]
    # Create coefficients dataframe
    coefficients = pd.DataFrame(negs, columns=['Coefficient']).reset_index()

    # Save number of selected assets
    num_assets = len(negs)
    print(num_assets)

    coefficients.rename(columns={'index': 'Asset Code'}, inplace=True)

    # Calucate OOS R2 Score
    OutY_Predicted = results.predict(X_test[keep_features])
    r2_oos = r2_score(y_test, OutY_Predicted)
    print(r2_oos)

    return r2_oos, coefficients, num_assets
