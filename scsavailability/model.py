from sklearn.linear_model import LinearRegression

def fit_n_r2(X, Y, model_type, **kwargs):
     """
    Summary
    -------
    Takes variables and fits model with arguments. Return model object.
    Parameters
    ----------
    X: pandas DataFrame
        dataframe of features
    Y: pandas Series
        series of target variables
    model_type: ModelClass
        model that we wish to test

    Returns
    -------
    model: ModelClass
        fitted model
    scores: list
        list of r2 values from cross-validation
    Example
    --------
    model, r2_scores = fit_n_r2((X, Y, LinearRegression))
    """
    model = model_type(**kwargs)
    
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    scores = cross_val_score(model, X, Y, scoring='r2', cv=folds)

    return model, scores



