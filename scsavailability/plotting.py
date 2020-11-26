"""
Plotting functions for scs modelling
"""
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydot 


def plot_graphs(model, X_train, y_train, y_test, pred):
    """
    Summary
    -------
    Creates table output of model r2 value with metadata and timestamp
    Parameters
    ----------
    model: ModelClass
        fitted model
    start_date: String
        String form of date to filter by in form YYYY-MM-DD
    Returns
    -------
    df_nielsen_clean: pandas DataFrame
        The cleaned and transformed Nielsen data.
    Example
    --------
    df_nielsen_clean = process_model_data(df_nielsen)
    """
    #Output Test Scatter and Distribution

    plt.scatter(y_test,pred)
    plt.xlabel('Actual Downtime')
    plt.ylabel('Predicted Downtime')
    plt.title('Predicted vs Actual Scatter from Test')

    plt.figure()
    sns.distplot(y_test-pred)
    plt.title('Distrubution of Residuals from Test')
    plt.xlabel('Residual')

    #Output Train Scatter and Distribution

    plt.figure()
    plt.scatter(y_train,model.predict(X_train))
    plt.xlabel('Actual Downtime')
    plt.ylabel('Predicted Downtime')
    plt.title('Predicted vs Actual Scatter from Train')

    plt.figure()
    sns.distplot(y_train-model.predict(X_train))
    plt.title('Distrubution of Residuals from Train')
    plt.xlabel('Residual')

    return None


def vis_tree(df, dtree_model):
    """
    Summary
    -------
    Creates table output of model r2 value with metadata and timestamp
    Parameters
    ----------
    df: pandas DataFrame
        dataframe of features
    dtree_model: sklearn Model
        fitted model
    start_date: String
        String form of date to filter by in form YYYY-MM-DD
    Returns
    -------
    df_nielsen_clean: pandas DataFrame
        The cleaned and transformed Nielsen data.
    Example
    --------
    df_nielsen_clean = process_model_data(df_nielsen)
    """

    features = list(df.columns[2:])

    dot_data = StringIO()  
    export_graphviz(dtree_model, out_file=dot_data,feature_names=features,filled=True,rounded=True)

    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    Image(graph[0].create_png())  
    