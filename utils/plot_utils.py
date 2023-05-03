import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import (roc_curve,
                             roc_auc_score,
                             auc,
                             confusion_matrix)


def plot_correlation_matrix(df: pd.DataFrame):
    """Creates a heatmap of the correlation matrix
    for a pandas DataFrame
    Args:
        df (pd.DataFrame): input DataFrame
    Returns: None
    """
    corr_matrix = df.corr()

    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale='Viridis',
        showscale=True
    )

    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(title='Columns', side='bottom'),
        yaxis=dict(title='Columns'),
        width=800, height=800,
        margin=dict(l=100, r=100, t=100, b=100)
    )
    fig.show()


def plot_categorical_variable(df, variable, is_horizontal=True):
    """
    Creates a horizontal bar chart for a categorical variable
    with different categories
    Args:
        df (pd.DataFrame): The input DataFrame
        variable (str): The name of the categorical variable to be plotted
        is_horizontal (bool, optional): Whether to plot the bars
        horizontally or vertically. Defaults to True (horizontal)
        is_horizontal (bool, optional):
        Whether to plot the bars horizontally or vertically
        Defaults to True (horizontal)
    Returns:
        None
    """
    df[variable] = df[variable].astype('category')
    counts = df.groupby(variable).size().reset_index(name='count')

    if is_horizontal:
        x = 'count'
        y = variable
        width = 800
        height = 500
    else:
        x = variable
        y = 'count'
        width = 500
        height = 800
    fig = px.bar(counts, x=x, y=y)

    fig.update_layout(
        title=f"{variable} Distribution",
        xaxis_title="Count" if is_horizontal else variable,
        yaxis_title=variable if is_horizontal else "Count",
        width=width, height=height,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    fig.show()


def plot_grouped_bar(df, x, group, is_percantage=False):
    """
    Creates a grouped bar chart for two variables

    Args:
        df (pd.DataFrame): The input DataFrame
        containing the data to be plotted
        x (str): The name of the variable to be plotted on the x-axis
        group (str): The name of the categorical variable
        to be used for grouping the data
        is_percantage (bool): If True,
        the y-axis will be shown as a percentage. Default is False

    Returns:
        None
    """
    histnorm = 'percent' if is_percantage else None
    fig = px.histogram(df, x=x, color=group,
                       barmode='group', histfunc='count',
                       color_discrete_sequence=px.colors.qualitative.Plotly,
                       histnorm=histnorm,
                       text_auto=True)

    title_percent = 'in %' if histnorm else ""
    fig.update_layout(
        title=f"{x} grouped by {group} {title_percent}",
        xaxis_title=x,
        legend_title=group,
        width=800, height=800,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    fig.show()


def plot_histogram(data, column, bins=None):
    """""
    Function to create a histogram plot

    Args:
    data (pd.DataFrame): The DataFrame containing the data to be plotted
    column (str): The column name in the DataFrame
    for which the histogram is to be plotted
    bins (int, optional): The number of bins to use for the histogram
    If not provided, Plotly will determine the number automatically

    Returns:
    plotly.graph_objects.Figure
    """

    fig = go.Figure(data=[go.Histogram(x=data[column], nbinsx=bins)])
    fig.update_layout(width=800, height=800,
                      title=f'{column} histogram',
                      xaxis=dict(title='Columns',
                                 side='bottom'),
                      yaxis=dict(title='Columns'),
                      margin=dict(l=100, r=100, t=100, b=100))
    return fig


def plot_histogram_by_group(df, cont_col, cat_col, bins=None):
    """
    Function to create a histogram grouped by a categorical variable

    Args:
        df (pd.DataFrame): The input DataFrame
        cont_col (str): The name of the continuous column to plot
        cat_col (str): The name of the categorical column to group by
    Returns:
        plotly.graph_objects.Figure
    """

    # Create the histogram
    fig = px.histogram(df, x=cont_col, color=cat_col, nbins=bins)
    fig.update_layout(width=800, height=800,
                      title=f'{cont_col} histogram grouped by {cat_col}',
                      xaxis=dict(title=f'{cont_col}', side='bottom'),
                      yaxis=dict(title='Count'),
                      margin=dict(l=100, r=100, t=100, b=100))
    return fig


def plot_scatterplot_matrix(df, dimensions, color):
    """
    Create a scatterplot matrix using Plotly.

    Args:
        df (pd.DataFrame): The input DataFrame.
        dimensions (list): List of column names to be included
        in the scatterplot matrix.
        color (str): The name of the column to be used for coloring the points

    Returns:
        plotly.graph_objs._figure.Figure
    """
    color_palette = px.colors.qualitative.Alphabet
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color,
                            color_discrete_sequence=color_palette)
    fig.update_layout(
        width=1200,
        height=800,
        title='Scatter plot',
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_traces(showlegend=True, selector=dict(type='scatter'))
    return fig


def plot_box_by_target_class(df, column):
    """
    Function to create a box plot using Plotly

    Args:
    data (pd.DataFrame): The DataFrame containing the data to be plotted
    column (str): The column name in the DataFrame
    for which the box plot is to be plotted

    Returns:
    plotly.graph_objs._figure.Figure
    """

    fig = go.Figure()
    fig.add_trace(go.Box(y=df[column],
                         x=df[df['classLabel'] == 0].classLabel,
                         name='target class 0',
                         marker_color='indianred'))
    fig.add_trace(go.Box(y=df[column],
                         x=df[df['classLabel'] == 1].classLabel,
                         name='target class 1',
                         marker_color='lightseagreen'))

    fig.update_layout(width=800, height=800,
                      title=f'{column} box plot',
                      xaxis=dict(title='classLabel',
                                 side='bottom'),
                      yaxis=dict(title='Values'),
                      margin=dict(l=100, r=100, t=100, b=100))
    return fig


def plot_roc_curve(model, X_test, y_test, name):
    """
    This function creates and displays an ROC curve for the given model

    Args:
        model (sklearn.estimator): The trained model to evaluate
        X_test (numpy.ndarray or pandas.DataFrame): The test features
        y_test (numpy.ndarray or pandas.Series): The actual labels
        name (str): The name of the model (used for the plot's title)

    Returns:
        None
    """
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure(data=[
        go.Scatter(x=fpr, y=tpr, name='ROC curve (area = %0.2f)' % roc_auc),
        go.Scatter(x=[0, 1], y=[0, 1],
                   mode='lines',
                   name='Random',
                   line=dict(color='red', dash='dash'))
    ])
    fig.update_layout(title_text=name+' ROC Curve',
                      autosize=False,
                      width=500, height=500,
                      template='plotly_dark',
                      margin=dict(l=100, r=100, t=100, b=100))
    fig.show()


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Plot a confusion matrix for the given model

    Args:
        model (sklearn.estimator): The trained model to evaluate
        X_test (numpy.ndarray or pandas.DataFrame): The test features
        y_test (numpy.ndarray or pandas.Series): The true labels for the test
        labels (list): The unique class labels
        title (str, optional): The title for the confusion matrix plot

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    # labels = np.unique(np.concatenate((y_test, y_pred)))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig = ff.create_annotated_heatmap(z=cm,
                                      x=[0, 1],
                                      y=[0, 1],
                                      colorscale='Viridis')

    # Update layout
    fig.update_layout(
        title_text=f'<i><b>{model_name} confusion matrix</b></i>',
        xaxis=dict(title='Predicted label', side='bottom'),
        yaxis=dict(title='True label'),
        autosize=False,
        template='plotly_dark',
        width=500, height=500,
        margin=dict(l=100, r=100, b=100, t=100, pad=10)
    )
    fig.show()
