import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plots import dt_graph


def eda_graph_plot(df, x_axis_features, y_axis_features, graph_type):

    """
    :param df: The Data Frame from the app.py
    :param x_axis_features: The Feature for x-axis
    :param y_axis_features: The Feature for y-axis
    :param graph_type: The Graph type which is selected
    :return: return on Graph type as fig
    """

    df = df.dropna()
    if graph_type == "Scatter":
        fig = px.scatter(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Line":
        fig = px.line(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == 'Area':
        fig = px.area(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == 'Bar':
        fig = px.bar(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == 'Funnel':
        fig = px.funnel(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == 'Timeline':
        fig = px.timeline(data_frame=df, x_start=x_axis_features, x_end=y_axis_features)
    elif graph_type == 'Pie':
        fig = px.pie(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values)
    elif graph_type == 'Subburst':
        fig = px.sunburst(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values)
    elif graph_type == 'Treemap':
        fig = px.treemap(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values)
    elif graph_type == "Icicle":
        fig = px.icicle(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values)
    elif graph_type == "Funnel Area":
        fig = px.funnel_area(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values)
    elif graph_type == "Histogram":
        fig = px.histogram(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Box":
        fig = px.box(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Violin":
        fig = px.violin(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Strip":
        fig = px.strip(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "ECDF":
        fig = px.ecdf(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Violin":
        fig = px.violin(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Density Heatmap":
        fig = px.density_heatmap(data_frame=df, x=x_axis_features, y=y_axis_features)
    elif graph_type == "Density Contour":
        fig = px.density_contour(data_frame=df, x=x_axis_features, y=y_axis_features)
    else:
        fig = px.histogram(data_frame=df, x=x_axis_features, y=y_axis_features)


    return fig


def heatmap_plot_confusion_matrix(cm, labels, title="Confusion Matix"):

    """ This function is not working so, it is not used
    :param cm: This is coonfusion matrix
    :param labels: list(df[df_columns_dropdown_label].unique()) the unique use in the label column
    :param title: Title of the figure
    :return: returns a fig
    """
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {"x": labels[i],
                 "y": labels[j],
                 "font": {"color": "white"},
                 "text": str(value),
                 "xref": "x1",
                 "yref": "y1",
                 "showarrow": False
                 }
            )
            layout = {
                "title": title,
                "xaxis": {"title": "Predicted value"},
                "yaxis": {"title": "Real value"},
                "annotations": annotations
            }
            fig = go.Figure(data=data, layout=layout)
            return fig


def ff_plot_confusion_matrix(z, x, y):

    """
    call as: ff_plot_confusion_matrix(cm, df_columns_dropdown_label, df_columns_dropdown_label)
    :param z: It is the confusion matrix
    :param x: x and y are the same as unique values in the label column
    :param y: same as x
    :return: returns a fig
    """
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    return fig


def get_dummy_variables(df, df_columns_dropdown_label):

    """
    This function is not used as I found to create dataframe and interactable data cells in the app.py program
    :param df: The Data Frame
    :param df_columns_dropdown_label: the columns of the Data Frame
    :return: return list of dictionaries of columns and append data cells with all values as zero
    """
    df = df.dropna()
    X = pd.get_dummies(df.drop(df_columns_dropdown_label, axis=1), drop_first=True)
    dummy_cols = X.columns
    col_dict = []
    for col in dummy_cols:
        col_dict.append({'name': col, 'id': 'col'})

    data = []
    data_dict = {}
    for item in dummy_cols:
        data_dict[item] = 0
    data.append(data_dict)
    return col_dict, data


def decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                  max_features, random_state, max_leaf_nodes, min_impurity_decrease, class_weight, ccp_alpha,
                  df_columns_dropdown_label):

    """
    These are the parameters for the Model to be trained
    :param df:
    :param criterion:
    :param splitter:
    :param max_depth:
    :param min_samples_split:
    :param min_samples_leaf:
    :param min_weight_fraction_leaf:
    :param max_features:
    :param random_state:
    :param max_leaf_nodes:
    :param min_impurity_decrease:
    :param class_weight:
    :param ccp_alpha:
    :param df_columns_dropdown_label:
    :return:
    """

    df = df.dropna()
    X = pd.get_dummies(df.drop(df_columns_dropdown_label, axis=1), drop_first=True)
    y = df[df_columns_dropdown_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                   random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                   min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
                                   ccp_alpha=ccp_alpha)
    model.fit(X_train, y_train)
    base_pred = model.predict(X_test)

    cm = confusion_matrix(y_test,base_pred)
    df_columns_dropdown_label = list(df[df_columns_dropdown_label].unique())
    fig = ff_plot_confusion_matrix(cm, df_columns_dropdown_label, df_columns_dropdown_label)

    plt.figure(figsize=(12, 8),dpi=150)
    plot_tree(model)
    plt.savefig('dt_tree',filled=True,feature_names=X.columns)

    data = model.feature_importances_
    data = data.round(3)

    df_feature = pd.DataFrame(index=X.columns,data=data).reset_index()
    df_feature.columns = ['Feature Name', 'Feature Importance']
    df_feature = df_feature.sort_values(by='Feature Importance', ascending=False)

    dummy_features_df = X[:1]
    dummy_features_df_columns = list(X.columns)

    dt_tree_graph = dt_graph(df, model)

    return fig, df_feature, dummy_features_df, dummy_features_df_columns, dt_tree_graph


def train_decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                        min_impurity_decrease, class_weight, ccp_alpha, df_columns_dropdown_label, input_features):

    """
    This is the same as above function, but it will train the model on whole Data and also return Prediction
    :param df:
    :param criterion:
    :param splitter:
    :param max_depth:
    :param min_samples_split:
    :param min_samples_leaf:
    :param min_weight_fraction_leaf:
    :param max_features:
    :param random_state:
    :param max_leaf_nodes:
    :param min_impurity_decrease:
    :param class_weight:
    :param ccp_alpha:
    :param df_columns_dropdown_label:
    :param input_features:
    :return:
    """
    X = pd.get_dummies(df.drop(df_columns_dropdown_label, axis=1), drop_first=True)
    y = df[df_columns_dropdown_label]
    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                   random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                   min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
                                   ccp_alpha=ccp_alpha)
    model.fit(X, y)
    #input_features = input_features.split(" ")
    #input_features = [float(x) for x in input_features]


    prediction = model.predict([input_features])

    return prediction

