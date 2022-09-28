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
import plotly.figure_factory as ff


def heatmap_plot_confusion_matrix(cm, labels, title="Confusion Matix"):
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

    # df = df.dropna()
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

    return fig, df_feature, dummy_features_df, dummy_features_df_columns


def train_decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                        min_impurity_decrease, class_weight, ccp_alpha, df_columns_dropdown_label, input_features):
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

