import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plots import dt_plotly, dt_heatmap_graph, ff_plot_confusion_matrix


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

    model_accuracy_score = accuracy_score(y_true=y_test, y_pred=base_pred)
    model_accuracy_score = round(model_accuracy_score, 4)

    cm = confusion_matrix(y_test, base_pred)
    df_columns_dropdown_label = list(df[df_columns_dropdown_label].unique())
    fig = ff_plot_confusion_matrix(cm, df_columns_dropdown_label, df_columns_dropdown_label)

    dt_fig_plotly = dt_plotly(model)

    data = model.feature_importances_
    data = data.round(3)

    df_feature = pd.DataFrame(index=X.columns, data=data).reset_index()
    df_feature.columns = ['Feature Name', 'Feature Importance']
    df_feature = df_feature.sort_values(by='Feature Importance', ascending=False)

    dummy_features_df = X[:1]
    dummy_features_df_columns = list(X.columns)

    dt_tree_graph = dt_heatmap_graph(df, model)

    return fig, df_feature, dummy_features_df, dummy_features_df_columns, dt_tree_graph, model_accuracy_score, dt_fig_plotly


def train_decision_tree(df, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
                        min_impurity_decrease, class_weight, ccp_alpha, df_columns_dropdown_label):
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

    data = model.feature_importances_
    data = data.round(3)

    df_feature_trained = pd.DataFrame(index=X.columns, data=data).reset_index()
    df_feature_trained.columns = ['Feature Name', 'Feature Importance']
    df_feature_trained = df_feature_trained.sort_values(by='Feature Importance', ascending=False)

    # input_features = input_features.split(" ")
    # input_features = [float(x) for x in input_features]

    # prediction = model.predict([input_features])

    # return prediction
    return model, df_feature_trained


def randomforest_classifier(df, n_estimators_rfc, criterion_rfc, max_depth_rfc, min_samples_split_rfc,
                            min_samples_leaf_rfc, min_weight_fraction_leaf_rfc, max_features_rfc, max_leaf_nodes_rfc,
                            min_impurity_decrease_rfc, bootstrap_rfc, oob_score_rfc, random_state_rfc, ccp_alpha_rfc,
                            df_columns_dropdown_label_rfc):
    """
    :param df:
    :param n_estimators_rfc:
    :param criterion_rfc:
    :param max_depth_rfc:
    :param min_samples_split_rfc:
    :param min_samples_leaf_rfc:
    :param min_weight_fraction_leaf_rfc:
    :param max_features_rfc:
    :param max_leaf_nodes_rfc:
    :param min_impurity_decrease_rfc:
    :param bootstrap_rfc:
    :param oob_score_rfc:
    :param random_state_rfc:
    :param ccp_alpha_rfc:
    :param df_columns_dropdown_label_rfc:
    :return:
    """

    df = df.dropna()
    X = pd.get_dummies(df.drop(df_columns_dropdown_label_rfc, axis=1), drop_first=True)
    y = df[df_columns_dropdown_label_rfc]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    model = RandomForestClassifier(n_estimators=n_estimators_rfc, criterion=criterion_rfc, max_depth=max_depth_rfc,
                                   min_samples_split=min_samples_split_rfc, min_samples_leaf=min_samples_leaf_rfc,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf_rfc, max_features=max_features_rfc,
                                   max_leaf_nodes=max_leaf_nodes_rfc, min_impurity_decrease=min_impurity_decrease_rfc,
                                   bootstrap=bootstrap_rfc, oob_score=oob_score_rfc, random_state=random_state_rfc,
                                   ccp_alpha=ccp_alpha_rfc)
    model.fit(X_train, y_train)
    base_pred = model.predict(X_test)

    model_accuracy_score = accuracy_score(y_true=y_test, y_pred=base_pred)
    model_accuracy_score = round(model_accuracy_score, 4)

    cm = confusion_matrix(y_test, base_pred)
    df_columns_dropdown_label_rfc = list(df[df_columns_dropdown_label_rfc].unique())
    fig = ff_plot_confusion_matrix(cm, df_columns_dropdown_label_rfc, df_columns_dropdown_label_rfc)

    data = model.feature_importances_
    data = data.round(3)

    df_feature = pd.DataFrame(index=X.columns, data=data).reset_index()
    df_feature.columns = ['Feature Name', 'Feature Importance']
    df_feature = df_feature.sort_values(by='Feature Importance', ascending=False)

    dummy_features_df = X[:1]
    dummy_features_df_columns = list(X.columns)

    return fig, df_feature, dummy_features_df, dummy_features_df_columns, model_accuracy_score


def train_randomforest_classifier(df, n_estimators_rfc, criterion_rfc, max_depth_rfc, min_samples_split_rfc,
                                  min_samples_leaf_rfc, min_weight_fraction_leaf_rfc, max_features_rfc,
                                  max_leaf_nodes_rfc, min_impurity_decrease_rfc, bootstrap_rfc, oob_score_rfc,
                                  random_state_rfc, ccp_alpha_rfc, df_columns_dropdown_label_rfc):
    """
    :param df:
    :param n_estimators_rfc:
    :param criterion_rfc:
    :param max_depth_rfc:
    :param min_samples_split_rfc:
    :param min_samples_leaf_rfc:
    :param min_weight_fraction_leaf_rfc:
    :param max_features_rfc:
    :param max_leaf_nodes_rfc:
    :param min_impurity_decrease_rfc:
    :param bootstrap_rfc:
    :param oob_score_rfc:
    :param random_state_rfc:
    :param ccp_alpha_rfc:
    :param df_columns_dropdown_label_rfc:
    :return:
    """

    X = pd.get_dummies(df.drop(df_columns_dropdown_label_rfc, axis=1), drop_first=True)
    y = df[df_columns_dropdown_label_rfc]
    model = RandomForestClassifier(n_estimators=n_estimators_rfc, criterion=criterion_rfc, max_depth=max_depth_rfc,
                                   min_samples_split=min_samples_split_rfc, min_samples_leaf=min_samples_leaf_rfc,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf_rfc, max_features=max_features_rfc,
                                   max_leaf_nodes=max_leaf_nodes_rfc, min_impurity_decrease=min_impurity_decrease_rfc,
                                   bootstrap=bootstrap_rfc, oob_score=oob_score_rfc, random_state=random_state_rfc,
                                   ccp_alpha=ccp_alpha_rfc)
    model.fit(X, y)

    data = model.feature_importances_
    data = data.round(3)

    df_feature_trained = pd.DataFrame(index=X.columns, data=data).reset_index()
    df_feature_trained.columns = ['Feature Name', 'Feature Importance']
    df_feature_trained = df_feature_trained.sort_values(by='Feature Importance', ascending=False)

    return model, df_feature_trained


def run_kmeans_cluster(df, n_clusters_kmc, init_kmc, n_init_kmc, max_iter_kmc, tol_kmc, random_state_kmc, copy_x_kmc,
                       algorithm_kmc):
    """
    :param df:
    :param n_clusters_kmc:
    :param init_kmc:
    :param n_init_kmc:
    :param max_iter_kmc:
    :param tol_kmc:
    :param random_state_kmc:
    :param copy_x_kmc:
    :param algorithm_kmc:
    :return:
    """
    X = pd.get_dummies(df, drop_first=True)
    model = KMeans(n_clusters=n_clusters_kmc, init=init_kmc, n_init=n_init_kmc, max_iter=max_iter_kmc, tol=tol_kmc,
                   random_state=random_state_kmc, copy_x=copy_x_kmc, algorithm=algorithm_kmc)
    model = model.fit(X)

    dummy_features_df = X[:1]
    dummy_features_df_columns = list(X.columns)

    label = model.fit_predict(X)

    col_x = df[label == 0].columns[0]
    col_y = df[label == 0].columns[-1]
    fig = go.Figure()
    for i in set(label):
        fig.add_scatter(x=df[label == i][col_x], y=df[label == i][col_y], mode="markers", name=f'Cluster {i}')

    return fig, model, dummy_features_df, dummy_features_df_columns


def logistic_regression(df, penalty_lr, dual_lr, tol_lr, c_lr, fit_intercept_lr, intercept_scaling_lr, random_state_lr,
                        solver_lr, max_iter_lr, multi_class_lr, l1_ratio_lr, df_columns_dropdown_label):
    """
    :param df:
    :param penalty_lr:
    :param dual_lr:
    :param tol_lr:
    :param c_lr:
    :param fit_intercept_lr:
    :param intercept_scaling_lr:
    :param random_state_lr:
    :param solver_lr:
    :param max_iter_lr:
    :param multi_class_lr:
    :param l1_ratio_lr:
    :param df_columns_dropdown_label:
    :return:
    """

    df = df.dropna()
    X = pd.get_dummies(df.drop(df_columns_dropdown_label, axis=1), drop_first=True)
    y = df[df_columns_dropdown_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    model = LogisticRegression(penalty=penalty_lr, dual=dual_lr, tol=tol_lr, C=c_lr, fit_intercept=fit_intercept_lr,
                               intercept_scaling=intercept_scaling_lr, random_state=random_state_lr, solver=solver_lr,
                               max_iter=max_iter_lr, multi_class=multi_class_lr, l1_ratio=l1_ratio_lr)
    model.fit(X_train, y_train)
    base_pred = model.predict(X_test)

    model_accuracy_score = accuracy_score(y_true=y_test, y_pred=base_pred)
    model_accuracy_score = round(model_accuracy_score, 4)

    cm = confusion_matrix(y_test, base_pred)
    df_columns_dropdown_label = list(df[df_columns_dropdown_label].unique())
    fig = ff_plot_confusion_matrix(cm, df_columns_dropdown_label, df_columns_dropdown_label)

    # Here Feature Importance Matrix is created manually
    data = model.coef_[0]  # [0] because it is array inside array
    data = data.round(3)

    df_feature = pd.DataFrame(index=X.columns, data=data).reset_index()
    df_feature.columns = ['Feature Name', 'Feature Importance']
    df_feature["importance"] = pow(math.e, data)  # This step can be skipped
    df_feature = df_feature.sort_values(by='Feature Importance', ascending=False)

    dummy_features_df = X[:1]
    dummy_features_df_columns = list(X.columns)

    return fig, df_feature, dummy_features_df, dummy_features_df_columns, model_accuracy_score


def train_logistic_regression(df, penalty_lr, dual_lr, tol_lr, c_lr, fit_intercept_lr, intercept_scaling_lr,
                              random_state_lr, solver_lr, max_iter_lr, multi_class_lr, l1_ratio_lr,
                              df_columns_dropdown_label):
    """
    :param df:
    :param penalty_lr:
    :param dual_lr:
    :param tol_lr:
    :param c_lr:
    :param fit_intercept_lr:
    :param intercept_scaling_lr:
    :param random_state_lr:
    :param solver_lr:
    :param max_iter_lr:
    :param multi_class_lr:
    :param l1_ratio_lr:
    :param df_columns_dropdown_label:
    :return:
    """

    df = df.dropna()
    X = pd.get_dummies(df.drop(df_columns_dropdown_label, axis=1), drop_first=True)
    y = df[df_columns_dropdown_label]
    model = LogisticRegression(penalty=penalty_lr, dual=dual_lr, tol=tol_lr, C=c_lr, fit_intercept=fit_intercept_lr,
                               intercept_scaling=intercept_scaling_lr, random_state=random_state_lr, solver=solver_lr,
                               max_iter=max_iter_lr, multi_class=multi_class_lr, l1_ratio=l1_ratio_lr)
    model.fit(X, y)

    # Here Feature Importance Matrix is created manually
    data = model.coef_[0]  # [0] because it is array inside array
    data = data.round(3)

    df_feature_trained = pd.DataFrame(index=X.columns, data=data).reset_index()
    df_feature_trained.columns = ['Feature Name', 'Feature Importance']
    df_feature_trained["importance"] = pow(math.e, data)  # This step can be skipped
    df_feature_trained = df_feature_trained.sort_values(by='Feature Importance', ascending=False)

    return model, df_feature_trained


def model_prediction(model, input_features):
    prediction = model.predict([input_features])
    return prediction
