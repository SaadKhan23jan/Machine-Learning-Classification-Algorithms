from igraph import Graph, EdgeSeq
from sklearn.tree import plot_tree
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


def get_plotly_data(E, coords):
    # E is the list of tuples representing the graph edges
    # coords is the list of node coordinates
    N = len(coords)
    Xnodes = [coords[k][0] for k in range(N)]  # x-coordinates of nodes
    Ynodes = [coords[k][1] for k in range(N)]  # y-coordnates of nodes

    Xedges = []
    Yedges = []
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def dt_plotly(model):

    coords = []
    texts = []
    for item in plot_tree(model):
        coords.append(list(item.get_position()))
        texts.append(item.get_text());

    G = Graph.Tree(len(coords), 2)  # 2 stands for children number
    E = [e.tuple for e in G.es]
    Xnodes, Ynodes, Xedges, Yedges = get_plotly_data(E, coords)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xnodes, y=Ynodes,
                             mode="markers+text", marker_size=15, text=texts, textposition='top center'))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig.add_trace(go.Scatter(x=Xedges,
                             y=Yedges,
                             mode='lines',
                             line_color='blue',
                             line_width=5,
                             hoverinfo='none'
                             ))
    fig.update_layout(title="Decision Tree Plot")

    return fig




def dt_heatmap_graph(df, model):

    labels = [''] * model.tree_.node_count
    parents = [''] * model.tree_.node_count
    labels[0] = 'root'
    for i, (f, t, l, r) in enumerate(zip(
            model.tree_.feature,
            model.tree_.threshold,
            model.tree_.children_left,
            model.tree_.children_right,
    )):
        if l != r:
            labels[l] = f'{df.columns} <= {t:g}'
            labels[r] = f'{df.columns} > {t:g}'
            parents[l] = parents[r] = labels[i]

    fig = go.Figure(go.Treemap(
        branchvalues='total',
        labels=labels,
        parents=parents,
        values=model.tree_.n_node_samples,
        textinfo='label+value+percent root',
        marker=dict(colors=model.tree_.impurity),
        customdata=list(map(str, model.tree_.value)),
        hovertemplate='''
    <b>%{label}</b><br>
    impurity: %{color}<br>
    samples: %{value} (%{percentRoot:%.2f})<br>
    value: %{customdata}'''
    ))

    return fig


def heatmap_plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
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


def eda_graph_plot(df, x_axis_features=None, y_axis_features=None, graph_type=None, color=None, symbol=None, size=None,
                   hover_name=None, hover_data=None, custom_data=None, text=None, facet_row=None, facet_col=None,
                   orientation=None, width=None, height=None, sort_by=None, latitude=None, longitude=None,
                   locations=None, locationmode=None):
    """
    :param df: The Data Frame from the app.py
    :param x_axis_features: The Feature for x-axis
    :param y_axis_features: The Feature for y-axis
    :param graph_type: The Graph type which is selected
    :param color:
    :param symbol:
    :param size:
    :param hover_name:
    :param hover_data:
    :param custom_data:
    :param text:
    :param facet_row:
    :param facet_col:
    :param orientation:
    :param width:
    :param height:
    :return: return on Graph type as fig
    :param sort_by: it sorts the dataframe by that column (by Ascending only)
    :param locationmode:
    :param locations:
    :param longitude:
    :param latitude:
    """

    if sort_by != None:
        df = df.sort_values(sort_by)

    df = df.dropna()
    if graph_type == "Scatter":
        fig = px.scatter(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, symbol=symbol, size=size,
                         hover_name=hover_name, hover_data=hover_data, custom_data=custom_data, text=text,
                         facet_row=facet_row, facet_col=facet_col, orientation=orientation)
    elif graph_type == "Line":
        fig = px.line(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, symbol=symbol,
                      hover_name=hover_name, hover_data=hover_data, custom_data=custom_data, text=text,
                      facet_row=facet_row, facet_col=facet_col, orientation=orientation)
    elif graph_type == 'Area':
        fig = px.area(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, symbol=symbol,
                      hover_name=hover_name, hover_data=hover_data, custom_data=custom_data, text=text,
                      facet_row=facet_row, facet_col=facet_col, orientation=orientation)
    elif graph_type == 'Bar':
        fig = px.bar(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                     hover_data=hover_data, custom_data=custom_data, text=text, facet_row=facet_row,
                     facet_col=facet_col, orientation=orientation)
    elif graph_type == 'Funnel':
        fig = px.funnel(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                        hover_data=hover_data, custom_data=custom_data, text=text, facet_row=facet_row,
                        facet_col=facet_col, orientation=orientation)
    elif graph_type == 'Timeline':
        fig = px.timeline(data_frame=df, x_start=x_axis_features, x_end=y_axis_features, color=color,
                          hover_name=hover_name, hover_data=hover_data, custom_data=custom_data, text=text,
                          facet_row=facet_row, facet_col=facet_col)
    elif graph_type == 'Pie':
        fig = px.pie(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values, color=color,
                     hover_name=hover_name, hover_data=hover_data, custom_data=custom_data)
    elif graph_type == 'Subburst':
        fig = px.sunburst(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values, color=color,
                          hover_name=hover_name, hover_data=hover_data, custom_data=custom_data)
    elif graph_type == 'Treemap':
        fig = px.treemap(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values, color=color,
                         hover_name=hover_name, hover_data=hover_data, custom_data=custom_data)
    elif graph_type == "Icicle":
        fig = px.icicle(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values, color=color,
                        hover_name=hover_name, hover_data=hover_data, custom_data=custom_data)
    elif graph_type == "Funnel Area":
        fig = px.funnel_area(data_frame=df, names=df[x_axis_features], values=df[x_axis_features].values, color=color,
                             hover_name=hover_name, hover_data=hover_data, custom_data=custom_data)
    elif graph_type == "Histogram":
        fig = px.histogram(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                           hover_data=hover_data, facet_row=facet_row, facet_col=facet_col, orientation=orientation)
    elif graph_type == "Box":
        fig = px.box(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                     hover_data=hover_data, custom_data=custom_data, facet_row=facet_row, facet_col=facet_col,
                     orientation=orientation)
    elif graph_type == "Violin":
        fig = px.violin(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                        hover_data=hover_data, custom_data=custom_data, facet_row=facet_row, facet_col=facet_col,
                        orientation=orientation)
    elif graph_type == "Strip":
        fig = px.strip(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                       hover_data=hover_data, custom_data=custom_data, facet_row=facet_row, facet_col=facet_col,
                       orientation=orientation)
    elif graph_type == "ECDF":
        fig = px.ecdf(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, symbol=symbol,
                      hover_name=hover_name, hover_data=hover_data, text=text, facet_row=facet_row, facet_col=facet_col,
                      orientation=orientation)
    elif graph_type == "Violin":
        fig = px.violin(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                        hover_data=hover_data, custom_data=custom_data, facet_row=facet_row, facet_col=facet_col,
                        orientation=orientation)
    elif graph_type == "Density Heatmap":
        fig = px.density_heatmap(data_frame=df, x=x_axis_features, y=y_axis_features, hover_name=hover_name,
                                 hover_data=hover_data, facet_row=facet_row, facet_col=facet_col,
                                 orientation=orientation)
    elif graph_type == "Density Contour":
        fig = px.density_contour(data_frame=df, x=x_axis_features, y=y_axis_features, color=color,
                                 hover_name=hover_name, hover_data=hover_data, facet_row=facet_row, facet_col=facet_col,
                                 orientation=orientation)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif graph_type == "Scatter Geo":
        fig = px.scatter_geo(data_frame=df, lat=latitude, lon=longitude, locations=locations, locationmode=locationmode)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    else:
        fig = px.histogram(data_frame=df, x=x_axis_features, y=y_axis_features, color=color, hover_name=hover_name,
                           hover_data=hover_data, facet_row=facet_row, facet_col=facet_col, orientation=orientation)

    fig.update_layout(width=width, height=height)

    return fig