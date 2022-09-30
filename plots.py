import plotly.graph_objects as go
import igraph
from igraph import Graph, EdgeSeq
from sklearn.tree import plot_tree


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


