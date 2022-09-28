import plotly.graph_objects as go

def dt_graph(df, model):

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
