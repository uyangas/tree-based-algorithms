import plotly.graph_objects as go


def display_table(temp, height=100, width=300, title=""):
    data = [['Survived_true','Died_true']]
    header = ['','Survived_pred','Died_pred']
    for i in range(len(temp)):
        data.append([temp[:,i][0], temp[:,i][1]])

    fig = go.Figure()
    fig.add_trace(go.Table(header_values=header,
                           cells_values=data,
                           domain=dict(x=[0, 1], y=[0, 1])
                           ))
    fig.update_layout(autosize=False,
                      height=height,
                      width=width,
                      title=title,
                      margin=dict(l=0, r=0, b=0, t=0)
                      )
    fig.show()
    
    
def display_pdtable(temp, height, width, title=""):
    data = []
    header = list(temp.columns)
    for col in header:
        data.append(list(temp[col]))
    
    fig = go.Figure()
    fig.add_trace(go.Table(header_values=header,
                           cells_values=data,
                           domain=dict(x=[0, 1], y=[0, 1])
                                         ))
    fig.update_layout(autosize=False,
                      height=height,
                      width=width,
                      title=title,
                      margin=dict(l=0, r=0, b=0, t=0)
                     )
    fig.show()