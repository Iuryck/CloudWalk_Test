import requests
import pandas as pd
import numpy as np
import duckdb
import json
import dash
from dash import html, Dash, dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


feed=[]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(id='title'),
    dcc.Graph(id='livestream'),
    dcc.Graph(id='pie'),
    dcc.Interval(id='interval', interval=500)
])


def data_gen():
    df = pd.read_csv('case_CloudWalk\\transactions_2.csv')

    df = duckdb.query("""
        SELECT time, coalesce(approved,0) as approved, coalesce(denied,0) as denied, coalesce(failed,0) as failed, coalesce((backend_reversed + reversed),0) as reversed, coalesce(refunded,0) as refunded, coalesce(processing,0) as processing
        FROM (                      
            pivot df
            on status
            using sum(count)
        ) as df
        ORDER BY time
        """).df()
    
    time_list = [c for c in df.time.unique()]
    i=0
    while True:
        time = time_list[i]
        i+=1
        yield df.loc[df.time==time].to_dict('records')[0]
        
        if time == '23h 59': i=0
            
            


gen = data_gen()





@app.callback( 
        Output('livestream','figure'),
        Output('title','children'),
        Output('pie','figure'),
        Input('interval','n_intervals')
        
          )
def update_figure(n_intervals):

    
    data = next(gen)    
        
    r = requests.post('http://127.0.0.1:5000/send_transactions', json=json.dumps(data))
    
    data = pd.DataFrame.from_records(r.json())
    data.time = pd.to_datetime(data['time'],format= '%Hh %M' ).dt.time
    data.anomaly_failed = data.anomaly_failed.astype(int)
    data.anomaly_denied = data.anomaly_denied.astype(int)
    data.anomaly_reversed = data.anomaly_reversed.astype(int)

    feed.append(data)    
    

    if len(feed) > 120:
        live_data=pd.concat(feed[-120:])
    else:
        live_data=pd.concat(feed)


    

    fig = make_subplots(specs=[[{"secondary_y": True}]])


  
    
    fig.add_trace(go.Scatter(x=live_data.time, y=live_data.approved, mode='lines', name='Approved', line_color='rgb(0, 188, 69)'))
    fig.add_trace(go.Scatter(x=live_data.time, y=live_data.denied, mode='lines', name='Denied', line_color='rgb(242, 8, 0)'))
    fig.add_trace(go.Scatter(x=live_data.time, y=live_data.reversed, mode='lines', name='Reversed', line_color='rgb(0, 84, 230)'))
    fig.add_trace(go.Scatter(x=live_data.time, y=live_data.failed, mode='lines', name='Failed', line_color='rgb(126, 05, 208)'))

     
 
    fig.add_trace(go.Bar(x=live_data.time, y=live_data.anomaly_failed, name='Anomaly Failed', opacity=0.5), secondary_y=True)

    fig.add_trace(go.Bar(x=live_data.time, y=live_data.anomaly_reversed, name='Anomaly Reversed', opacity=0.5), secondary_y=True)

    fig.add_trace(go.Bar(x=live_data.time, y=live_data.anomaly_denied, name='Anomaly Denied', opacity=0.5), secondary_y=True)   
    
    fig.layout.yaxis2.update(showticklabels=False)

    fig.update_layout(xaxis_rangeslider_visible=False, height=400)

    if len(feed) >20:
        values = [np.sum(live_data.approved.values[-20:]),
                  np.sum(live_data.denied.values[-20:]),
                  np.sum(live_data.reversed.values[-20:]),
                  np.sum(live_data.failed.values[-20:])]
        names = ['Approved','Denied','Reversed','Failed']
    else:
        values = [np.sum(live_data.approved),
                  np.sum(live_data.denied),
                  np.sum(live_data.reversed),
                  np.sum(live_data.failed)]
        names = ['Approved','Denied','Reversed','Failed']

    pie_chart = px.pie(values=values, names=names, hole=.3)

    return fig, f'Monitoring - {data.time.values[0]}', pie_chart

if __name__ == '__main__':
   
    app.run_server(debug=True)

    

 

    

    

    
        



    


