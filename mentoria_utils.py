
import numpy as np 
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose 
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

'''
def plots_agregados(df, ):
    
    n = 0
    fig, ax = plt.subplots(1,3, figsize = (24,4))
    for ano in df['arrival_date_w'].unique():
        dataset = df[df['arrival_date_year']==ano].copy()
        #plt.figure(figsize = (12,3))
        graph = sns.countplot(x='arrival_date_month',data = dataset, order = list(calendar.month_name)[1:], ax=ax[n])
        graph.set_xticklabels(graph.get_xticklabels(), rotation = 30)
        graph.set_title(ano)
        #graph.set_xlabel('meses')
        graph.set_ylabel('qtd de bookings')
        n = n + 1
'''

def boxplot_numerical_agg(df, target_col, period_col, category_order = None):

    #target_col, period_col = 'arrival_weekday', 'children'
     
    fig = go.Figure(data = [go.Box(
            x = df.loc[df[period_col] == tp, period_col],
            y = df.loc[df[period_col] == tp, target_col],
            boxmean = True,
            #order = list(calendar.day_name),
            #marker=dict(color=color_palette[i], outliercolor=outlier_color),
            name = str(tp)
        ) for indice, tp in enumerate(aux[period_col].unique())])

    if category_order is not None:
        fig.update_xaxes(categoryorder='array', categoryarray = category_order)

    fig.show()