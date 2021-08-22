"""Módulo com funções para plots de série temporal."""

import numpy as np 
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose 
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def set_default_config(fig, linecolor='rgba(100, 100, 100, 0)', 
                       tickcolor='rgba(100, 100, 100, 0.8)', showgrid=False, 
                       paper_color='rgba(0, 0, 0, 0)', 
                       plot_color='rgb(243, 243, 243)', hovermode='x', 
                       subplots=[]):
    '''Sets default configuration for plots

    Parameters
    ----------
    fig : Plotly Figure object
        Figure to set configuration on
    linecolor : str, optional
        Rgba code of color for axes' lines (default is 'rgba(100, 100, 100, 0)')
    tickcolor : str, optional
        Rgba code of color for axes' ticks (default is 'rgba(100, 100, 100, 0.8)')
    showgrid : bool, optional
        Whether or not to show axes' gridlines (default is False)
    paper_color : str, optional
        Rgba code of color for plot's paper background (default is 'rgba(0, 0, 0, 0)')
    plot_color : str, optional
        Rgba code of color for plot's inside background (default is 'rgb(243, 243, 243)')
    hovermode : str, optional
        Hovermode to be set on graph (default is 'x')
    subplots : list, optional
        List of integers identifying the number of rows and columns for subplots with [int, int] format (default is [])

    Returns
    -------
    fig
        Plotly Figure object with default configurations applied
    '''

    if not subplots:
        fig.update_layout(
            xaxis=dict(linecolor=linecolor, showgrid=showgrid, tickfont=dict(color=tickcolor), zeroline=False),
            yaxis=dict(linecolor=linecolor, showgrid=showgrid, tickfont=dict(color=tickcolor), zeroline=False)
        )
    else:
        for row in range(1, subplots[0] + 1):
            for col in range(1, subplots[1] + 1):
                fig.update_xaxes(linecolor=linecolor, showgrid=showgrid, tickfont=dict(color=tickcolor), zeroline=False, row=row, col=col)
                fig.update_yaxes(linecolor=linecolor, showgrid=showgrid, tickfont=dict(color=tickcolor), zeroline=False, row=row, col=col)
        
    fig.update_layout(
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode=hovermode,
        separators=',.',
        plot_bgcolor=plot_color,
        paper_bgcolor=paper_color
    )
    
    return fig
# Categóricas
def plt_ts_categorical(df, col, time_period='year', size=[800, 370],
                        number_format=',.2f', qtd_format='.d'):
    '''Plot proportion of categories in observations per time_period

    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame
    col : str
        Name of the column referring to the desired variable
    time_period : str, optional
        Time period where we are calculating the proportion of the variable
        (default = 'year')
    size : list, optional
        Size of the figure (default = [800,370])
    number_format : str, optional
        Format of the numbers that will be ploted (default = '.2f')
    qtd_format : str , optional
        Rounding format (default = '.d')
    Returns
    -------
    fig
        Figure with a "time series" for each category of the variable
    '''
    df_auxiliar = df[[time_period, col]]

    categorias = df[col].value_counts()
    categorias = categorias.reset_index(drop=False)
    categorias = list(categorias['index'])

    number_of_categories = len(categorias)
    for categoria in categorias:
        df_auxiliar[categoria] = np.where(df_auxiliar[col] == categoria, 1, 0)

    table = pd.pivot_table(df_auxiliar, categorias, index=time_period,
                           aggfunc=np.sum)
    table.loc[:, 'Total'] = table.sum(axis=1)
    table = table[:].div(table.Total, axis=0)

    color_palette = list(
        sns.diverging_palette(200, 6, s=50, l=30, center='dark',
                              n=number_of_categories).as_hex()
    )

    # ! Testando?
    fig = go.Figure(layout=go.Layout(title='testando'))

    for i, categoria in enumerate(categorias):
        fig.add_trace(
            go.Scatter(
                x=table.index, y=table[categoria], name=categoria,
                line_shape='hv', line=dict(color=color_palette[i])
            )
        )

    fig = set_default_config(fig)

    return fig

# Numéricas
# ! Estamos supondo que a série terá período anual.
# ! Por isso todas as nossas análises estão em 'year'

def plt_ts_line(df, date, col, time_period = 'year', resample_method='mean', 
                number_format='.2f'):
    '''Plot the variation of a numerical variable in each time period

    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame
    date : str
        Name of the column referring to the time series dates
    col : str
        Name of the column referring to the desired variable
    time_period : str, optional
        Period of time for wich we will plot the distribution of the variable (default = 'year')
        Available periods : 'year'
    resample_method : str , optional
        Method used for grouping samples with the same time period (default is 'mean')
        Available methods are 'mean', 'sum' and 'count'.
    number_format : str, optional
        Format of the numbers that will be ploted (default = '.2f')

    Returns
    -------
    fig
        Figure with the variation of the numerical value in each time period
    '''    
    
    df = df.copy()
    
    if time_period != 'year':
        print('Time period not available.')

    if (df[col].max() - df[col].min()) > 100:
        number_format=',.0f'

    if resample_method == 'mean':
        df = df.resample('M').mean()
    elif resample_method == 'sum':
        df = df.resample('M').sum()
    elif resample_method == 'count':
        df = df.resample('M').count()
    else:
        print('Invalid resample method.')
        return -1

    df['year'] = df.index.year
    years = np.unique(df['year'].values)
    number_of_years = len(years)
    color_palette = list(sns.diverging_palette(200, 6, s=50, l=30, center = 'dark', n=number_of_years).as_hex())
    fig = go.Figure()

    for i, year in enumerate(years):
        fig.add_trace(
            go.Scatter(
                x = df.loc[df.year == year].index.month,
                y = df.loc[df.year == year, col],
                name = str(year),
                mode = 'lines',
                line_color=color_palette[i]
            )
        )

    fig = set_default_config(fig)
    fig.update_xaxes(hoverformat='d')
    fig.update_yaxes(hoverformat=number_format)
    fig.update_layout(title=col)

    return fig

def plt_ts_boxplot(df, col, col_period, number_format=',.2f', 
                   qtd_format='.d', outlier_color='rgba(233, 75, 59, 1)'):
    '''Plot BoxPlot of the distribution of a numerical variable in each timestamp

    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame
    date : str
        Name of the column referring to the time series dates
    col : str
        Name of the column referring to the desired variable
    time_period : str , optional
        Period of time for which we will plot the distribution of the variable (default = 'year')
        Available periods are 'year' or 'months'
    number_format : str, optional
        Format of the numbers that will be ploted (default = '.2f')
    qtd_format : str , optional
        Rounding format (default = '.d') 
    outlier_color : str , optional
        Rgba code of extra color to be used in the graph (default = 'rgba(233, 75, 59, 1)')
    
    Returns
    -------
    fig
        BoxPlot of distributions in each time period
    '''

    data_numeric = df[col]
    if (data_numeric.max() - data_numeric.min()) > 100:
        number_format=',.0f'

    ncols =  df[col_period].nunique()
    color_palette = list(sns.diverging_palette(200, 6, s=50, l=30, center = 'dark', n=ncols).as_hex())
    
    fig = go.Figure(data=[go.Box(
        x=df.loc[df[col_period] == tp, col_period],
        y=df.loc[df[col_period] == tp, col],
        boxmean=True,
        marker=dict(color=color_palette[i], outliercolor=outlier_color),
        name=str(tp)
    ) for i, tp in enumerate(df[col_period].unique())])

    fig = set_default_config(fig)
    
    fig.update_layout(title=col, showlegend=False)
    fig.update_xaxes(hoverformat='d', dtick='Y1')
    fig.update_yaxes(hoverformat=number_format)

    return fig
    
def plt_ts_decompose(df, date, col, size=[800, 370], number_format=',.2f', 
                     qtd_format='.d', color='rgba(20, 36, 44, 0.7)', 
                     outlier_color='rgba(233, 75, 59, 1)'):
    '''Plot observations, sazonality, trend and residuals
             of numerical variable in time serie analysis

    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame
    date : str
        Name of the column referring to the time series dates
    col : str
        Name of the column referring to the desired variable
    size : list, optional
        Size of the figure (default = [800,370])
    number_format : str, optional
        Format of the numbers that will be ploted (default = '.2f')
    qtd_format : str , optional
        Rounding format (default = '.d') 
    color : str , optional
        Rgba code of color to be used in the graph (default = 'rgba(20, 36, 44, 0.7)')
    missing_color : str , optional
        Rgba code of extra color to be used in the graph (default = 'rgba(233, 75, 59, 1)')

    Returns
    -------
    fig
        Figure with 4 subplots, rangeselector and rangeslider   
    '''
    
    fig = None
    data_numeric = df[col]

    if (data_numeric.max() - data_numeric.min()) > 100:
        number_format=',.0f'

    def adding_trace(fig, data, i , color=color, number_format=number_format):
        fig.add_trace(
            go.Scatter(  
                x=data.index,
                y=data,
                mode='lines',
                marker=dict(color=color),
                hovertemplate='%{y:,.2f}<extra></extra>'
            ),
            row=i+1, col=1
        )
        fig.update_yaxes(hoverformat=number_format, row=i+1, col=1)
        return fig

    type_of_serie = ['additive', 'multiplicative']

    for type_serie in type_of_serie:

        try:
            decomposition = seasonal_decompose(data_numeric, model=type_serie,
                                               extrapolate_trend='freq')
        except ValueError:
            # print(f"{type_serie} decomposition doesn't support these values.")
            continue

        ys = [data_numeric, decomposition.trend,
              decomposition.seasonal, decomposition.resid]

        fig = make_subplots(rows=4,
                            cols=1,
                            row_heights=[0.4, 0.2, 0.2, 0.2],
                            shared_xaxes='all',
                            vertical_spacing=0.08,
                            subplot_titles=['observacoes', 'tendencia',
                                            'sazonalidade', 'residuo'])
        fig = set_default_config(fig, subplots=[4, 1])

        for i, y in enumerate(ys):
            fig = adding_trace(fig, y, i)

        fig.update_layout(
            title=col + ' with ' + type_serie + ' decomposition.',
            title_font_color=color,
            font_color=color,
            width=800,
            height=800,
            showlegend=False
        )

        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.05),
            row=4,
            col=1
        )

        fig.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month",
                         stepmode="backward"),
                    dict(count=6, label="6m", step="month",
                         stepmode="backward"),
                    dict(count=1, label="YTD", step="year",
                         stepmode="todate"),
                    dict(count=1, label="1y", step="year",
                         stepmode="backward"),
                    dict(step="all")
                ]
            ),
            row=1, col=1
        )

    if not fig:
        fig = go.Figure()
        fig = set_default_config(fig)
        fig.add_trace(
            go.Scatter(
                x=data_numeric.index,
                y=data_numeric,
                mode='lines',
                marker=dict(color=color))
            )
        fig.update_yaxes(hoverformat=number_format)
        return fig

    else:
        return fig

def plt_ts_fillnan(df, date, col, freq, title=None):
    '''Plot timeseries with missing points filled with backfill and
    linear interpolation.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame with info
    date : str
        Name of the column referring to the time series dates
    col : str
        Name of the column referring to the desired variable
    freq : str
        Frequency in which the time series is composed
    title : str, optional
        Plot's title (default is the desired variable's name)

    Returns
    -------
    fig
        Plotly Figure object with timeseries filled using backfill and
        linear interpolation
    DataFRame
        df_interpol with the interpolation on missing values
    '''

    def add_line(fig, df, color, row, column, lgroup, name, showlegend=True):
        fig = fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                marker=dict(color=color),
                legendgroup=lgroup,
                name=name,
                showlegend=showlegend
            ),
            row=row,
            col=column
        )

    if not title:
        title = col

    df_missing = df.copy()
    df_missing.set_index(date, inplace=True)
    df_missing = df_missing.asfreq(freq)  # Input nos missing values da série

    df_missing = df_missing.filter(items=[col])
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes='all',
                        vertical_spacing=0.15,
                        subplot_titles=['Backward fill',
                                        'Interpolação linear'])
    fig = set_default_config(fig, subplots=[2, 1])

    df_backward = df_missing.fillna(method='bfill')
    df_missing[col] = df_missing[col].astype(float)
    df_interpol = df_missing.interpolate()

    add_line(fig, df_backward, color='rgba(233, 75, 59, 1)',
             row=1, column=1, lgroup='fill', name='Fill')
    add_line(fig, df_missing, color='rgba(20, 36, 44, 1)',
             row=1, column=1, lgroup='orig', name='Original')
    add_line(fig, df_interpol, color='rgba(233, 75, 59, 1)',
             row=2, column=1, lgroup='fill', name='Fill', showlegend=False)
    add_line(fig, df_missing, color='rgba(20, 36, 44, 1)',
             row=2, column=1, lgroup='orig', name='Original', showlegend=False)

    fig.update_layout(hovermode='closest', title=title)
    fig.update_traces(hovertemplate='<b>%{x}:</b> %{y}')

    fig.update_xaxes(
        row=2, col=1,
        rangeslider=dict(visible=True, thickness=0.05)
    )

    fig.update_xaxes(
        row=1, col=1,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    df_interpol = df_interpol.interpolate(method='bfill')
    return fig, df_interpol
