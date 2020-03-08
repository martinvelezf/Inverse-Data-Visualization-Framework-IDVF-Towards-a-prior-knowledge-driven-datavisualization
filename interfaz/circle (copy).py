import matplotlib.pyplot as plt
import ipywidgets as widgets
import pandas as pd


def plot_w(dataframe,ticker):

    I = data_df.columns == ticker

    print(data_df.loc[:, I].head(10))
    #Code fails in the line below. 
    df = dataframe.loc[:, I].plot(x=dataframe.index, y=dataframe[ticker], style=['-bo'], figsize=(8, 5), fontsize=11, legend='False')

    plt.plot(df[ticker], label = ticker)
    #plt.plot(df["AMZN"], label = "Amazon")

    plt.legend(loc = "upper center", shadow = True, fontsize = "small", facecolor = "black")

    plt.show()

widgets.interact(plot_w,
    dataframe = widgets.fixed(data_df),
    ticker = widgets.Dropdown(
            options=data_df.columns,
            value='ATVI',
            description='Company ticker:',
            disabled=False,
        )
)
dataframe.loc[:, ['Date',ticker]].plot(x='Date', y=ticker, style=['-bo'], figsize=(8, 5), fontsize=11, legend='False')

