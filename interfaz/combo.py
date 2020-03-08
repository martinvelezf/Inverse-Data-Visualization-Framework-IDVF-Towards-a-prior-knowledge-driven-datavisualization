import pandas as pd
import numpy as np
import matplotlib as mpl # optional (here)
import matplotlib.pyplot as plt
import seaborn as sns # Optional, will only affect the color of bars and the grid

from ipywidgets import widgets, interactive

from io import StringIO

testdata=StringIO("""Year,Sex,Area,Count
2015,W,Dhaka,6
2015,M,Dhaka,3
2015,W,Khulna,1
2015,M,Khulna,8
2014,M,Dhaka,13
2014,W,Dhaka,20
2014,M,Khulna,9
2014,W,Khulna,6
2013,W,Dhaka,11
2013,M,Dhaka,2
2013,W,Khulna,8
2013,M,Khulna,5
2012,M,Dhaka,12
2012,W,Dhaka,4
2012,W,Khulna,7
2012,M,Khulna,1
    """)
def plotit(area, start_year, end_year):
    """
    Filters and plot the dataframe as a stacked bar chart of count of Male versus Women

    Args:
    -----
        * area (str): the area to filter on, or "All" to display all Areas

        * start_year, end_year (int, as float): the start and ends years, inclusive

        Note: the dataframe to plot is globally defined here as `df`

    Returns:
    --------
        A matplotlib stacked bar chart

    """
    if start_year > end_year:
        print("You must select a start year that is prior to end year")
    else:
        df2 = df.copy()
        if area != 'All':
            df2 = df2[df2.Area == area]

        # Filter between min and max years (inclusive)
        df2 = df2[(df2.Year >= start_year) & (df2.Year <= end_year)]


        # Plot it (only if there's data to plot)
        if len(df2) > 0:
            df2.groupby(['Year', 'Sex']).sum()['Count'].unstack().plot(kind='bar', stacked=True, title="Area = {}".format(area))
            plt.show();
        else:
            print("No data to show for current selection")
interactive(plotit, area=area, start_year=start_year, end_year=end_year)
