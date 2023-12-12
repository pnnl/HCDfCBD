import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np

def make_igrad_plot_df(rank_df, raw_data_df, num_biomols = 10, id_vars = "Name", var_name = "SampleID", value_name = "Input_value", score_name = "IGrad_value"):
    """Create a long plotting df from a dataframe of scores and a dataframe of raw data

    Args:
        rank_df (pd.DataFrame): A dataframe with biomolecule identifiers (features) in the first column and scores in the second column, sorted by score descending.
        raw_data_df (pd.DataFrame): A dataframe with biomolecule identifiers (features) in the first column and raw data for each sample in the remaining columns.
        num_biomols (int, optional): The number of top scoring biomolecules to subset down to. Defaults to 10.
        id_vars (str, optional): The id var for melt() to turn the raw_data_df from wide to long. Defaults to "Name".
        var_name (str, optional): Passed to var_name in melt for both the raw_data_df and rank_df. Defaults to "SampleID".
        value_name (str, optional): Passed to value_name in melt for the raw_data_df. Defaults to "Input_value".
        score_name (str, optional): Passed to value_name for the rank_df. Defaults to "IGrad_value".
    """
    raw_data_df = raw_data_df.melt(id_vars = id_vars, var_name = var_name, value_name = value_name)
    raw_data_df = raw_data_df[raw_data_df[var_name].isin(rank_df.columns)]
    
    tmp_df = rank_df.iloc[:num_biomols,]
    mycols = tmp_df.columns[:2].to_list()
    igrad_plot_df = tmp_df.melt(id_vars = mycols, var_name = var_name, value_name = score_name)
    
    igrad_plot_df = igrad_plot_df.merge(raw_data_df, how = 'left', left_on = [mycols[0], var_name], right_on = [id_vars, var_name])
    
    return(igrad_plot_df)

def igrad_beeswarm_plot(
    plot_df, 
    ycol = "Lipids",
    xcol = "IGrad_value", 
    raw_val_name = "Input_value", 
    palette = "Spectral", 
    xlabel = "Integrated Gradient Scores",
    ylabel = "Lipids",
    title = "",
    size = 4,
    use_stripplot = False
    ):
    """Make a beeswarm plot using the output from make_igrad_plot_df()

    Args:
        plot_df (pd.DataFrame): A long-format dataframe with biomolecule identifiers, scores, and raw data values.  Probably the output of make_igrad_plot_df().
        ycol (str, optional): Column to use for y-axis values. Defaults to "Lipids".
        xcol (str, optional): Column to use for x-axis values. Defaults to "IGrad_value".
        raw_val_name (str, optional): Column to use for the raw input values. Defaults to "Input_value".
        palette (str, optional): A color palette name. Defaults to "Spectral".
        xlabel (str, optional): X label. Defaults to "Integrated Gradient Scores".
        ylabel (str, optional): Y label. Defaults to "Lipids".
        title (str, optional): Title. Defaults to "".
        size (int, optional): Size of the scatter points. Defaults to 4.
    """

    bin = np.linspace(min(plot_df[raw_val_name]),max(plot_df[raw_val_name]),60)
    plot_df['bin'] = pd.cut(plot_df[raw_val_name],bin,precision=0)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    cpal = sns.color_palette(palette, 5, desat = 1.)
    fn = sns.swarmplot if not use_stripplot else sns.stripplot
    fn(x = xcol, y = ycol, hue = "bin", edgecolor='black', linewidth = 0.5, data = plot_df, size = size, palette = cpal, legend=None)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    cmap = plt.get_cmap(palette)
    norm = plt.Normalize(plot_df[raw_val_name].min(), plot_df[raw_val_name].max())
    sm =  cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)

    plt.axvline(x=0, color = 'black', linestyle = '--', linewidth = 0.5)

    plt.show()
    