import seaborn as sns
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

def plot_violin_with_ref(structure_name, ref_be, df):

    # Filtering columns that match the given structure name
    matching_columns = [col for col in df.columns if col.startswith(structure_name)]

    # Multiplying values by -1
    df = df[matching_columns] * -1

    # Data for the violin plot
    data = [df[col].dropna().astype(float) for col in matching_columns]

    # Creating the violin plot
    plt.figure(figsize=(7, 5))
    plt.violinplot(data)

    # Adjusting x-tick labels
    processed_columns = [col.replace(structure_name + "_", "").replace("_", "/") for col in matching_columns]
    plt.xticks(range(1, len(matching_columns) + 1), processed_columns, rotation=45, ha='right')

    plt.ylabel('Binding Energy [kcal $\mathrm{mol^{-1}}$]', size= 12)
    #plt.title(f'Violin Plots for {structure_name}')

    # Adding horizontal reference line and adjusting y-ticks if structure name is in the dictionary
    if structure_name in ref_be:
        ref_value = ref_be[structure_name] * -1
        plt.axhline(y=ref_value, color='C3', linestyle='--', alpha=0.7)
        current_ticks = plt.gca().get_yticks()
        #plt.yticks(np.append(current_ticks, ref_value))

    # Adding mean value annotations
    for i, col in enumerate(matching_columns):
        mean_val = np.mean(df[col].dropna())
        plt.text(i + 1.1, mean_val, f'{mean_val:.1f}', horizontalalignment='left', color='k', fontsize=10)

    plt.tight_layout()
    plt.savefig(structure_name+"_violin_dft.svg")

def plot_lowest_values_with_indices(df, num):
    # Melt the dataframe to make it long-form and easier to filter and sort
    df_melted = df.reset_index().melt(id_vars='index', var_name='Method', value_name='Value')
    df_melted['Full Method'] = df_melted['Method'] + '//' + df_melted['index']
    
    # Find the 10 lowest values across all methods
    lowest_values = df_melted.nsmallest(num, 'Value')
    
    # Plot horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(lowest_values['Full Method'], lowest_values['Value'], color='skyblue')
    plt.xlabel('Values')
    plt.ylabel('Full Method')
    plt.title('10 Lowest Values in DataFrame with Indices')
    plt.gca().invert_yaxis()  # To have the lowest at the bottom of the chart
    plt.tight_layout()
    plt.savefig("bar_dft.svg")


def rmsd_histograms(df, molecule_name, plot_path, cmap_style='RdYlGn'):
    """
    This function creates and saves a histogram for each row in the DataFrame,
    as well as a histogram for the mean of all rows.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - cmap_style: String representing the colormap (default is 'RdYlGn').
    """
    # Import necessary libraries
    from matplotlib import cm, colors
    import matplotlib.pyplot as plt
    
    # Reverse the colormap for better visuals (green for low, red for high)
    cmap = cm.get_cmap(cmap_style).reversed()

    # Find global minimum and maximum values across the entire DataFrame
    global_min = df.min().min()
    global_max = df.max().max()

    # Normalize the color map based on the global min and max values
    norm = colors.Normalize(vmin=global_min, vmax=global_max)

    # Plot histogram for each row
    for index, row in df.iterrows():
        fig, ax = plt.subplots()
        # Process the labels to include only the last two items
        labels = ['_'.join(label.split('_')[-2:]) for label in row.index]
        values = row.values

        # Create the bars for the histogram
        bars = ax.bar(labels, values)

        # Apply the colormap to the bars using the global normalization
        for bar, value in zip(bars, values):
            bar.set_color(cmap(norm(value)))

        # Add the colorbar to the axes
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=15)

        # Rotate the x labels for better readability
        ax.set_xticklabels(labels, rotation=90)

        # Set the title
        title = 'Histogram for ' + index
        plt.title(title)
        plt.tight_layout()

        # Save the figure
        plt.savefig(plot_path+f"/{index}_histogram.svg")
        plt.close(fig)

    # Plot histogram for the mean of all rows
    mean_values = df.mean()
    fig, ax = plt.subplots()
    # Process the labels to include only the last two items
    mean_labels = ['_'.join(label.split('_')[-2:]) for label in mean_values.index]
    bars = ax.bar(mean_labels, mean_values.values)

    # Apply the colormap to the bars using the global normalization
    for bar, value in zip(bars, mean_values.values):
        bar.set_color(cmap(norm(value)))

    # Add the colorbar to the axes
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Mean Value', rotation=270, labelpad=15)

    # Rotate the x labels for better readability
    ax.set_xticklabels(mean_labels, rotation=90)

    # Set the title
    plt.title('Histogram of Mean RMSD Values')
    plt.tight_layout()

    # Save the figure
    plt.savefig(plot_path+f"/mean_values_histogram_{molecule_name}.svg")
    plt.close(fig)


## ENERGY BENCHMARK PLOTS

def plot_violins(df, structure_names, mol_name, folder_name, ref_df, en_type="BE"):
    n = len(structure_names)
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(7, 5 * n), sharex=True, sharey=False)

    if n == 1:
        axs = [axs]

    for i, structure_name in enumerate(structure_names):
        mask = df.index.astype(str).str.contains(structure_name)
        data = df[mask].T * -1
        matching_columns = data.columns.tolist()

        ax = axs[i]
        ax.violinplot(data)

        processed_columns = [col.replace(structure_name + "_", "").replace("_", "/") for col in matching_columns]
        ax.set_xticks(range(1, len(matching_columns) + 1))
        ax.set_xticklabels(processed_columns, rotation=45, ha='right')

        ax.set_ylabel('Binding Energy [kcal $\mathrm{mol^{-1}}$]', size=12)
        ax.set_title(f'Violin Plots for {structure_name}')

        for j, col in enumerate(matching_columns):
            mean_val = np.mean(data[col])
            ax.text(j + 1.1, mean_val, f'{mean_val:.1f}', horizontalalignment='left', color='k', fontsize=10)

        # Get the reference BE value for the current structure name and add a red line to the plot
        ref_value = ref_df.loc[structure_name, en_type] * -1
        ax.axhline(y=ref_value, color='red', linestyle='--', alpha=0.7)

        if i == n - 1:
            ax.set_xlabel('Level of Theory', size=12)

    plt.tight_layout()
    plt.savefig(str(folder_name) +  f"/violin_{mol_name}.svg")

def plot_density_panels(df, bchmk_struct, opt_lot, mol_name, folder_path_plots, panel_width=6, panel_height=3, color='#18b6f4', transparency=0.3):
    """
    Creates a density plot for each row in the DataFrame, with each row plotted in a separate panel.
    """
    struct_dict = {}
    for struct in bchmk_struct:
        df_f = df[df.index.str.contains(struct)] * -1
        if df_f.empty:
            continue
        struct_dict[struct] = df_f

    n_rows = max(df_f.shape[0] for df_f in struct_dict.values())  # Find the maximum number of rows any structure has
    n_cols = len(struct_dict)  # number of columns determined by number of structures


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), squeeze=False)

    for i, (name, df_f) in enumerate(struct_dict.items()):
        for j in range(df_f.shape[0]):  # Loop over actual number of rows in df_f
            ax = axes[j, i]
            sns.kdeplot(df_f.iloc[j], ax=ax, fill=True, color=color, alpha=transparency)
            ax.set_title(f'{df_f.index[j]}', fontsize=7)
            ax.set_xlabel('')
            ax.set_ylabel('Density')

        # Set x-axis label for the last row of the actual data, or the bottom of the column if fewer data rows
        if n_rows > 1 and j == df_f.shape[0] - 1:
            axes[j, i].set_xlabel('BE error (kcal/mol)')
        elif n_rows == 1:
            axes[0, i].set_xlabel('BE error (kcal/mol)')

    plt.tight_layout()
    plt.savefig(str(folder_path_plots) + f"/density_plots_{mol_name}.svg")

def plot_mean_errors(df, bchmk_struct, opt_lot, mol_name, folder_path_plots):
    """
    Creates a bar plot for each `opt_lot` entry, showing the 15 columns with the lowest mean absolute error (MAE).

    :param df: pandas DataFrame containing the data to plot.
    :param opt_lot: List of optimization levels to plot.
    :param mol_name: Name of the molecule for file naming.
    :param folder_path_plots: Directory path to save the plot.
    """
    n_rows = len(opt_lot)
    n_col = 1
    fig, axes = plt.subplots(n_rows, n_col, figsize=(10, 15), squeeze=False)  # Adjust size and force 2D array

    for i, lot in enumerate(opt_lot):
        df_tmp = df[df.index.str.contains(lot)].abs()
        mean_errors = df_tmp.mean(axis=0)

        # Find the 10 columns with the lowest mean error
        lowest_10_mean_errors = mean_errors.nsmallest(15)

        # Plot the results
        ax = axes[i, 0]  # Use 2D indexing
        ax.bar(lowest_10_mean_errors.index, lowest_10_mean_errors.values)
        ax.set_xlabel('Columns')
        ax.set_ylabel('MAE')
        ax.set_title(f'MAE at {lot} geometry')
        ax.set_xticks(range(len(lowest_10_mean_errors.index)))
        ax.set_xticklabels(lowest_10_mean_errors.index, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()  # Adjusts plot parameters to give some padding
    plt.savefig(str(folder_path_plots) + f"/mae_{mol_name}.svg")


def plot_ie_vs_de(df_de, df_ie, bchmk_struct, opt_lot, mol_name, folder_path_plots):
    n_rows = len(opt_lot)
    n_col = 1  # Since you have only one column
    fig, axes = plt.subplots(n_rows, n_col, figsize=(10, 15), squeeze=False)

    for i, lot in enumerate(opt_lot):
        df_tmp_de = df_de[df_de.index.str.contains(lot)].abs()
        mean_errors_de = df_tmp_de.mean(axis=0)
        mean_errors_de.index = mean_errors_de.index.str.replace("^de-", "", regex=True)

        df_tmp_ie = df_ie[df_ie.index.str.contains(lot)].abs()
        mean_errors_ie = df_tmp_ie.mean(axis=0)
        mean_errors_ie.index = mean_errors_ie.index.str.replace("^ie-", "", regex=True)

        distances = np.sqrt(mean_errors_de**2 + mean_errors_ie**2)
        closest_indices = distances.nsmallest(5).index

        ax = axes[i, 0]  # Correct indexing for a 2D axes array
        # Plot all points
        ax.scatter(mean_errors_de, mean_errors_ie, alpha=0.5, label='All Points')

        # Prepare different shades of green for the closest points
        green_gradients = ['#004c00', '#007200', '#009900', '#00bf00', '#00e500']  # Dark to light green

        # Plot and label the 5 closest points with green gradient
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label='All Points',
                                     markerfacecolor='grey', markersize=10, alpha=0.5)]

        for j, idx in enumerate(closest_indices):
            point_val = (mean_errors_de[idx], mean_errors_ie[idx])
            ax.scatter(mean_errors_de[idx], mean_errors_ie[idx], color=green_gradients[j], alpha=0.7)
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             label=f'{idx}: {tuple(round(num, 2) for num in point_val)}',
                                             markerfacecolor=green_gradients[j], markersize=10, alpha=0.7))

        ax.legend(handles=legend_handles, title="Legend", loc="best")
        ax.set_xlabel('Absolute DE_RE')
        ax.set_ylabel('Absolute IE_RE')
        ax.set_ylim([0., 1.])
        ax.set_xlim([0., 1.])
        ax.set_title(f'{lot}')
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(folder_path_plots) + f"/ie_vs_de_dft_{mol_name}.svg")

