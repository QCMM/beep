import matplotlib.pyplot as plt
import numpy as np

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


#def rmsd_histogram(data, mol_name):
#    """
#    Create a histogram for the given data with colors indicating value magnitude.
#    Lower values are represented with green and higher values with red.
#
#    Parameters:
#    data (dict): A dictionary with labels as keys and numerical values as values.
#
#    Returns:
#    None: This function displays the histogram plot.
#    """
#    import matplotlib.pyplot as plt
#    import matplotlib.cm as cm
#    import matplotlib.colors as colors
#
#    # Extract labels and values from the data
#    labels = list(data.keys())
#    values = list(data.values())
#
#    # Create the figure and axes for the histogram
#    fig, ax = plt.subplots(figsize=(10, 6))
#
#    # Create the bars for the histogram
#    bars = ax.bar(labels, values)
#
#    # Generate a color map based on the values
#    norm = colors.Normalize(vmin=min(values), vmax=max(values))
#    cmap = cm.RdYlGn.reversed()
#
#    # Apply the colormap to the bars
#    for bar, value in zip(bars, values):
#        bar.set_color(cmap(norm(value)))
#
#    # Create a ScalarMappable with the same colormap and norm as used for the bars
#    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm.set_array([])
#
#    # Add the colorbar to the axes
#    cbar = fig.colorbar(sm, ax=ax)
#    cbar.set_label('MAE [Angstrom]', rotation=270, labelpad=15)
#
#    # Rotate the x labels for better readability
#    ax.set_xticklabels(labels, rotation=90)
#
#    # Show the plot
#    plt.title('Histogram of MAE values for Popular DFT Functionals for {mol_name}')
#    plt.tight_layout()
#    plt.savefig(f"rmsd_histogram_{mol_name}.svg")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


# Function to create histograms for each row and the mean of all rows
#def rmsd_histograms(df, cmap_style='RdYlGn'):
#    """
#    This function creates and saves a histogram for each row in the DataFrame,
#    as well as a histogram for the mean of all rows.
#
#    Parameters:
#    - df: Pandas DataFrame containing the data.
#    - cmap_style: String representing the colormap (default is 'RdYlGn').
#    """
#    # Reverse the colormap for better visuals (green for low, red for high)
#    cmap = cm.get_cmap(cmap_style).reversed()
#
#    # Normalize the color map based on the entire DataFrame's values
#    norm = colors.Normalize(vmin=df.min().min(), vmax=df.max().max())
#
#    # Plot histogram for each row
#    for index, row in df.iterrows():
#        fig, ax = plt.subplots()
#        labels = labels = ['_'.join(label.split('_')[-2:]) for label in row.index] 
#        values = row.values
#
#        # Create the bars for the histogram
#        bars = ax.bar(labels, values)
#
#        # Apply the colormap to the bars
#        for bar, value in zip(bars, values):
#            bar.set_color(cmap(norm(value)))
#
#        # Add the colorbar to the axes
#        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#        sm.set_array([])
#        cbar = fig.colorbar(sm, ax=ax)
#        cbar.set_label('Value', rotation=270, labelpad=15)
#
#        # Rotate the x labels for better readability
#        ax.set_xticklabels(labels, rotation=90)
#
#        # Set the title
#        title = 'Histogram for ' + index
#        plt.title(title)
#        plt.tight_layout()
#
#        # Save the figure
#        plt.savefig(f"{index}_histogram.png")
#        plt.close(fig)
#
#    # Plot histogram for the mean of all rows
#    mean_values = df.mean()
#    fig, ax = plt.subplots()
#    bars = ax.bar(mean_values.index, mean_values.values)
#
#    # Apply the colormap to the bars
#    for bar, value in zip(bars, mean_values.values):
#        bar.set_color(cmap(norm(value)))
#
#    # Add the colorbar to the axes
#    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm.set_array([])
#    cbar = fig.colorbar(sm, ax=ax)
#    cbar.set_label('Mean Value', rotation=270, labelpad=15)
#
#    # Rotate the x labels for better readability
#    ax.set_xticklabels(mean_values.index, rotation=90)
#
#    # Set the title
#    plt.title('Histogram of Mean Values')
#    plt.tight_layout()
#
#    # Save the figure
#    plt.savefig("mean_values_histogram.png")
#    plt.close(fig)
#
#
#   #plt.savefig(f"histogram_mean_rmsd_{mol_name}.svg")

def rmsd_histograms(df, molecule_name,  cmap_style='RdYlGn'):
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
        plt.savefig(f"{index}_histogram.svg")
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
    plt.savefig("mean_values_histogram_{molecule_name}.svg")
    plt.close(fig)





def plot_violin(df, mol_name, folder_name):

    # Filtering dataframe  to match a given structure name
    mask = df.index.astype(str).str.contains(structure_name)
    data = df[mask].T * -1
    matching_columns = data.columns.tolist()

    # Creating the violin plot
    plt.figure(figsize=(7, 5))
    plt.violinplot(data)

    # Adjusting x-tick labels
    processed_columns = [col.replace(structure_name + "_", "").replace("_", "/") for col in matching_columns]
    plt.xticks(range(1, len(matching_columns) + 1), processed_columns, rotation=45, ha='right')

    plt.ylabel('Binding Energy [kcal $\mathrm{mol^{-1}}$]', size= 12)
    plt.title(f'Violin Plots for {structure_name}')

    # Adding horizontal reference line and adjusting y-ticks if structure name is in the dictionary
    #if structure_name in ref_be:
    #    ref_value = ref_be[structure_name] * -1
    #    plt.axhline(y=ref_value, color='C3', linestyle='--', alpha=0.7)
    #    current_ticks = plt.gca().get_yticks()
    #    #plt.yticks(np.append(current_ticks, ref_value))

    # Adding mean value annotations
    for i, col in enumerate(matching_columns):
        mean_val = np.mean(data[col])
        plt.text(i + 1.1, mean_val, f'{mean_val:.1f}', horizontalalignment='left', color='k', fontsize=10)

    plt.tight_layout()
    plt.savefig(str(folder_name_) + f"violin_{mol_name}.svg")



def plot_density_panels(df, bchmk_struct, opt_lot,  mol_name, panel_width=6, panel_height=3, color='#18b6f4', transparency=0.3):
    """
    Creates a density plot for each row in the DataFrame, with each row plotted in a separate panel.

    :param df: pandas DataFrame containing the data to plot.
    :param panel_width: width of each panel.
    :param panel_height: height of each panel.
    :param color: Hex color code for the plot shading.
    :param transparency: Transparency level for the plot shading.
    """
    struct_dict = {}
    for struct in  bchmk_struct:
        df_f = df[df.index.str.contains(struct)] * -1
        struct_dict[struct] = df_f

    # Plotting
    n_rows = df_f.shape[0]
    n_col = len(struct_dict)
    fig, axes = plt.subplots(n_rows, n_col, figsize=(10, 6))  # Adjust size as needed

    i = 0
    for name, df in struct_dict.items():
        for j in range(n_rows):
            ax = axes[j, i]
            sns.kdeplot(df.iloc[j], ax=ax, fill=True, color='#18b6f4')
            ax.set_title(f'{df.index[j]}', fontsize=7)
            ax.set_xlabel('')
            ax.set_ylabel('Density')
        # Set column title for the first column or any specific column
        axes[-1, i].set_xlabel('BE error (kcal/mol)')
        i += 1

    plt.tight_layout()
    plt.savefig(f'be_error_{mol_name}.svg')


def plot_mean_errors(df, bchmk_struct, opt_lot,  mol_name):
    """
    Creates a density plot for each row in the DataFrame, with each row plotted in a separate panel.

    :param df: pandas DataFrame containing the data to plot.
    :param panel_width: width of each panel.
    :param panel_height: height of each panel.
    :param color: Hex color code for the plot shading.
    :param transparency: Transparency level for the plot shading.
    """
    #opt_lot = "PWB6K-D3BJ_def2-svp"
    opt_lot = ['CAM-B3LYP-D3BJ_def2-tzvp', "PWB6K-D3BJ_def2-svp"]

    print(df.head())

    n_rows = len(opt_lot)
    print(n_rows)
    n_col = 1
    fig, axes = plt.subplots(n_rows, n_col, figsize=(10, 15))  # Adjust size as needed

    for i, lot in enumerate(opt_lot):
        df_tmp = df[df.index.str.contains(lot)].abs()
        mean_errors = df_tmp.mean(axis=0)

        # Step 2: Find the 10 columns with the lowest mean error
        lowest_10_mean_errors = mean_errors.nsmallest(15)

        # Step 3: Plot the results
        ax = axes[i]
        ax.bar(lowest_10_mean_errors.index, lowest_10_mean_errors.values)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Mean Error')
        ax.set_title(f'Mean Errors at {lot} geometry')
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout() # Adjusts plot parameters to give some padding
    plt.savefig(f'best_be_dft_{mol_name}.svg')o

def plot_ie_vs_de(df_de, df_ie, bchmk_struct, opt_lot, mol_name, folder_name):
    # Adjustments for DataFrame index prefix already applied here
    
    opt_lot = ['CAM-B3LYP-D3BJ_def2-tzvp', "PWB6K-D3BJ_def2-svp"]
    n_rows = len(opt_lot)
    print(n_rows)
    n_col = 1
    fig, axes = plt.subplots(n_rows, n_col, figsize=(10, 15))

    for i, lot in enumerate(opt_lot):
        df_tmp_de = df_de[df_de.index.str.contains(lot)].abs()
        mean_errors_de = df_tmp_de.mean(axis=0)
        mean_errors_de.index = mean_errors_de.index.str.replace("^de-", "", regex=True)
        print(mean_errors_de.sort_values())
        
        df_tmp_ie = df_ie[df_ie.index.str.contains(lot)].abs()
        mean_errors_ie = df_tmp_ie.mean(axis=0)
        mean_errors_ie.index = mean_errors_ie.index.str.replace("^ie-", "", regex=True)
        print(mean_errors_ie.sort_values())

        distances = np.sqrt(mean_errors_de**2 + mean_errors_ie**2)
        closest_indices = distances.nsmallest(5).index
        print(distances.sort_values())
        break

        ax = axes[i]
        # Plot all points
        ax.scatter(mean_errors_de, mean_errors_ie, alpha=0.5, label='All Points')

        # Prepare different shades of green
        green_gradients = ['#004c00', '#007200', '#009900', '#00bf00', '#00e500']  # Dark to light green

        # Plot and label the 5 closest points with green gradient and add their values to the legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label='All Points',
                                     markerfacecolor='grey', markersize=10, alpha=0.5)]

        for j, idx in enumerate(closest_indices):
            point_val = (mean_errors_de[idx], mean_errors_ie[idx])
            ax.scatter(mean_errors_de[idx], mean_errors_ie[idx], color=green_gradients[j], alpha=0.7)
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             label=f'{idx}: {tuple(round(num, 2) for num in point_val)}',
                                             markerfacecolor=green_gradients[j], markersize=10, alpha=0.7))

        ax.legend(handles=legend_handles, title="Legend", loc="best")

        ax.set_xlabel('Absolute IE_RE')
        ax.set_ylabel('Absolute DE_RE')
        ax.set_ylim([0.,1.])
        ax.set_xlim([0.,1.])
        ax.set_title(f'{lot}')
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(folder_name) +  f'/ie_vs_de_dft_{mol_name}.svg')
