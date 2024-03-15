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

