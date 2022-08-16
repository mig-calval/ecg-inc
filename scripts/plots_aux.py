import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# This is the order of the 12 signals for each file.
signals = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2','V3','V4','V5', 'V6']

# Set standarized color for the urgency level
urgency_colors = {1:'#EC0D0D', 2:'#EC7D0D', 3:'#E9EC0D', 4:'#0DEC84'}

# Utility function
def make_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def get_metadata(info, labels):

    """
    Get the metadata of the provided ecg. This code might look messy, but that's the way
    the .hea files' info is retrieved.

    Inputs:
        info - tuple, this tuple is the one obtained through wfdb.rdsamp() when we read a
            .hea file. The first argument contains the waves, while the second one contains
            metadata such as gender, age, diagnosis, frequency sample, length, units (mV), etc.
        labels - pd.DataFrame, contains the mapping of SNOMED CT Codes to a description
            of the diagnosis.
    """

    metadata = dict()

    metadata['dimension'] = info[0].shape

    try:
        metadata['fs'] = info[1]['fs']
    except IndexError:
        metadata['fs'] = "No Frequency Sample"

    try:        
        metadata['sig_len'] = info[1]['sig_len']
    except IndexError:
        metadata['sig_len'] = "No Signal Length"

    try:
        metadata['duration'] = info[1]['sig_len']/info[1]['fs']
    except IndexError:
        metadata['duration'] = "No Frequency Sample"

    try:
        metadata['age'] = info[1]['comments'][0][5:]
    except IndexError:
        metadata['age'] = "No Age"

    try:
        metadata['sex'] = info[1]['comments'][1][5:]
    except IndexError:  
        metadata['sex'] = "No Sex"
    
    try:
        metadata['dx'] = [labels.loc[int(dx), 'Diagnostic Description'] 
                                    for dx in info[1]['comments'][2][4:].split(',')]
    except IndexError:
        metadata['dx'] = "No Diagnoses"

    return metadata


def print_metadata(info, labels):

    """
    Print the metadata of the provided ecg.

    Inputs:
        Same as get_metadata()
    """

    metadata = get_metadata(info, labels)

    print("The dimensions of this ECG are: ", metadata['dimension'])
    # print("Frequency sample: ", metadata['fs'])
    # print("Singal length: ", metadata['sig_len'])
    print("Duration (seconds): ",  metadata['duration'])
    print("Patient's age: ", metadata['age'])
    print("Patient's sex: ",  metadata['sex'])
    print("Patient's diagnosis: ", metadata['dx'])


def plot_ecg(waves, ylim=(-6,6), figsize=(16,10), metadata=None, dims=[6, 2], show=True):

    """
    Plot the 12-lead ecg. Note this function assumes we are using the standard 12 leads in the 
    following order:
    
    signals = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2','V3','V4','V5', 'V6']
    
    Inputs:
        waves - np.array, shape=(n, 12); n is the number of simulations for each individual signal.
        ylim - float; sets the lower & upper limit for every signal.
        
    
    Possible Upgrades:
        Don't immediately assume it has 12 channels, nor the order. Instead pass both as parameters.
        The y limits by themselves are not good enough to represent a more realistic image. We could
        use some overlapping between waves.
        Maybe the patients id and the diagnostic can be added at the top.
    """
    
    fig, axs = plt.subplots(dims[0],dims[1],figsize=figsize, facecolor='w', edgecolor='k')    

    # Adjust the spacing and axs given the dimension chosen. Note that this function is
    # optimized for the [6,2] case.

    if dims == [12,1]:
        fig.subplots_adjust(hspace = -0.11)
        axs = axs.ravel()

    elif dims == [6,2]:
        fig.subplots_adjust(hspace = -0.11, wspace=0.15)
        axs = axs.T.ravel()

    else:
        print("Please enter a valid dimension, either [12,1] or [6,2]")
        return None

    # Plot the waves, setting the appropriate labels and adjusting the limits.

    for channel in range(12):
        axs[channel].plot(range(len(waves[:, channel])), waves[:, channel])
        axs[channel].set_ylabel(signals[channel], fontsize=14, rotation=0, labelpad=20)
        axs[channel].set_ylim(ylim[0], ylim[1])

    # Add additional information that will appear on the title, as well as choose to save
    # the plot on a given path.

    if isinstance(metadata, dict):
        
        metadata_ = get_metadata(metadata["info"], metadata["labels"])
        sex_corr = "    " if str(metadata_['sex']) == "Male" else "" # Add spaces if it's Male
        meta_title = "Dimensions:                " + str(metadata_['dimension']) + "\n" \
                   + "Duration (seconds):             " + str(metadata_['duration'])  + "\n" \
                   + "Age:                                        " + str(metadata_['age'])  + "\n"\
                   + "Sex:                                 " + sex_corr + str(metadata_['sex'])  + "\n\n"\
                   + "Dx: " + str(metadata_['dx'])
        plt.suptitle(meta_title, fontsize=16, y = 1.05)

        if isinstance(metadata['path'], str):
            plt.savefig(metadata['path']+'.png', bbox_inches='tight')

    if not show:
        plt.close()


def plot_various_given_code(Y_code, code, requirements, n_sample=5, show=False, verbose=False):

    """"
    Plot and save n_sample ecgs that belong to the same diagnosis (code).

    Inputs:
        Y_code
    """

    for k in range(n_sample):
        
        # This is the actual index to find the needed signals in X
        f_number = Y_code.index[k]        
        
        # Retrieve the signal and metadata
        ecg_file =  wfdb.rdsamp(requirements['path'] + requirements['hea_files'][f_number])
        
        ### Define the paths to store the plot in two folders:
        ### dx : The folder names are the SNOMED CT Codes themselves
        ### dx_desc : The folder names are the descriptions for the respective codes
        
        # dx
        ecg_path = requirements['imgs_path'] + requirements['current_db'] + "/ecg/dx/" + str(code)
        make_dir(ecg_path)        
        
        # dx_desc
        ecg_desc_path = requirements['imgs_path'] + requirements['current_db'] + "/ecg/dx_desc/" \
                                 + requirements['labels'].loc[code]['Diagnostic Description'].replace(' ', '_')
        make_dir(ecg_desc_path)
        
        ### Define the metadata, plot and store in both aforementioned folders
        
        # dx
        metadata = {"info": ecg_file, "labels": requirements['labels'],
                    "path": ecg_path + '/' + requirements['hea_files'][f_number]}
        plot_ecg(ecg_file[0], metadata=metadata, show=show)
        
        # dx_desc
        metadata = {"info": ecg_file, "labels": requirements['labels'],
                    "path": ecg_desc_path + '/' + requirements['hea_files'][f_number]}
        plot_ecg(ecg_file[0], metadata=metadata, show=False) # It suffices to plot 1

        if verbose:
            print("Register number ", f_number, " plotted")
    

def top_n_dx_plot(n, Y_df, requirements, m=0, urgency_colors=urgency_colors, title=None):

    # Retrieve the most frequent dx given the Y_df
    top_n_dx = Y_df.sum(0).sort_values(ascending=False)[m:(n+m)]
    n = len(top_n_dx)
    top_n_dx = pd.DataFrame(top_n_dx)
    top_n_dx = top_n_dx.join(requirements['labels'])[[0, 'Diagnostic Description', 'Urgency']]
    top_n_dx.columns = ['counts',  'dx', 'urgency']
    top_n_dx['color'] = top_n_dx['urgency'].apply(lambda x : urgency_colors[x])

    # We create a horizontal barplot in order to better show the dx description
    fig, ax = plt.subplots(figsize=(12,12))
    ax.barh(top_n_dx['dx'].astype(str), top_n_dx['counts'], color=top_n_dx['color'])
    plt.gca().invert_yaxis()

    # We add labels to show the urgency level color
    handles = [Line2D([0], [0], linewidth = 10, color=urgency_colors[k]) for k in range(1,5)]
    labels = ['Urgency 1', 'Urgency 2', 'Urgency 3', 'No Urgency']
    ax.legend(handles = handles, labels = labels, fontsize = 18, frameon=False, loc='lower right')

    # Get the rectangles/bars for each of the dx
    rectangles = [x for x in ax.get_children() if isinstance(x, matplotlib.patches.Rectangle)][:-1]

    # For each, add the text at the end of the bar
    for rectangle in rectangles:
        ax.text(rectangle.get_width() - top_n_dx['counts'].min()/2,
                rectangle.xy[1] + rectangle.get_height()/1.6,
                rectangle.get_width(),
                size=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)  )

    # Final adjustments
    ax.tick_params(axis='y', which='major', labelsize=12)    
    if isinstance(title, str):
        ax.set_title(title, fontsize = 18, y = 1.02)
    else:
        if m != 0:
            print("Beware if you set a value for m != 0, since it avoids the first m values, then \
                   the following title is misleading.")  
        ax.set_title(f'Top {n} most frequent Dx', fontsize = 18, y = 1.02)
        

def compare_2_ecgs(filename_1, filename_2, requirements, ylim_1 = (-5,5),  ylim_2 = (-5,5), show_unique_n_nan=True) :

    # From https://physionet.org/content/challenge-2020/1.0.2/
    example_1 = wfdb.rdsamp(filename_1)
    metadata_1 = {"info": example_1, "labels": requirements['labels'], "path":None}

    # From https://www.kaggle.com/code/bjoernjostein/physionet-challenge-2020/data?select=WFDB
    example_2 = wfdb.rdsamp(filename_2)
    metadata_2 = {"info": example_2, "labels": requirements['labels'], "path":None}

    print("File 1\n", example_1[0])
    print("\nFile 2\n", example_2[0])
    print("\nFile 1\n")
    print_metadata(example_1, requirements['labels'])
    print("\nFile 2\n")
    print_metadata(example_2, requirements['labels'])        

    if show_unique_n_nan:
        # Divide to get the constant factor
        frac = example_1[0]/example_2[0]
        print('\n\nWe can see that the constant is 0.2 (or 5, depending on the order): (Unique values of the divisions)',
            np.unique(frac), "\n")

        # Check that the nan values are caused by 0 divisions
        frac = frac.ravel()
        print('Sum of the squared denominators that caused nan: ', 
            (example_2[0].ravel()[~np.isclose(frac, 0.2)]**2).sum())

    # Plot with different ylim to adjust for the factor
    
    plot_ecg(example_1[0], ylim=ylim_1, figsize=(16,6))
    plot_ecg(example_2[0], ylim=ylim_2, figsize=(16,6))

def plot_model_history(fitted_model):

    fig, axs = plt.subplots(1,2, figsize = (16, 5))

    n_epochs = len(fitted_model.history['loss'])

    # Loss
    loss_df = pd.DataFrame({'train' : fitted_model.history['loss'],
                            'val' : fitted_model.history['val_loss']})
    axs[0].plot(loss_df)
    axs[0].set_xlabel('epoch')
    axs[0].set_title('Loss', fontsize = 18)
    axs[0].legend(['train', 'val'], frameon=False, fontsize=12)

    # Accuracy
    acc_df = pd.DataFrame({'train' : fitted_model.history['accuracy'],
                        'val' : fitted_model.history['val_accuracy']})
    axs[1].plot(acc_df)
    axs[1].set_xlabel('epoch')
    axs[1].set_title('Accuracy', fontsize = 18)
    axs[1].legend(['train', 'val'], frameon=False, fontsize=12)