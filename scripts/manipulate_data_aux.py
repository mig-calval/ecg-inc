import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import scipy.signal
from scipy.fft import fft, ifft

from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

from loess.loess_1d import loess_1d
from statsmodels.nonparametric.smoothers_lowess import lowess



def initial_data_preproc(X, Y, X_dtype=np.float16):

    # Convert X to a np.array
    X = np.array(X, dtype=X_dtype)

    # We create a DF to better portray the metadata information
    Y_df = pd.DataFrame(columns=['age', 'sex', 'dx', 'rx', 'hx', 'sx', 'fs', 'sig_len', 'n_sig', 'units', 'sig_name'])

    # Access each component of the metadata and make an auxiliary DF
    for y in Y:

        y_aux = pd.DataFrame({'age':       [y['comments'][0][5:]],
                              'sex':       [y['comments'][1][5:]],
                              'dx':        [y['comments'][2][4:]], 
                              'rx':        [y['comments'][3][4:]], 
                              'hx':        [y['comments'][4][4:]], 
                              'sx':        [y['comments'][5][4:]],
                              'fs':        [y['fs']], 
                              'sig_len':   [y['sig_len']], 
                              'n_sig':     [y['n_sig']], 
                              'units':     [y['units']], 
                              'sig_name':  [y['sig_name']]})
        
        Y_df = pd.concat([Y_df, y_aux])
    
    Y = Y_df
    Y.reset_index(drop = True, inplace = True)

    return X, Y

def y_labelling(dx, labels):

    """
    Compute the 0-1 label matrix for a given set of diagnosis. That is we
    create a matrix in which the rows are the registers, the columns are the
    different SNOMED CT Codes, and the values are 1 if said register was diagnosed
    with a code, and 0 otherwise. 

    Note : This function can take a long time to give results. In the future, it is
    advised to rather create a new csv file that has it already computed.

    Inputs:
        dx - pd.Series, contains the diagnosis separated by a ',' for each row
        labels - list, contains all of the available (or desired) labels from
            the SNOMED CT Codes.
    """
    # Split all values of dx and turn them into integers
    Y_aux = dx.apply(lambda x : x.split(','))
    Y_aux = Y_aux.apply(lambda x : [int(v) for v in x])

    # Initialize OHE DataFrame
    Y_ohe = pd.DataFrame(columns = labels)

    # For each column and record, write 1 if there is an intersection,
    # 0 otherwise
    for k in range(len(Y_aux)):
        
        new_row = pd.DataFrame(index = [k], columns = labels)
        
        for c in labels:
            
            new_row.loc[k, c] = int(c in Y_aux[k])
            
        Y_ohe = pd.concat([Y_ohe, new_row])

    return Y_ohe

def sample_Y_by_code(Y_df, code, n_sample=5, random_state=203129):

    """
    Keep only the rows in which there is a positive diagnostic i.e. value of 1
    for the given SNOMED CT Code, and sample n_sample values to plot them.
    """
    Y_code = Y_df[code][Y_df[code].apply(lambda x : True if x > 0 else False)]
    Y_code = Y_code.sample(n_sample, random_state = random_state)

    return Y_code

def dx_by_urgency(urgency_level, requirements):

    """
    Filter the diagnoses by their urgency level, and by whether or not the current db
    actually contains registers of those.
    """

    # Filter given the urgenvy level
    urgency = requirements['labels'][requirements['labels']['Urgency'] == urgency_level]

    # Filter given the db at hand
    urgency = urgency[urgency[requirements['current_db']] != 0]

    # Keep important columns
    urgency = urgency[['Diagnostic Description', 'Abbreviation', requirements['current_db'], 'Total', 'Kind']]

    return urgency

    
def dx_by_kind(kind_level, requirements):

    """
    Filter the diagnoses by their kind (Most vs Less Frequent), and by whether or not the current db
    actually contains registers of those.
    """

    # Filter given the urgenvy level
    kind = requirements['labels'][requirements['labels']['Kind'] == kind_level]

    # Filter given the db at hand
    kind = kind[kind[requirements['current_db']] != 0]

    # Keep important columns
    kind = kind[['Diagnostic Description', 'Abbreviation', requirements['current_db'], 'Total', 'Urgency']]

    return kind

def calculating_class_weights(y_true, linear_factor=2, exp_factor=None):

    """

    """

    if exp_factor is None:
        weights = 1 / (y_true.sum(0) / y_true.sum()) / linear_factor
    
    else:
        weights = (1 / (y_true.sum(0) / y_true.sum())) ** exp_factor

    keys = np.arange(0,y_true.shape[1],1)
    weight_dictionary = dict(zip(keys, weights))

    return weight_dictionary
    

def age_and_sex_set(values, indices, mean=None):
    
    """
    Retrieve the age and sex from a list of indices. 
    Manipulate them so that they are integer columns.
    """
    
    # Retrieve the selected values given the indices
    age_n_sex = values.iloc[indices][['age', 'sex']]

    # Calculate the mean if it is not provided
    if mean is None:
        aux = age_n_sex['age'][age_n_sex['age'] != 'NaN']
        aux = aux.astype(int)
        aux = aux.mean().round()
        mean_age = int(aux)

    else:
        mean_age = mean

    # There are some 'NaN' values, so we set them to be the mean
    age_n_sex['age'] = age_n_sex['age'].apply(lambda x : mean_age if x == 'NaN' else int(x))
    
    # Bolleanize the sex column
    age_n_sex['sex'] = age_n_sex['sex'].apply(lambda x : 1 if x == 'Male' else 0)

    #Convert all to np.array
    age_n_sex = np.array(age_n_sex)
    
    # Return the mean to use it for the other sets.
    if mean is None:
        return age_n_sex, mean_age
    
    else:
        return age_n_sex


def fit_loess(x, y, frac=0.2):
    
    still_not_fitted = True
    
    while still_not_fitted:
        
        try:
            _, fitted_y, _ = loess_1d(x, y, degree=2, frac=frac)                        
            return fitted_y
            
        except np.linalg.LinAlgError:
            print(f'frac = {frac} generates LinAlgError')
            if frac > 1:
                print('The fraction cannot be greater than 1.')
            else:
                frac += 0.02
                pass


def smooth_ecg_signal(x, y, std_threshold = 2, n_intervals = 50, outlier_frac = 0.01, gross_frac = 0.1, use_lowess = True, analysis=False):


    stds = []
    lowess_by_intervals = []

    n_in_each_interval = 5000 // n_intervals
    join_counter = 0

    for k in range(n_intervals):
        
        begin = k * n_in_each_interval
        end =  (k+1) * n_in_each_interval
        
        yy = y[begin:end]
        xx = x[begin:end]
        
        std = np.std(yy)
        stds.append(std)
        
        if std > std_threshold:
            if join_counter > 0:
                
                if use_lowess:
                    fitted_y = lowess(yyy, xxx, frac=gross_frac)[:, 1]
                    lowess_by_intervals.append(fitted_y)            
                    join_counter = 0
                else:
                    fitted_y = fit_loess(xxx, yyy, frac=gross_frac)
                    lowess_by_intervals.append(fitted_y)            
                    join_counter = 0
                        
            fitted_y = lowess(yy, xx, frac=outlier_frac)[:, 1]
            lowess_by_intervals.append(fitted_y)
        
        else:
            if join_counter == 0:            
                yyy = yy
                xxx = xx
                join_counter +=1
            else:
                yyy = np.append(yyy, yy)
                xxx = np.append(xxx, xx)
                join_counter +=1
                
    if join_counter > 0:
        if use_lowess:
            fitted_y = lowess(yyy, xxx, frac=gross_frac)[:, 1]
            lowess_by_intervals.append(fitted_y)                        
        else:
            fitted_y = fit_loess(xxx, yyy, frac=gross_frac)
            lowess_by_intervals.append(fitted_y)

    if analysis == False:
        return  np.concatenate(lowess_by_intervals)
    else:
        return np.concatenate(lowess_by_intervals), \
               [std_threshold,  np.array(stds)]

def clean_ecg_signal(y, smoothed_y, low_q=0.025, upp_q=0.995, apply_quantiles=True, apply_butter=True, Wn=0.2, analysis=False):

    # The new Y is the original minus the smoothed one.
    y_new = y - smoothed_y 
    
    if apply_quantiles:
        y_corrected = y_new.copy()
        lower_q = np.quantile(y_corrected, low_q)
        upper_q = np.quantile(y_corrected, upp_q)
        below_lower_q = y_corrected[y_corrected < lower_q] # Used if analysis==True
        above_upper_q = y_corrected[y_corrected > upper_q] # Used if analysis==True
        y_corrected[y_corrected < lower_q] = lower_q
        y_corrected[y_corrected > upper_q] = upper_q

        if apply_butter:            
            b, a = scipy.signal.butter(4, Wn, 'low', analog = False)
            y_butter = scipy.signal.filtfilt(b, a, y_corrected)

            if analysis == False:
                return y_new, y_corrected, y_butter
            else:
                return y_new, y_corrected, y_butter, \
                       [low_q, lower_q, below_lower_q], \
                       [upp_q, upper_q, above_upper_q]

        else:
            if analysis == False:
                return y_new, y_corrected
            else:
                return y_new, y_corrected, \
                       [low_q, lower_q, below_lower_q], \
                       [upp_q, upper_q, above_upper_q]

    else:
        return y_new


def pad_zeros_before_n_after(values, cut):

    a = values.copy()

    left_cut = int(a.shape[1] * (1 - cut) / 2)
    right_cut = a.shape[1] -  left_cut

    left_zeros = np.zeros((a.shape[0], left_cut, a.shape[2]))
    right_zeros = left_zeros

    a = a[: , left_cut:right_cut, :]
    a = np.concatenate([left_zeros, a, right_zeros], 1)
    
    return a

### !!! (add citation) !!!
# Function from #1 team in Physionet Challenge 
def apply_filter(signal, filter_bandwidth, fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                 order=order, frequency=filter_bandwidth, 
                                 sampling_rate=fs)
    return signal
