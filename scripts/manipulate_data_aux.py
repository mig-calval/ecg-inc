import numpy as np
import pandas as pd

def initial_data_preproc(X, Y):

    # np.arrays are more managable than lists
    # X = np.array(X)
    X = np.array(X, dtype=np.float32)

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