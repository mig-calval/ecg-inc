import pandas as pd

def load_labels(path):

    """
    Load the SNOMED CT labels from the path, along with the urgency of the treatment
    according to the criteria supplied by Instituto Nacional de Cardiología Ignacio Chávez.
    
    Inputs:
        path - str; contains relative path to the excel files.
    """
    
    # Load labels given the paths
    most_freq_labels = pd.read_excel(path+'Most_Frequent_Labels-DAG.xlsx', header=6)
    less_freq_labels = pd.read_excel(path+'Less_Frequent_Labels-DAG.xlsx', header=6)

    # The last one is just the sum per columns
    most_freq_labels = most_freq_labels[:-1]
    less_freq_labels = less_freq_labels[:-1]

    # Convert the SNOMED CT Code to integer
    most_freq_labels['SNOMED CT Code'] = most_freq_labels['SNOMED CT Code'].apply(lambda x : int(x))
    less_freq_labels['SNOMED CT Code'] = less_freq_labels['SNOMED CT Code'].apply(lambda x : int(x))

    # Rename the column for the Urgency of the treatment
    most_freq_labels = most_freq_labels.rename(columns={'Here:':'Urgency'})
    less_freq_labels = less_freq_labels.rename(columns={'Here:':'Urgency'})

    # Modify the "empty" case from a "." to a 4
    most_freq_labels['Urgency'] = most_freq_labels['Urgency'].apply(lambda x : 4 if x == '.' else x)
    less_freq_labels['Urgency'] = less_freq_labels['Urgency'].apply(lambda x : 4 if x == '.' else x)
    
    # Add an identifer to know whether it belongs to the "most" or "less" frquent kind
    most_freq_labels['Kind'] = 'most'
    less_freq_labels['Kind'] = 'less'
    
    # Add everything into one DF and set the index to be the SNOMED CT Code
    labels = pd.concat([most_freq_labels, less_freq_labels])
    labels.index =  labels['SNOMED CT Code']
    labels.drop(['SNOMED CT Code'], axis=1, inplace=True)

    return labels