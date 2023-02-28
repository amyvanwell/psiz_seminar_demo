import numpy as np
import pandas as pd
import tensorflow as tf
import psiz
import os

## DATA_PARSING.PY
## INPUT: FILENAME FOR A CSV FILE WITH PSIZ-COLLECTION OBSERVATIONS.
## OUTPUT: SAVES TFDS FOR THE TRAINING, VALIDATION, AND TEST OBSERVATIONS. 

def parseRawData(data_file_path, preexisting_labels = 0):
    df = pd.read_csv(data_file_path)

    # NOTE:
    # to convert to tensorflow, need a numpy array of all the same type 
    # convert the raw df input into matrix with cols 1-9 for embedding output, col 'attention' for attention weights
    # those values will be put into an array similar to ['T32', 'G12', ... , '0.9']

    content = []
    stimset = []
    select_indices = []
    query_indices = {}
    final_query_indices = {} if preexisting_labels == 0 else preexisting_labels
    
    # go through rows to get a list of all the possible images
    # save the indices of each query to a separate dictionary to be used as labels for those items in the embedding arrays
    for index, row in df.iterrows():
        query = row["query"]
        if query not in stimset: 
            stimset.append(query)
            # NOTE: ITEMS ARE LABELED AS THEIR INDEX +1 SO THAT 0 IS NOT A LABEL
            query_indices[query] = len(stimset)

    # iterate though list to get the embedding arrays
    for index, row in df.iterrows():
        embedding_row = np.array(row["embedding_output"].replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(","))       
        # convert the stimulus title with its array index in the stimset array
        for index, item in enumerate(embedding_row):
            if preexisting_labels == 0:
                final_query_indices[int(query_indices[item])] = embedding_row[index]
                embedding_row[index] = int(query_indices[item])
            else:
                presaved_label = 0
                for label in final_query_indices:
                    if final_query_indices[label] == item:
                        presaved_label = label
                embedding_row[index] = int(presaved_label)

        select_indices.append([embedding_row[1], embedding_row[2]])

        if len(embedding_row) == 3: content.append(embedding_row)

    content_np_array = np.array(content).astype(int)
    select_indices = np.array(select_indices).astype(int)

    return content_np_array, stimset, final_query_indices, select_indices

def main():
    data_file = input("Please write the name of the input file. Example: 'nov22.csv' ")
    data_file_path = os.getcwd() + "/" + data_file
    label_path = "saved_data/original_labels.npy"

    # check to see if there is any existing labels or data
    if os.path.isfile(label_path):
        stimulus_labels = np.load("saved_data/original_labels.npy",allow_pickle='TRUE').item()
        # parse raw excel data for stimuli array with observations, content, and observations
        stimuli, stimulus_set, stimulus_labels, select_indices = parseRawData(data_file_path, stimulus_labels)
    else:
        # parse raw excel data for stimuli array with observations, content, and observations
        stimuli, stimulus_set, stimulus_labels, select_indices = parseRawData(data_file_path)

    print("\nOBSERVATIONS:\n", stimuli)
    print("\nSTIMSET:\n",stimulus_set)
    print("\nSTIM LABELS:\n",stimulus_labels)
    print("\nSELECT INDICES:\n",select_indices)

    # save the observations and labels to numpy items
    # labels used when plotting the data
    np.save("saved_data/observations", stimuli)
    np.save("saved_data/labels", stimulus_labels)

    ### CONVERT OBSERVATIONS TO TFDS, PARTITION THEM ###
    # NOTE: ADD THE CHECK HERE FOR IF THE DATA SAVE HAS ALREADY OCCURED, IF SO ABORT FOR NOW
    # TODO: CREATE WAY TO CONVERT AND ADD NEW TRIALS TO EXISTING DATASETS
    
    if not os.path.isdir("saved_data/tfds_test"):
        n_trial = len(stimuli)
        batch_size = 128 # lower batch size to min local optima

        # Generate a random set of trials
        # These are trained to match our dataset
        content = psiz.data.Rank(stimuli, n_select=1)
        # Data is in order, therefore outcome is always 0, choice 0 is 1 and 2 items 
        # in array of possibilities (matrix)
        outcome_index = np.zeros([n_trial, 1], dtype=int)

        # convert the outcome indices into a psiz object with proper number of trials
        outcome = psiz.data.SparseCategorical(outcome_index, depth = content.n_outcome)
        # create dataset that contains the options (content) and the choice made (outcome)
        pds = psiz.data.Dataset([content,outcome])
        # export the psiz dataset into the tensorflow dataset
        tfds_all = pds.export(export_format='tfds')

        # Partition data into 80% train, 10% validation and 10% test set.
        # NOTE: np.floor returns a float
        n_trial_train = int(np.floor(.8 * n_trial))
        n_trial_val = int(np.floor(.1 * n_trial))

        # NOTE: shuffle batches for stochasticity
        # NOTE: buffer size is amount of batches to preprocess before saving to model, complete trial size if small amount
        tfds_train = tfds_all.take(n_trial_train).cache().shuffle(buffer_size=n_trial_train,
                reshuffle_each_iteration=True).batch(
            batch_size, drop_remainder=False
        )
        tfds_valtest = tfds_all.skip(n_trial_train)
        tfds_val = tfds_valtest.take(n_trial_val).cache().batch(
            batch_size, drop_remainder=False
        )
        tfds_test = tfds_valtest.skip(n_trial_val).cache().batch(
            batch_size, drop_remainder=False
        )

        tfds_train.save("saved_data/tfds_train")
        tfds_val.save("saved_data/tfds_val")
        tfds_test.save("saved_data/tfds_test")

    else:
        print("PARSED DATA ALREADY EXISTS. LOADING.")

        print(tf.data.Dataset.load("saved_data/tfds_train"))
        print(tf.data.Dataset.load("saved_data/tfds_val"))
        print(tf.data.Dataset.load("saved_data/tfds_test"))

main()