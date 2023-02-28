import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import psiz
from scipy.stats import pearsonr
import pydash as _ 

class SimilarityModel(tf.keras.Model):
    """A similarity model."""

    def __init__(self, percept=None, kernel=None, **kwargs):
        """Initialize."""
        super(SimilarityModel, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

    def call(self, inputs):
        """Call."""
        stimuli_axis = 1
        z = self.percept(inputs['rate2/stimulus_set'])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.kernel([z_0, z_1])

def main():
    folder_one = input("Folder name one: ")
    folder_one = './' + folder_one + "/saved_data/"
    folder_two = input("Folder name two: ")
    folder_two = './' + folder_two + "/saved_data/"

    # Load labels and model 1
    stimulus_labels_1 = np.load(folder_one+'labels.npy',allow_pickle='TRUE').item()
    print(stimulus_labels_1)
    model_inferred_1 = tf.keras.models.load_model(folder_one+'model_inferred_1')

    # Load labels and model 2
    stimulus_labels_2 = np.load(folder_two+'labels.npy',allow_pickle='TRUE').item()
    model_inferred_2 = tf.keras.models.load_model(folder_two+'model_inferred_1')
    print(stimulus_labels_2)
   
   ### COMPARE MODELS ###
    # how to compare 2 models:
    # take the pearson r of their similarity matrices
    n_stimuli = 160
    batch_size = 128

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    # NOTE: We include an placeholder "target" component in dataset tuple to
    # satisfy the assumptions of `predict` method.
    content_pairs = psiz.data.Rate(
        psiz.utils.pairwise_indices(np.arange(n_stimuli) + 1, elements='upper')
    )
    dummy_outcome = psiz.data.Continuous(np.ones([content_pairs.n_sample, 1]))
    tfds_pairs = psiz.data.Dataset(
        [content_pairs, dummy_outcome]
    ).export().batch(batch_size, drop_remainder=False)

    # Define model that outputs similarity based on inferred model.
    model_inferred_similarity_1 = SimilarityModel(
        percept=model_inferred_1.behavior.percept,
        kernel=model_inferred_1.behavior.kernel
    )

    # Define model that outputs similarity based on inferred model.
    model_inferred_similarity_2 = SimilarityModel(
        percept=model_inferred_2.behavior.percept,
        kernel=model_inferred_2.behavior.kernel
        )

    # Compute similarity matrix.
    simmat_1 = model_inferred_similarity_1.predict(tfds_pairs)
    simmat_2 = model_inferred_similarity_2.predict(tfds_pairs)

    # note: chunk these and apply labels?

    res = pearsonr(simmat_1, simmat_2)
    print(res)




main()