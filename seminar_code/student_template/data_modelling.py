import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

import psiz

class BehaviorModel(tf.keras.Model):
    """A behavior model.

    No Gates.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize."""
        super(BehaviorModel, self).__init__(**kwargs)
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)
    
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

def build_model(n_stimuli, n_dim):
    """Build model.

    Args:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.

    Returns:
        model: A TensorFlow Keras model.

    """
    # Create a group-agnostic percept layer.
    percept = tf.keras.layers.Embedding(n_stimuli +1, n_dim, mask_zero=True)
    # Create a group-agnostic kernel.
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            # rho = 2 sets to euclidean distance, good for most :)
            rho_initializer=tf.keras.initializers.Constant(2.),
            # array, same dimensions as space, weights for axes, constant 1 -> equalized weights
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        # converts similarity to distance
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            # beta of 10 makes it faster, can trade this off and get same results
            # fix it to 10 for optimizability, doesn't affect interpreting
            # takes longer if 1 for same thing
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel
    )
    model = BehaviorModel(behavior=rank)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        # alter this learning rate if convergence is issue
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model

def fit_model(n_stimuli, tfds_train, tfds_val, tfds_test, n_trial_train):
    ### MODEL ###

    # Use early stopping.
    early_stop = tf.keras.callbacks.EarlyStopping(
        'val_cce', patience=30, mode='min', restore_best_weights=True
    )

    # Use Tensorboard callback.
    #fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
    # cb_board = tf.keras.callbacks.TensorBoard(
    #     log_dir=fp_board_frame, histogram_freq=0,
    #     write_graph=False, write_images=False, update_freq='epoch',
    #     profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    # )
    callbacks = [early_stop]

    # hard code to 2
    # alt, check which one best on validation set
    n_dim = 2
    epochs = 10000

    # Infer embedding.
    model_inferred = build_model(n_stimuli, n_dim)
    history = model_inferred.fit(
        x=tfds_train, validation_data=tfds_val, epochs=epochs,
        callbacks=callbacks, verbose=0
    )
    train_cce = history.history['cce'][-1]
    val_cce = history.history['val_cce'][-1]
    test_metrics = model_inferred.evaluate(
        tfds_test, verbose=0, return_dict=True
    )
    test_cce = test_metrics['cce']

    print(
            '    n_trial_train_frame: {0:4d} | train_cce: {1:.2f} | '
            'val_cce: {2:.2f} | test_cce: {3:.2f} '.format(
                n_trial_train, train_cce,
                val_cce, test_cce
            )
        )
    
    return model_inferred

def main():
    # import TFDS
    tfds_train = tf.data.Dataset.load("saved_data/tfds_train")
    tfds_val = tf.data.Dataset.load("saved_data/tfds_val")
    tfds_test = tf.data.Dataset.load("saved_data/tfds_test")

    n_stimuli = int(input("Please indicate the number of stimuli you  have. Ex. '20': "))
    #n_stimuli = 48
    n_trial_train = len(tfds_train)
    batch_size = 128

    ### MODEL ###
    model_inferred_1 = fit_model(n_stimuli, tfds_train, tfds_val, tfds_test, n_trial_train)
    # NOTE: try this with a different subset of data, i.e. shuffle and then re subset
    model_inferred_2 = fit_model(n_stimuli, tfds_train, tfds_val, tfds_test, n_trial_train)

    # save models
    model_inferred_1.save("saved_data/model_inferred_1")
    model_inferred_2.save("saved_data/model_inferred_2")

    ### COMPARE MODELS ###
    # how to compare 2 models:
    # take the pearson r of their similarity matrices

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

    rho, _ = pearsonr(simmat_1, simmat_2)
    print("--- \nData convergence: (Closer to 1.0 indicates greater stability)")
    print(rho**2)


main()