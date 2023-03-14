import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import psiz 

def getImage(path, zoom=0.05):
    return OffsetImage(plt.imread(path), zoom=zoom)

def main():
    # Load labels and model 1
    stimulus_labels = np.load('saved_data/labels.npy',allow_pickle='TRUE').item()
    model_inferred_1 = tf.keras.models.load_model('saved_data/model_inferred_1')

    ### PLOT ###

    loc = model_inferred_1.behavior.percept.embeddings.numpy()
    # drop zeroes
    if model_inferred_1.behavior.percept.mask_zero: loc = loc[1:]

    # plot the distances of the embeddings
    fig, ax = plt.subplots()
    plt.scatter(loc[:,0], loc[:,1])
    ax.set_ylim(-.7, .7)
    ax.set_xlim(-.7, .7)

    for i in range(1, len(loc)):
        label = ''.join([i for i in stimulus_labels[i+1] if not i.isdigit()])
        plt.text(x=loc[i][0]+.035, y = loc[i][1], s=label, fontsize='xx-small')
    
    for x0, y0, label in zip(loc[:,0], loc[:,1], range(1,161)):
        path = "./warblers_finalized/" + stimulus_labels[label] + ".jpg"
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Save figure with nice margin
    plt.savefig('post_training.png', dpi = 300, pad_inches = .1)
    #plt.show()

    # plotting of single species
    for species in ["G", "R", "T", "C"]:
        plt.clf()
        species_indices = []
        
        for i in range(1, len(loc)):
            if species in stimulus_labels[i]:
                species_indices.append(i)
        
        # plot the distances of the species-only embeddings
        fig, ax = plt.subplots()

        for x0, y0, label in zip(loc[:,0], loc[:,1], range(1,161)):
            if label in species_indices:
                path = "./warblers_finalized/" + stimulus_labels[label] + ".jpg"
                ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
                ax.add_artist(ab)

        ax.set_ylim(-.7, .7)
        ax.set_xlim(-.7, .7)

        label = species + "_post_training.png"
        plt.savefig(label, dpi = 300, pad_inches = .1)

    
main()