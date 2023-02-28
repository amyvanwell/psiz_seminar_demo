import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os as os

def getImage(path, zoom=0.05):
    return OffsetImage(plt.imread(path), zoom=zoom)

def main():
    # Load labels and model 1
    stimulus_labels = np.load('saved_data/labels.npy',allow_pickle='TRUE').item()
    model_inferred_1 = tf.keras.models.load_model('saved_data/model_inferred_1')
    image_folder = input("Please write the name of the folder with your images. Example:'warblers_finalized' ")
    image_folder_path = os.getcwd() + "/" + image_folder + "/"
    
    #print(stimulus_labels)
    #print(len(stimulus_labels))

    ### PLOT ###
    loc = model_inferred_1.behavior.percept.embeddings.numpy()
    # drop zeroes
    if model_inferred_1.behavior.percept.mask_zero: loc = loc[1:]

    # plot the distances of the embeddings
    fig, ax = plt.subplots()
    plt.scatter(loc[:,0], loc[:,1])

    # set the axes
    # NOTE: YOU MAY NEED TO ADJUST THESE VALUES FOR YOUR EMBEDDING TO BEAUTIFY IT
    #ax.set_ylim(-.15, .15)
    #ax.set_xlim(-.15, .15)

    for i in range(1, len(loc)-1):
        label = ''.join([i for i in stimulus_labels[i+1] if not i.isdigit()])
        # NOTE: ADJUST THE TEXT OFFSET TO MOVE LABELS FURTHER/CLOSER
        text_offset = .01
        plt.text(x=loc[i][0]+text_offset, y = loc[i][1], s=label, fontsize='xx-small')
    
    for x0, y0, label in zip(loc[:,0], loc[:,1], range(1,len(loc)-1)):
        path = image_folder_path + stimulus_labels[label] + ".jpg"
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Save figure with nice margin
    plt.savefig('embedding_visualization.png', dpi = 300, pad_inches = .1)
    #plt.show()

    
main()