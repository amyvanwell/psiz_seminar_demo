import pandas as pd
import tensorflow as tf
import os

def main():
    df_data = []
    data_folder = "./raw_data"
    img_folder = "./" + input("Please indicate the folder of your images in psiz-collect. ex. 'img/warblers_finalized': ") + "/"
    img_type = input("Please indicate the image type you use. Ex. '.jpg': ")

    for file in os.listdir(data_folder):
        file_path = data_folder + "/" + file
        df = pd.read_csv(file_path)
        print(df)
        df["n_reference"] = 2
        df["n_select"] = 1
        df["is_ranked"] = True
        df = df[df["trial_type"] == "2c1-embedding-module-new"]

        # convert the embedding arrays into arrays of ints
        for index, row in df.iterrows():
            embedding_array = row["embedding_output"].split(",")
            print(embedding_array)
            new_array = []

            for idx, item in enumerate(embedding_array):
                new_item = item
                for char in [img_folder, img_type]:
                    new_item = new_item.replace(char, "")

                new_array.append(new_item)

                if (idx == 0):
                    df.at[index, "query"] = new_item
                    df.at[index, "img_name"] = new_item


            df.at[index, "embedding_output"] = new_array
    
        df_data.append(df)

    all_data = pd.concat(df_data).reset_index()

    all_data = all_data.drop(['pid', 'view_history', 'success', 'key_press', 'stimulus', 'rt', 'index', 'trial_type', 'trial_index', 'time_elapsed', 'internal_node_id', 'selected', 'references'], axis=1)
    print("Succesfull data parsing. Output:\n")
    print(all_data)
    filename = input("What do you want to name your output CSV? ex. 'Feb27'. Don't include .csv in your input: ")
    filename = filename + ".csv"
    print("\nConverting to CSV ...\n")
    all_data.to_csv(filename)
    print("Succesfully saved to CSV.\n")

main()