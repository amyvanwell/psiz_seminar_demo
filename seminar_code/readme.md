## PSIZ SEMINAR DEMO CODE ##

## A COMPLETE EXAMPLE CAN BE SEEN IN THE FOLDER 'demo_code'
## YOU WILL BE OPERATING WITHIN THE 'student_template' FOLDER

## SET UP
1: IMPORT YOUR DATA FOLDER FROM THE PSIZ-COLLECT CODE INTO THE RAW_DATA FOLDER UNDER 'student_template' 
2: PASTE YOUR IMAGE FOLDER INTO the 'student_template' folder

## DATA CONVERSION
This file converts your raw data from psiz-collect into one CSV.

To convert your data in the raw_data folder, 
1: Open a terminal

2: Navigate to the student_template folder by executing the following lines:
`cd student_template`

3: Run the python file 'convert_raw_data'
`python convert_raw_data.py`

The file will prompt you for the name of your *original* image folder, i.e. the image folder in psiz-collect. The purpose of this is to remove that from the raw data in order to only use the image name only, i.e. it will beautify the values. You will also be prompted for the file type for your images. 

Once the conversion is done, you will be prompted for name you'd like to save the output to.

The data will be saved to this file. You will see it pop up in your directory when the program is complete.

## DATA PARSING
This file converts the CSV with your converted raw data, into a format that works with tensorflow. Your trials will be segmented into train and validate. As well, the labels for your stimuli are generated and saved at this stage.

To parse your converted data: 
1: Open a terminal

2: Navigate to the student_template folder by executing the following lines:
`cd student_template`

3: Run the python file 'convert_raw_data'
`python data_parsing.py`

You will be prompted for the same of your converted CSV file.

Once the program has finished running, you will see folder for new tensorflow datasets in your 'saved_data' folder, alongside saved labels and observations.

## DATA MODELLING
You are now ready to take your data and create the psychological embeddings.

To model your data:
1: Open a terminal

2: Navigate to the student_template folder by executing the following lines:
`cd student_template`

3: Run the python file 'data_modelling'
`python data_modelling.py`

You will be prompted for the number of stimuli that you have. Afterwards, the modelling will begin. This may take a few minutes, depending on how many observations you have. The program assumes you have 2C1 trials.

Once the program is over, it will print out a value for data convergence for the current model.

## DATA VISUALIZATION
This file takes a completed model and creates a similarity space visualization.

To visualize your  data: 
1: Open a terminal

2: Navigate to the student_template folder by executing the following lines:
`cd seminar_code`
`cd student_template`

3: Run the python file 'convert_raw_data'
`python data_visualization.py`

You will be prompted for the same of your image folder, in the *current* directory.

The image will be saved as 'embedding_visualization.png'