# highlighting_mislabelled_specimens
This repository contains the code for my paper titled 'A computer vision method for finding mislabelled specimens within natural history collections'




Instructions for Running Scripts in This Repository

    Configure and Run the Control File:
        Open control_file.py and set the desired number of iterations (e.g., iterations = 100).
        Run the script. This will produce a file called iterations.txt, which contains the iteration count.

    Run the Image Classification Script:
        Execute image_classification_multirun.py. This script will run for the number of iterations defined in iterations.txt.
        Each run will generate an Excel file containing TensorFlow evaluation results.

    Aggregate Results:
        Open and run master_excel_file_creation.ipynb from start to finish. This notebook collects all Excel files generated during the runs and aggregates the results, producing a final Excel file for analysis in the paper.
   
