# Single-Image Super Resolution

This GitLab project repository contains:
1.	The main notebook used for training	the models, **sisr_training_final.ipynb**.
2.	The U-Net model architecture used, **model.py**.
3.	The preprocess class used for pre-processing the data, **preprocess.py**.
4.	The patch extractor class used to extract low-resolution and high-resolution patches from the dataset, **patch_extractor.py**.
5.	The notebooks in which the model was evaluated on the test sets (one different notebook for each test set). The visualized results are also included in the notebooks.
(**Testing_BSD100.ipynb**, **Testing_manga109.ipynb**, **Testing_Set5.ipynb**, **Testing_Set14.ipynb**, **Testing_urban100.ipynb**)
6.	The metric results for the 5 different test sets on which the models were tested in the **results folder** as **CSV files** (one for each test set, averages are at the end of CSV files).
7.	Loss curves visualization notebook (**loss_curves_visualization.ipynb**) that contains loss and metric curves vs. epoch for each model.
8.	TensorBoard logs of all the model runs in the **tensorboard_runs folder**.
