# Machine-Learning-Course_HW4
Information about the code files:
1. main.py - This file includes all parts of the project except for the pre-processing of the data. 
	First, the data is read from the csv files. After that, all the steps of part B are performed, including selecting features according to the K value and classification, using CV. 
	In addition, the implementation of parts C and D is also performed in this file, all the results are saved to an Excel file.
	At the end of the file, the statistical analyzes required for the results obtained are performed.

2. feature_selection.py - This file contains functions of the six feature selection algorithms we implemented. 
			This file is activated using the main.

3. New_PCA_IG_SVM.py - This file contains the improved algorithm that we implemented in part C of the task. 
			This algorithm is run from the main file.

4. data_preprocessing.py - This file contains the pre-processing of the data according to the instructions in the task,
 			and includes saving the new data to the attached csv files.

In order to run the parts of the project, you need to run the main file.
