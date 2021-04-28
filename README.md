# Synthesizability-PU-CGCNN
Synthesizability-PU-CGCNN is a python code for predicting CLscore (crystal-likeness score) which is quantitative synthesizability metric of inorganic crystals. This is a partially supervised machine learning protocol (PU-learning) using CGCNN classifier (by T. Xie et al.) developed by prof. Yousung Jung group at KAIST (contact: ysjn@kaist.ac.kr).

## Developers
Jidon Jang<br> Geun Ho Gu<br>

## Prerequisites
Python3<br> Numpy<br> Pytorch<br> Pymatgen<br>

## Publication
Jidon Jang, Geun Ho Gu, Juhwan Noh, Juhwan Kim, and Yousung Jung, "Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised Learning", Journal of the American Chemical Society, 2020, 142, 44, 18836–18843 (DOI: 10.1021/jacs.0c07384)

## Usage
### [1] Define a customized dataset and generate crystal graphs
To input crystal structures to Synthesizability-PU-CGCNN, you will need to define a customized dataset and pre-generate crystal graph as pickle files for bootstrap aggregating in partially supervised learning. Note that this is required for both training and predicting.
If you want to use cif data in the folder named as “cif_files”, following files should be needed to generate crystal graph.
#### 1) id_prop.csv: a CSV file with two columns for positive data(synthesizable) and unlabeled data(not-yet-synthesized). The first column recodes a unique ID for each crystal, and the second column recodes the value (1 = positive, 0 = unlabeled) according to whether they were synthesized already or not.
#### 2) atom_init.json: a JSON file that stores the initialization vector for each element.
#### 3) ID.cif: a CIF file that recodes the crystal structure, where ID is the unique ID for the crystal.
ex) If you want to generate crystal graph with cutoff radius 8A, maximum 12 neighbors:<br>
`python generate_crystal_graph.py --root ./cif_files --n 12 --r 8 --f ./saved_crystal_graph`<br>
Then, you will obtain preloaded crystal graph files in folder “saved_crystal_graph”<br>

### [2] Train a Synthesizability-PU-CGCNN model
`python main_PU_learning.py --bag 100 --graph ./saved_crystal_graph --split ./split`<br>

Load crystal graph information from 'saved_crystal_graph folder' and generate data split files for PU-learning in 'split' folder.<br>
After training, prediction results for test-unlabeled data (csv file) corresponding to each iteration will be generated.<br>
Result of bootstrap aggregating is saved as 'test_results_ensemble_100models.csv'<br>

### [3] Predict synthesizability of new crystals with pre-trained models
`python predict_PU_learning.py --bag 100 --graph ./saved_crystal_graph --cifs ./cifs --modeldir ./trained_models`<br>

Load crystal graph information from 'saved_crystal_graph' folder and pre-trained models from 'trained_models' folder.<br>
Predict synthesizability of crystal structures in 'cifs' folder (with id_prop.csv file) using the loaded models.<br>
Result of bootstrap aggregating is saved as 'test_results_ensemble_100models.csv'
