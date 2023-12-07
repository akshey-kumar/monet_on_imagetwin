### MONET on imagetwin data

This repo is meant to detect image plagiarism in the from of duplications of western blot images by using algorithm MONET on image twin data. MONET is tested and evaluated on the train, test and validation data.

To run, first create the virtual environment and pip install -r requirements.txt

Then add the directory dataset into the repo with the following structure

dataset
├── README.md
├── data_test
│   ├── features_test
│   └── labels_test
├── data_train
│   ├── features_train
│   └── labels_train
├── data_validation
│   ├── features_validation
│   └── labels_validation
└── load_data.py

Then run the script:

python3 classification.py --dataset train --batch-size 16  

This executes the code in classification.py which loads the data using dataloader in the specific form required for torch modules. Then it passes the data through the monet model, and as output produces masks. Each mask is saved within the original parent directories where the images are for ease of comparison.  Then from the masks, a classification output is decided based on whether a sufficiently large fraction of the image is detected as duplicated. Finally, from the predicted labels, the prec, acc, recall, f1 are computed by comparision with the true labels.


