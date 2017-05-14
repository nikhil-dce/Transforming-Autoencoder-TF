# Transforming-Autoencoder-TF
Tensorflow implementation of "Transforming Autoencoders" (Proposed by G.E.Hinton, et al.)

### Result
Column 1 is the input and Column 2 is the expected output after translation. Column 3 represents the output generated from the transforming autoencoder after training for 800 epochs.
![Result](extras/epoch_800.png)

### Source
+ `capsule.py` is the complex capsule which recognizes and generates the respective visual entity after applying the transformation
+ `trans_ae.py` creates the above capsules for all visual entities in the data
+ `tf_model.py` Trains and validates the code

