# Autoencoder ablation

## File structure

* `model.py`: model architecture
* `hyper.csv`: hyperparameter settings
* `logdirs`: tensorboard dir
* `0[1-4]`: run output

## Results

* `bce` [./01/model.hdf5](https://github.com/Cyb3rBlaze/CAIDM-dementia/blob/master/pretrained/autoencoder/01/model.hdf5)
```
              precision    recall  f1-score   support

           0       0.91      0.90      0.90       249
           1       0.53      0.55      0.54        51

    accuracy                           0.84       300
   macro avg       0.72      0.72      0.72       300
weighted avg       0.84      0.84      0.84       300

```

* `bce, contrastive` [./02/model.hdf5](https://github.com/Cyb3rBlaze/CAIDM-dementia/blob/master/pretrained/autoencoder/02/model.hdf5)
```
              precision    recall  f1-score   support

           0       0.89      0.92      0.91       249
           1       0.55      0.45      0.49        51

    accuracy                           0.84       300
   macro avg       0.72      0.69      0.70       300
weighted avg       0.83      0.84      0.84       300

```

* `bce, reconstruction` [./03/model.hdf5](https://github.com/Cyb3rBlaze/CAIDM-dementia/blob/master/pretrained/autoencoder/03/model.hdf5)
```
              precision    recall  f1-score   support

           0       0.92      0.92      0.92       249
           1       0.59      0.59      0.59        51

    accuracy                           0.86       300
   macro avg       0.75      0.75      0.75       300
weighted avg       0.86      0.86      0.86       300

```

* `bce, contrastive, reconstruction` [./04/model.hdf5](https://github.com/Cyb3rBlaze/CAIDM-dementia/blob/master/pretrained/autoencoder/04/model.hdf5)
```
              precision    recall  f1-score   support

           0       0.92      0.67      0.77       249
           1       0.31      0.73      0.43        51

    accuracy                           0.68       300
   macro avg       0.62      0.70      0.60       300
weighted avg       0.82      0.68      0.72       300

```