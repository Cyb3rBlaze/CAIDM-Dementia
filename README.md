# caidm_dementia
  
## The repository structure:

* yml/ - Contains YML configuration files for loading data.
* train/ - Contains model implementation, hyperparameter settings, and folders of experiments.
* scripts/ - Contains automation scripts to run cluster experiments and Jupyter notebooks for EDA and evaluation.

## Training a model:

Example for training a model.

```shell
$ cd scripts

$ ./run.sh 96x128 rtx
```

This will send all models defined in `train/96x128/hyper.csv` to the cluster for training.

Given a `train/96x128/hyper.csv` such as:

```shell
$ cat train/96x128/hyper.csv

output_dir,fold,batch_size,LR,iterations
/home/user/dementia/train/96x128/02/01,0,3,0.001,200
```

The training results will be viewable at `/home/user/dementia/train/96x128/02/*`.

Additionally, you may bring up a tensorboard using these commands:

```shell
$ cd /home/user/dementia/train/96x128/02/

$ tensorboard --logdir jmodels/ --bind_all --port 10000
```

If for example you are on `http://128.195.185.249:8000`, open `http://128.195.185.249:10000` to view the tensorboard.
