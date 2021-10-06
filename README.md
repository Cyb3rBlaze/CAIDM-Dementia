# caidm_dementia

Follow this guide to run an experiment on the cluster using an automated script.

## Step 1: Run an Experiment

Run the `./run.sh` command from the terminal, supplying the necessary parameters. Note, all of the commands for this step can be done manually as well by using the jarvis commands from the shell script files. Refer to Dr. Chang's [README](https://github.com/peterchang77/caidm/blob/master/cluster/README.md) on the private CAIDM repository.

**Important, inside the `train/` directory, please ensure that you have deleted the `scripts/` directory and the directory for the current experiment defined in the `hyper.csv` (if either exist) before running this step's script.**

### Examples

Run an experiment out of the 96x128 train folder on RTX GPUs (0-1):

```
$ ./run.sh 96x128 rtx
```

Run an experiment out of the raw train folder on RTX / GTX GPUs (0-1):

```
$ ./run.sh raw [gr]tx
```

This will send all models defined in `train/raw/hyper.csv` to the cluster for training.

Given a `train/raw/hyper.csv` such as:

```
$ cat train/raw/hyper.csv

output_dir,fold,batch_size,LR,iterations
/home/user/dementia/train/raw/02/01,0,3,0.001,200
```

The training results will be viewable at `/home/user/dementia/train/raw/02/*`.

## Step 2: View the TensorBoard

Run the `./tensorboard.sh` command from the terminal, supplying the necessary parameters.

### Examples

Run a TensorBoard instance for the directory path `96x128/01` which represents the first experiment in the `96x128/` directory:

```
$ ./tensorboard 96x128/01
```

If for example you are on `http://128.195.185.251:8000`, open `http://128.195.185.251:10000` to view the TensorBoard.

## Step 3: Evaluate the Trained Model(s)

Load the correct client YAML files and trained model, then run the respective Jupyter notebook cells to evaluate.