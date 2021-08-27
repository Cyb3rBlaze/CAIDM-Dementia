# Experiment Script Guide

Follow this guide to run an experiment on the cluster using an automated script.

## Step 1: Run an Experiment

Run the `./run.sh` command from the terminal, supplying the necessary parameters. Note, all of the commands for this step can be done manually as well by using the jarvis commands from the shell script files. Refer to Dr. Chang's [README](https://github.com/peterchang77/caidm/blob/master/cluster/README.md) on the private CAIDM repository.

**Important, inside the `train/` directory, please ensure that you have deleted the `scripts/` directory and the directory for the current experiment defined in the `hyper.csv` (if either exist) before running this step's script.**
### Examples

Run an experiment out of the 96x128 train folder on RTX GPUs (0-1): 
```
./run.sh 96x128 "rtx.*worker-[0,1]"
```

Run an experiment out of the raw train folder on RTX / GTX GPUs (0-1):
```
./run.sh raw "[gr]tx.*worker-[0,1]"
```

## Step 2: View the TensorBoard

Run the `./tensorboard.sh` command from the terminal, supplying the necessary parameters.

### Examples

Run a TensorBoard instance for the directory path `96x128/01` which represents the first experiment in the `96x128/` directory:
```
./tensorboard 96x128/01
```

## Step 3: Evaluate the Trained Model(s)

Load the correct client YAML files and trained model, then run the respective Jupyter notebook cells to evaluate.