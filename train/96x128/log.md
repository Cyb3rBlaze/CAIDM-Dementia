# Training Log

## Experiment 01

Goal: Determine a start point for learning rate.
Results: Learning rate within range of 0.0001 seems reasonable. Further testing of more learning rates required.

## Experiment 02

Goal: Measure more learning rates.
Results: Learning rate of 0.0001 seems like a good baseline.

## Experiment 03

Goal: Test filter scale and number of channels before contrastive block.
Results: Further experimentation required. Could try less feature channels and an increased filter scale (best to keep filter scale >= 8).

## Experiment 04

Goal: Test reconstruction with weightage of different magnitudes on a small U-Net architecture (2 blocks).
Results: Reconstruction results look good. Models with higher reconstruction loss weightage had slightly lower reconstruction losses during training.

## Experiment 05

Goal: Test reconstruction with weightage of different magnitudes on a larger U-Net architecture (3 blocks).
Results: Reconstruction results look good. Findings consistent with previous experiment's conclusions.

## Experiment 06

Goal: Optimize classification loss while maintaining reconstruction peformance.
Results: Difficult to differentiate whether regularization is better.

## Experiment 07

Goal: Optimize classification loss while maintaining reconstruction performance. Also test increasing filter scale and decreasing feature channels.
Results: 