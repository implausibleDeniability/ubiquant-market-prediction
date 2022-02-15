# How to run
```
python training.py --exp-name test_run --config small_base
```

### Debug mode
```
python training.py --debug
```
Debug mode uses 0.01% of the data, turns of wandb and weights checkpointing.

### CPU training
```
python training.py --exp-name run2 --cpu
```
