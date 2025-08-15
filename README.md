# Multi-Step Retrieval via Reinforcement Learning 

## Training
To train Contriever Embedder with Q-RAG on Babilong see `train_pqn.py`. 
To modify hyperparameters either use yaml configs or CLI.

#### Configs
All hyperparameters are set in `configs/`. You can look at the following configs:
* `configs/training.yaml`
* `configs/envs/babilong.yaml`
* `configs/algo/pqn.yaml`

#### CLI
You can change any config you want by directly passing it into training script:

```python
python train_babi_pqn.py seed=100 batch_size=16 accumulate_grads=3 algo.pqn.hyperparams.max_grad_norm=0.5 envs.num_sentences=50 
```


## Testing
Testing only hyperparams are stored in `configs/testing.yaml`. 
To test one of your pretrained models you need to specify path to the folder with training config (`config.yaml`) 
and model weights (`model_best.pt`):
```python
python eval_babi_pqn.py pretrained_path=runs/May30_03-44-01_PQN_qa2_two-supporting-facts envs.num_sentences=1200 num_samples=200
```
Hyperparameters specified in CLI or `configs/testing.yaml` overwrites values from the config in the pretrained_path. 
Priority between all sources is the following: 

`CLI hyperparams > configs/testing.yaml > pretrained_path/config.yaml`, 
where `A > B` means that `A.param1` overwrites `B.param1`.


