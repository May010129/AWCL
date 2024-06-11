## Acknowledgements

We utilized the overall framework [SupContrast]https://github.com/HobbitLong/SupContrast for our code, we mainly modify the 'losses.py' and 'main_supcon.py' to realize our idea.

Usage:
```python
from losses import SupConLoss

# define loss with a temperature `temp`
criterion = SupConLoss(temperature=temp)

# features: [bsz, n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
features = ...
# labels: [bsz]
labels = ...

# SupContrast
loss = criterion(features, labels)
# or SimCLR
loss = criterion(features)
...
```

## Running
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs, and/or switch to CIFAR100 by `--dataset cifar100`.  
**(1) Standard Cross-Entropy**
```
python main_ce.py --batch_size 1024 \
  --learning_rate 0.8 \
  --cosine --syncBN \
```
**(2) Supervised Contrastive Learning**  
Pretraining stage:
```
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.1 \
```

<s>You can also specify `--cosine` and `--syncBN` 

Linear evaluation stage:
```
python main_linear.py --batch_size 512 \
  --learning_rate 5 \
  --ckpt /path/to/model.pth
```
**(3) SimCLR**  
Pretraining stage:
```
python main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --method SimCLR
  --version weighted
```
The `--method SimCLR` flag simply stops `labels` from being passed to `SupConLoss` criterion.
The `--version weighted` flag simply choose the loss function during training.
Linear evaluation stage:
```
python main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --ckpt /path/to/model.pth
```

## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```
