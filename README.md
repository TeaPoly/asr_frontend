# ASR Frontend
PyTorch implementation ASR frontend, like [PCEN](https://arxiv.org/pdf/1607.05666.pdf), [Mel filter bank log energy](https://www.sciencedirect.com/science/article/abs/pii/S0167639311001622?via%3Dihub).

# Usage
The following is a example for using PCEN:
```python
import pcen
import numpy as np

b, s, d = 32, 100, 40
filterbanks = np.random.uniform(low=0.5, high=13.3, size=(b, s, d))
filterbanks = torch.from_numpy(filterbanks.astype(dtype=np.float32))
trainable_pcen = pcen.Pcen(d)
pcen_features = trainable_pcen(filterbanks)
```

## Citation
Wang, Yuxuan, Pascal Getreuer, Thad Hughes, Richard F. Lyon, and Rif A. Saurous. [Trainable frontend for robust and far-field keyword spotting](https://arxiv.org/pdf/1607.05666.pdf). In _Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on_, pp. 5670-5674. IEEE, 2017.
```tex
@inproceedings{wang2017trainable,
  title={Trainable frontend for robust and far-field keyword spotting},
  author={Wang, Yuxuan and Getreuer, Pascal and Hughes, Thad and Lyon, Richard F and Saurous, Rif A},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on},
  pages={5670--5674},
  year={2017},
  organization={IEEE}
}
```
