# Continuous Query Decomposition

This repository contains the official implementation for our ICLR 2021 (Oral, Outstanding Paper Award) paper, [**Complex Query Answering with Neural Link Predictors**](https://openreview.net/forum?id=Mos9F9kDwkz).

```bibtex
@inproceedings{
    arakelyan2021complex,
    title={Complex Query Answering with Neural Link Predictors},
    author={Erik Arakelyan and Daniel Daza and Pasquale Minervini and Michael Cochez},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Mos9F9kDwkz}
}
```

In this work we present CQD, a method that reuses a pretrained link predictor to answer complex queries, by scoring atom predicates independently and aggregating the scores via t-norms and t-conorms.

Our code is based on an implementation of ComplEx-N3 available [here](https://github.com/facebookresearch/kbc).

### 1. Download the pre-trained models

```bash
$ mkdir models/
$ for i in "fb15k" "fb15k-237" "nell"; do for j in "betae" "q2b"; do wget -c http://data.neuralnoise.com/kgreasoning-cqd/$i-$j.tar.gz; done; done
$ for i in *.tar.gz; do tar xvfz $i; done
```
