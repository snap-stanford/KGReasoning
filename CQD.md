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

## 1. Download the pre-trained models

To download and decompress the pre-trained models, execute the folloing commands:

```bash
$ mkdir models/
$ for i in "fb15k" "fb15k-237" "nell"; do for j in "betae" "q2b"; do wget -c http://data.neuralnoise.com/kgreasoning-cqd/$i-$j.tar.gz; done; done
$ for i in *.tar.gz; do tar xvfz $i; done
```

## 2. Answer the complex queries

One catch is that the query answering process in CQD depends on some hyperparameters, i.e. the "beam size" `k`, the t-norm to use (e.g. `min` or `prod`), and the normalisation function that maps scores to the `[0, 1]` interval; in our experiments, we select these on the validation set. Here are the commands to execute to evaluate CQD on each type of queries:

### 2.1 -- FB15k

1p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete
[..]
Test 1p MRR at step 99999: 0.891426
Test 1p HITS1 at step 99999: 0.857939
Test 1p HITS3 at step 99999: 0.915589
[..]
```

2p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --valid_steps 20 --tasks 2p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm prod --cqd-k 64 --cuda
[..]
Test 2p HITS3 at step 99999: 0.791121
[..]
```

3p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --valid_steps 20 --tasks 3p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm prod --cqd-sigmoid --cqd-k 4 --cuda
[..]
Test 3p HITS3 at step 99999: 0.459223
[..]
```

2i queries:

```bash
PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2i --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm prod --cqd-k 16 --cuda
[..]
Test 2i HITS3 at step 99999: 0.788954
[..]
```

3i queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3i --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm prod --cqd-k 16 --cuda
[..]
Test 3i HITS3 at step 99999: 0.837378
[..]
```

ip queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks ip --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm prod --cqd-k 16 --cuda
[..]
Test ip HITS3 at step 99999: 0.649221
[..]
```

pi queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pi --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm prod --cqd-k 64 --cuda
[..]
Test pi HITS3 at step 99999: 0.681604
[..]
```

2u queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2u --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm min --cqd-normalize --cqd-k 16 --cuda
[..]
Test 2u-DNF HITS3 at step 99999: 0.853601
[..]
```

up queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks up --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-q2b --cqd discrete --cqd-t-norm min --cqd-sigmoid --cqd-k 16 --cuda
[..]
Test up-DNF HITS3 at step 99999: 0.709496
[..]
```

### 2.2 -- FB15k-237

1p queries:

```bash
$ PYTHONPATH=. python3 PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cuda
[..]
Test 1p HITS3 at step 99999: 0.511910
[..]
```

2p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-k 64 --cuda
[..]
Test 2p HITS3 at step 99999: 0.286640
[..]
```

3p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3p --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-sigmoid --cqd-k 4 --cuda
[..]
Test 3p HITS3 at step 99999: 0.199947
[..]
```

2i queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2i --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-normalize --cqd-k 16 --cuda
[..]
Test 2i HITS3 at step 99999: 0.376709
[..]
```

3i queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3i --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-normalize --cqd-k 16 --cuda
[..]
Test 3i HITS3 at step 99999: 0.488725
[..]
```

ip queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks ip --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-k 16 --cuda
[..]
Test ip HITS3 at step 99999: 0.182000
[..]
```

pi queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pi --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-normalize --cqd-k 64 --cuda
[..]
Test pi HITS3 at step 99999: 0.267872
[..]
```

2u queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2u --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm min --cqd-normalize --cqd-k 16 --cuda
[..]
Test 2u-DNF HITS3 at step 99999: 0.323751
[..]
```

up queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/FB15k-237-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks up --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-q2b --cqd discrete --cqd-t-norm prod --cqd-sigmoid --cqd-k 16 --cuda
[..]
Test up-DNF HITS3 at step 99999: 0.225360
[..]
```

### 2.2 -- NELL 995

1p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cuda
[..]
Test 1p HITS3 at step 99999: 0.663197
[..]
```

2p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2p --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm prod --cqd-k 64 --cuda
[..]
Test 2p HITS3 at step 99999: 0.351218
[..]
```

3p queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --valid_steps 20 --tasks 3p --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm prod --cqd-sigmoid --cqd-k 2 --cuda
[..]
Test 3p HITS3 at step 99999: 0.263724
[..]
```

2i queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2i --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm prod --cqd-normalize --cqd-k 16 --cuda
[..]
Test 2i HITS3 at step 99999: 0.422821
[..]
```

3i queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3i --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm prod --cqd-normalize --cqd-k 16 --cuda
[..]
Test 3i HITS3 at step 99999: 0.538633
[..]
```

ip queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks ip --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm prod --cqd-k 16 --cuda
[..]
Test ip HITS3 at step 99999: 0.234066
[..]
```

pi queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pi --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm prod --cqd-normalize --cqd-k 64 --cuda
[..]
Test pi HITS3 at step 99999: 0.315222
[..]
```

2u queries:

```bash
$ PYTHONPATH=. python3 main.py --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2u --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm min --cqd-normalize --cqd-k 16 --cuda
[..]
Test 2u-DNF HITS3 at step 99999: 0.541287
[..]
```

up queries:

```bash
$ PYTHONPATH=. python3 main.py --do_valid --do_test --data_path data/NELL-q2b -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks up --print_on_screen --test_batch_size 1 --checkpoint_path models/nell-q2b --cqd discrete --cqd-t-norm min --cqd-sigmoid --cqd-k 16 --cuda
[..]
Test up-DNF HITS3 at step 99999: 0.290282
[..]
```
