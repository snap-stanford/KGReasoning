# KGReasoning
This repo contains several algorithms for multi-hop reasoning on knowledge graphs, including the official Pytorch implementation of [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://arxiv.org/abs/2010.11465).

**Models**
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [GQE](https://arxiv.org/abs/1806.01445)

**KG Data**

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](). Note the two use the same training queries, but the difference is that the valid/test queries in BetaE paper have a maximum number of answers, making it more realistic.

**Examples**

Please refer to the `examples.sh` for the scripts of all 3 models on all 3 datasets.

**Citations**

If you use this repo, please cite the following paper.

```
@inproceedings{
 ren2020beta,
 title={Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs},
 author={Hongyu Ren and Jure Leskovec},
 booktitle={Neural Information Processing Systems},
 year={2020}
}
```
