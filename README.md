# Sparse Oblique Rule Boosting

This repository contains the implementation, datasets, and experimental code associated with the paper **“Sparse Oblique Rule Boosting for Simpler Additive Rule Ensembles.”**

The proposed method, **Logistic Linear Transformation Boosting (LLTBoost)**, extends traditional additive rule boosting by replacing axis-parallel threshold conditions with **sparse oblique propositions** based on learnable linear transformations of the input features. This provides more expressive rule conditions while explicitly controlling model complexity, with the aim of constructing simpler and interpretable additive rule ensembles without sacrificing predictive accuracy.

The repository includes:

* `lltboost.py` — implementation of the proposed LLTBoost method.
* `dataset.py` and `datasets/` — utilities and datasets used in the experiments.
* `test_method.ipynb` — an example notebook for running and testing the method.
* `other_experiments/` — additional code used for experiments reported in the paper.

## Paper

For a detailed description of the method, theoretical development, and empirical evaluation, see:

**Sparse Oblique Rule Boosting for Simpler Additive Rule Ensembles**
[arXiv preprint](https://arxiv.org/abs/2609.06426)
[Journal](https://link.springer.com/article/10.1007/s10618-026-01241-8?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20261003&utm_content=10.1007/s10618-026-01241-8)

The link will be updated to the final published version once available.
