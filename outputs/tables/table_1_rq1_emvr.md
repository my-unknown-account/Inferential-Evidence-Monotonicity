Table 1: Evidence monotonicity violation rate (EMVR) across retrievers. Confidence intervals are obtained by bootstrap resampling at the question level.

| Retriever | EMVR (%) | 95% CI |
| --- | ---: | --- |
| BM25 | 66.68 | [65.89, 67.48] |
| SPLADE | 35.04 | [34.40, 35.67] |
| DPR | 37.64 | [37.00, 38.35] |
| ColBERT | 32.60 | [32.04, 33.19] |
| BGE | 30.17 | [29.62, 30.69] |
| ReasonIR | 55.72 | [55.04, 56.39] |

Note: Estimates and confidence intervals are read from `results/bootstrap/<retriever>/bootstrap_cis.csv`.
