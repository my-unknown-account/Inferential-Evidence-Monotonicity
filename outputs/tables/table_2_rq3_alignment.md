Table 2: Alignment between retrieval-score changes and downstream reader improvements. The last column reports \(P(R\uparrow\mid S\uparrow)-P(R\uparrow\mid S\downarrow)\), with question-level bootstrap 95% confidence intervals.

| Retriever | \(P(R\uparrow\mid S\downarrow)\) | \(P(R\uparrow\mid S\uparrow)\) | Difference (95% CI) |
| --- | ---: | ---: | --- |
| BM25 | 50.14 | 49.15 | -0.99 [-2.05, 0.20] |
| SPLADE | 49.07 | 50.17 | +1.11 [0.03, 2.20] |
| DPR | 49.19 | 50.14 | +0.95 [-0.09, 1.94] |
| ColBERT | 47.71 | 50.78 | +3.07 [2.04, 4.16] |
| BGE | 47.53 | 50.75 | +3.22 [2.15, 4.31] |
| ReasonIR | 47.98 | 52.04 | +4.06 [3.07, 5.14] |

Note: \(S\) means retrieval score and \(R\) means reader utility. Estimates and confidence intervals are read from `results/bootstrap/<retriever>/bootstrap_cis.csv`.
