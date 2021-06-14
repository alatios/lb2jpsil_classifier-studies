The naming convention for BDT files is as follows:

```
BDT_{P}_{LR}_{MD}_{MF}_{NE}_{SS}
```

Legend:
* `P`: symbol for daughter particles momenta variables used.
Either `VF` for Vertex Fitter, `DTF` for DecayTreeFitter without $\Lambda$ mass constraint, `DTFL` for DecayTreeFitter with $\Lambda$ mass constraint and 'NO` for... none of them.
* `LR`: the learning rate (shrinks the contribution of each tree).
* `MD`: maximum depth, i.e. maximum number of nodes in each tree.
* `MF`: maximum number of features considered at each split.
* `NE`: number of boosting stages (*estimators*, in sklearn lingo).
* `SS`: subsample ratio, i.e. fraction of samples used for fitting base learners.
