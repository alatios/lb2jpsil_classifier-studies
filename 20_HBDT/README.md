The naming convention for HBDT files is as follows:

```
HBDT_{P}_{LR}_{MLN}_{MI}_{BAL}
```

Legend:
* `P`: symbol for daughter particles momenta variables used.
Either `VF` for Vertex Fitter, `DTF` for DecayTreeFitter *without* &Lambda; mass constraint, `DTFL` for DecayTreeFitter *with* &Lambda; mass constraint and `NO` for... none of them (an option considered to avoid bias).
* `LR`: the learning rate (shrinks the contribution of each tree).
* `MLN`: maximum number of leaves in each tree.
* `MI`: maximum number of iterations of the boosting process.
* `BAL`: training set balance, either `BAL` for balanced (same number of signal and bkg events) or `SKEW` for skewed (way more background events).
