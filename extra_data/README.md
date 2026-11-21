# extra_data

Datasets that the experiments use but that are not in the authors' repository. Put the files here
(they are not committed) and run `bash exp/run_all.sh moa e3 figures` again: only the runs that use
the new files are added, everything already finished is skipped.

| Files | Used by | What happens |
|---|---|---|
| `SUP2I.arff`, `SUP3A.arff`, `SUP3G.arff` | E1 (paper Tabs. 3-4, Figs. 4-5) | each file present adds that dataset for all E1 methods except AXGBr |
| `FriedmanLea_1` ... `FriedmanLea_10`, `FriedmanLea_12` (likewise `FriedmanGra_*`, `FriedmanGsg_*`) | E3 (paper Fig. 7) | each dataset whose 11 files are all present is added; `_1` ... `_10` are the training sets, `_12` the test set |

The Friedman split files are ARFF files without the `.arff` extension, named as the authors'
`exp/Bias_Variance_Decomp/BVDecompSyntheticLatest.py` reads them. Unzip `.zip` archives first.
