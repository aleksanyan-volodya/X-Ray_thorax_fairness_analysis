#!/bin/bash
python train_classifieur.py\
 --logdir ./expe_log/\
 --datadir ./1_clem/\
 --csv ./1_clem/metadata_both.csv\
 --weights_col WEIGHTS\
 --csv_out ./expe_log/preds.csv
