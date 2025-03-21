#!/bin/bash
python train_classifieur.py\
 --logdir ./expe_log/\
 --datadir ./DATA_DEMO/\
 --csv ./DATA_DEMO/metadata.csv\
 --weights_col WEIGHTS\
 --csv_out ./expe_log/preds.csv