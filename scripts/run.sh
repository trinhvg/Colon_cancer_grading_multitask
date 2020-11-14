#!/bin/bash
for run_infor in class_ce regress_mae regress_mse multi_ce_mse multi_ce_mae
do
  python trainer.py --run_infor ${run_infor}
done