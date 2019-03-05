#!/usr/bin/env bash

oarsub -l "walltime=48:0:0"   "source gpu_setVisibleDevices.sh;
                                                    source activate py3.5_tf1.5;
                                                    cd /scratch/artemis/blecouat/averaging_comparison_tf;
                                                    python -u wgan_gp.py --log wgan"

oarsub -l "walltime=48:0:0"   "source gpu_setVisibleDevices.sh;
                                                    source activate py3.5_tf1.5;
                                                    cd /scratch/artemis/blecouat/averaging_comparison_tf;
                                                    python -u extra.py --log extra"