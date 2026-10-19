#!/bin/bash
ls -la ~/hailo10_work/opt_official/ ~/hailo10_work/calib/
C=$(ls -d /mnt/c/study/3/2/2026*/oslab_mobile/rn50/cache); echo $C; ls $C | grep -i pets
source ~/hailo10_venv/bin/activate
python3 -c "import numpy as np,glob;c=glob.glob('/mnt/c/study/3/2/2026*/oslab_mobile/rn50/cache')[0];print(np.load(c+'/pets_dev_y.npy').shape)"
nproc; free -g
