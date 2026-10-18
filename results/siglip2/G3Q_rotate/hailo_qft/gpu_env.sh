# GPU DFC용 환경 (~/.bashrc 118~121행과 같음). GPU 작업에서만 source. 다른 DFC 작업은 CUDA_VISIBLE_DEVICES=-1 유지
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_CACHE_PATH=$HOME/.nv/ComputeCache
export CUDA_VISIBLE_DEVICES=0
