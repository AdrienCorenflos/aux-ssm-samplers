JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style kalman --parallel --gpu --freq 1
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman --parallel --gpu --freq 2
wait
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style kalman --parallel --gpu --freq 4 &
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style kalman --parallel --gpu --freq 8
wait
JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 1
JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 2
JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 4
JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 8
