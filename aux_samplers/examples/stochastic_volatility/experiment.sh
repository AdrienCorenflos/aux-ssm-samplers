

JAX_PLATFORMS=cpu python experiment.py --style kalman-1 --T 50 --D 30 --no-parallel --no-gpu --target-alpha 0.5 &
JAX_PLATFORMS=cpu python experiment.py --style kalman-2 --T 50 --D 30 --no-parallel --no-gpu --target-alpha 0.75
wait
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman-1 --T 50 --D 30 --parallel --gpu --target-alpha 0.5 &
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman-2 --T 50 --D 30 --parallel --gpu --target-alpha 0.75
wait
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 500 --T 50 --D 30 --no-parallel --gpu --gradient --target-alpha 0.5 &
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc-guided --N 500 --T 50 --D 30 --no-parallel --gpu --no-gradient --target-alpha 0.5
wait
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --T 50 --D 30 --parallel --gpu --gradient --target-alpha 0.5
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --T 50 --D 30 --parallel --gpu --no-gradient --target-alpha 0.5