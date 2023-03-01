JAX_PLATFORMS=cpu python experiment.py --style kalman-1 --T 250 --D 30 --no-parallel --no-gpu --target-alpha 0.5
JAX_PLATFORMS=cpu python experiment.py --style kalman-2 --T 250 --D 30 --no-parallel --no-gpu --target-alpha 0.5
JAX_PLATFORMS=cpu python experiment.py --style csmc --N 25 --T 250 --D 30 --no-parallel --no-gpu --no-gradient --target-alpha 0.5
JAX_PLATFORMS=cpu python experiment.py --style csmc --N 25 --T 250 --D 30 --no-parallel --no-gpu --gradient --target-alpha 0.5
JAX_PLATFORMS=cpu python experiment.py --style csmc-guided --N 25 --T 250 --D 30 --no-parallel --no-gpu --no-gradient --target-alpha 0.5
JAX_PLATFORMS=cpu python experiment.py --style csmc-guided --N 25 --T 250 --D 30 --no-parallel --no-gpu --gradient --target-alpha 0.5
#wait
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman-1 --T 250 --D 30 --parallel --gpu --target-alpha 0.5 --precision double &
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style kalman-2 --T 250 --D 30 --parallel --gpu --target-alpha 0.5 --precision double
#wait
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --T 250 --D 30 --parallel --gpu --gradient --target-alpha 0.5 &
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc --N 25 --T 250 --D 30 --parallel --gpu --no-gradient --target-alpha 0.5