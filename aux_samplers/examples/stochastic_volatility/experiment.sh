JAX_PLATFORM_NAME=cpu python experiment.py --style kalman-1 --T 250 --D 30 --no-parallel --no-gpu --target-alpha 0.5
JAX_PLATFORM_NAME=cpu python experiment.py --style kalman-2 --T 250 --D 30 --no-parallel --no-gpu --target-alpha 0.5
JAX_PLATFORM_NAME=cpu python experiment.py --style csmc --N 25 --T 250 --D 30 --no-parallel --no-gpu --no-gradient --target-alpha 0.5
JAX_PLATFORM_NAME=cpu python experiment.py --style csmc --N 25 --T 250 --D 30 --no-parallel --no-gpu --gradient --target-alpha 0.5
JAX_PLATFORM_NAME=cpu python experiment.py --style csmc-guided --N 25 --T 250 --D 30 --no-parallel --no-gpu --no-gradient --target-alpha 0.5
JAX_PLATFORM_NAME=cpu python experiment.py --style csmc-guided --N 25 --T 250 --D 30 --no-parallel --no-gpu --gradient --target-alpha 0.5
#wait
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=2 python experiment.py --style kalman-1 --T 250 --D 30 --parallel --gpu --target-alpha 0.5 --precision double
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=3 python experiment.py --style kalman-2 --T 250 --D 30 --parallel --gpu --target-alpha 0.5 --precision double
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=2 python experiment.py --style csmc --N 25 --T 250 --D 30 --parallel --gpu --gradient --target-alpha 0.5
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=3 python experiment.py --style csmc --N 25 --T 250 --D 30 --parallel --gpu --no-gradient --target-alpha 0.5