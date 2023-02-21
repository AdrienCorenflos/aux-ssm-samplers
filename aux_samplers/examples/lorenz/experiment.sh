#JAX_PLATFORMS=cpu python experiment.py --style kalman-1 --T 250 --D 30 --no-parallel --no-gpu --target-alpha 0.5
#JAX_PLATFORMS=cpu python experiment.py --style kalman-2 --T 250 --D 30 --no-parallel --no-gpu --target-alpha 0.5
#JAX_PLATFORMS=cpu python experiment.py --style csmc --N 25 --T 250 --D 30 --no-parallel --no-gpu --no-gradient --target-alpha 0.5
#JAX_PLATFORMS=cpu python experiment.py --style csmc --N 25 --T 250 --D 30 --no-parallel --no-gpu --gradient --target-alpha 0.5
#JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 5
#JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 10
#JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 20
#JAX_PLATFORMS=cpu python experiment.py --style kalman --no-parallel --no-gpu --freq 40
#wait
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
#wait

#wait
#JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman-1 --T 250 --D 30 --parallel --gpu --target-alpha 0.5 --precision double &
#JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style kalman-2 --T 250 --D 30 --parallel --gpu --target-alpha 0.5 --precision double
#wait
#JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --T 250 --D 30 --parallel --gpu --gradient --target-alpha 0.5 &
#JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc --N 25 --T 250 --D 30 --parallel --gpu --no-gradient --target-alpha 0.5