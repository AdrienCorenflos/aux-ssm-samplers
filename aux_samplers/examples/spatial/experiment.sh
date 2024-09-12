#CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman --D 8 --T 1024 --parallel --gpu --gradient --target-alpha 0.5 --no-verbose &
#CUDA_VISIBLE_DEVICES=1 python experiment.py --style kalman --D 8 --T 1024 --parallel --gpu --no-gradient --target-alpha 0.5 --no-verbose &
#CUDA_VISIBLE_DEVICES=2 python experiment.py --style csmc --N 25 --D 8 --T 1024 --parallel --gpu --gradient --target-alpha 0.25 --no-verbose &
#CUDA_VISIBLE_DEVICES=3 python experiment.py --style csmc --N 25 --D 8 --T 1024 --parallel --gpu --no-gradient  --target-alpha 0.25 --no-verbose
#sleep 3

CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu python experiment.py --style kalman --D 8 --T 1024 --no-parallel --no-gpu --gradient --target-alpha 0.5 --no-verbose
CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu python experiment.py --style kalman --D 8 --T 1024 --no-parallel --no-gpu --no-gradient --target-alpha 0.5 --no-verbose
CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu python experiment.py --style csmc --N 25 --D 8 --T 1024 --no-parallel --no-gpu --gradient --target-alpha 0.25 --no-verbose
CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu python experiment.py --style csmc --N 25 --D 8 --T 1024 --no-parallel --no-gpu --no-gradient  --target-alpha 0.25 --no-verbose

## loop over time steps
#for T in 32 64 128 256 512 1024
#do
#    # Kalman only
#    JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman --D 8 --T $T --parallel --gpu --target-alpha 0.5 --no-verbose
#    sleep 5
#    JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman --D 8 --T $T --no-parallel --cpu --target-alpha 0.5 --no-verbose
#    sleep 5
#    # CSMC separable
#    JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --D 8 --T $T --parallel --gpu --gradient --target-alpha 0.25 --no-verbose
#    sleep 5
#    JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --D 8 --T $T --no-parallel --cpu --no-gradient  --target-alpha 0.25 --no-verbose
#    sleep 5
#    # CSMC non-separable
#    JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc-guided --N 25 --D 8 --T $T --no-parallel --cpu --no-gradient  --target-alpha 0.25 --no-verbose
#    sleep 5
#    JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc-guided --N 25 --D 8 --T $T --no-parallel --cpu --gradient --target-alpha 0.25 --no-verbose
#    sleep 5
#done
#
#
