


JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --D 8 --parallel --gpu --gradient --target-alpha 0.25 --no-verbose
sleep 5
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc --N 25 --D 8 --parallel --gpu --no-gradient  --target-alpha 0.25 --no-verbose
sleep 5
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman --D 8 --parallel --gpu --target-alpha 0.5 --no-verbose
sleep 5
#  JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --D $D --no-parallel --gpu --gradient --target-alpha 0.25 --no-verbose&
#  JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc --N 25 --D $D --no-parallel --gpu --no-gradient  --target-alpha 0.25 --no-verbose
#  sleep 5
