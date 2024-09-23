export XLA_FLAGS="--xla_force_host_platform_device_count=8"
BATCH_SIZE=8


JAX_PLATFORM_NAME=cpu sleep 5; python experiment.py --style kalman --no-parallel --no-gpu --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE} &
JAX_PLATFORM_NAME=cpu sleep 5; python experiment.py --style csmc --N 25 --no-parallel --no-gpu --no-gradient --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE} &
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=3 sleep 5; python experiment.py --style kalman --parallel --gradient --gpu --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE}

JAX_PLATFORM_NAME=cpu sleep 5; python experiment.py --style csmc --N 25 --no-parallel --no-gpu --gradient --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE} &
JAX_PLATFORM_NAME=cpu sleep 5; python experiment.py --style csmc-guided --N 25 --no-parallel --no-gpu --no-gradient --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE} &
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=3 sleep 5; python experiment.py --style csmc --N 25 --parallel --gpu --gradient --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE}

JAX_PLATFORM_NAME=cpu sleep 5; python experiment.py --style csmc-guided --N 25 --no-parallel --no-gpu --gradient --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE} &
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=3 sleep 5; python experiment.py --style csmc --N 25 --parallel --gpu --no-gradient --target-alpha 0.5 --precision double --batch-size ${BATCH_SIZE}