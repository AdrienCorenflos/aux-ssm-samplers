
for D in 8 16; do
  JAX_PLATFORMS=CUDA_VISIBLE_DEVICES=0 python experiment.py --style kalman --D $D --parallel --gpu --target-alpha 0.5&
  sleep 5
  JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 $T --D $D --parallel --gpu --gradient --target-alpha 0.25&
  JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc --N 25 --D $D --parallel --gpu --no-gradient  --target-alpha 0.25
  sleep 5
done