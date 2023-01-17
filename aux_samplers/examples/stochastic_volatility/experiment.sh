for T in 50 100 200; do
  for D in 5 30 100; do
    for style in kalman-1 kalman-2; do
      JAX_PLATFORMS=cpu python experiment.py --style $style --T $T --D $D --no-parallel --no-gpu
    done
    for N in 25 100; do
      for style in csmc csmc-guided; do
        JAX_PLATFORMS=cpu python experiment.py --style $style --N $N --T $T --D $D --no-parallel --no-gpu --gradient
        JAX_PLATFORMS=cpu python experiment.py --style $style --N $N --T $T --D $D --no-parallel --no-gpu --no-gradient
      done
    done
  done
done


for T in 50 100 200; do
  for D in 5 30; do
    for style in kalman-1 kalman-2; do
      JAX_PLATFORMS=CUDA_VISIBLE_DEVICES=0 python experiment.py --style $style --T $T --D $D --parallel --gpu
    done
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 python experiment.py --style csmc --N 25 --T $T --D $D --parallel --gpu --gradient &
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 python experiment.py --style csmc --N 25 --T $T --D $D --parallel --gpu --no-gradient
  done
done