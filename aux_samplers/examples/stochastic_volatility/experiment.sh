for T in 50 100 200; do
  for D in 5 30 100; do
    python experiment.py --style kalman --T $T --D $D --parallel --gpu
    python experiment.py --style kalman --T $T --D $D --no-parallel --no-gpu
    for N in 25 100 500; do
      for style in csmc csmc-guided; do
        python experiment.py --style $style --N $N --T $T --D $D --no-parallel --no-gpu --gradient
        python experiment.py --style $style --N $N --T $T --D $D --no-parallel --no-gpu --no-gradient
        python experiment.py --style $style --N $N --T $T --D $D --parallel --gpu --gradient
        python experiment.py --style $style --N $N --T $T --D $D --parallel --gpu --no-gradient
      done
    done
  done
done