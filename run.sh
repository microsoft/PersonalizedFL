python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.1
python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1
python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --mu 0.1
python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.1 --model_momentum 0.3
python main.py --alg metafed --dataset medmnist --iters 300 --wk_iters 1 --threshold 1.1 --nosharebn --non_iid_alpha 0.1

python main.py --alg base --dataset medmnist --iters 100 --wk_iters 3 --non_iid_alpha 0.01
python main.py --alg fedavg --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01
python main.py --alg fedprox --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --mu 0.01
python main.py --alg fedbn --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01
python main.py --alg fedap --dataset medmnist --iters 300 --wk_iters 1 --non_iid_alpha 0.01 --model_momentum 0.3
python main.py --alg metafed --dataset medmnist --iters 50 --wk_iters 6 --threshold 1.1 --non_iid_alpha 0.01