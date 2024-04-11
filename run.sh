# hyper-parameter tuning
python -u search/search.py search/yamls/yelp_lightgcn.yaml

# graph augment
python -u augment.py --dataset=gowalla --model=jgcf --alpha=0.6 --gpu_id=0 --norm

# train
python -u main.py --dataset=gowalla --model=bpr
python -u main.py --dataset=gowalla --model=lightgcn
python -u main.py --dataset=gowalla --model=currigcn
python -u main.py --dataset=gowalla --model=randgcn
python -u main.py --dataset=gowalla --model=prepgcn
python -u main.py --dataset=gowalla --model=ngcf
python -u main.py --dataset=gowalla --model=ultragcn
python -u main.py --dataset=gowalla --model=cagcn
python -u main.py --dataset=gowalla --model=gde
python -u main.py --dataset=gowalla --model=pgsp
python -u main.py --dataset=gowalla --model=svd_gcn
python -u main.py --dataset=gowalla --model=gtn
python -u main.py --dataset=gowalla --model=apegnn
python -u main.py --dataset=gowalla --model=lgcn
python -u main.py --dataset=gowalla --model=freqgcn
python -u main.py --dataset=gowalla --model=gfcf
python -u main.py --dataset=gowalla --model=stablegcn

python -u main.py --dataset=sports --model=lightgcn

python -u main.py --dataset=ml-1m --model=bpr
python -u main.py --dataset=ml-1m --model=lightgcn
python -u main.py --dataset=ml-1m --model=pgsp
python -u main.py --dataset=ml-1m --model=svd_gcn
python -u main.py --dataset=ml-1m --model=gde
python -u main.py --dataset=ml-1m --model=jgcf
python -u main.py --dataset=ml-1m --model=freqgcn
python -u main.py --dataset=ml-1m --model=gfcf

python -u main.py --dataset=books --model=lightgcn
python -u main.py --dataset=books --model=bpr
python -u main.py --dataset=books --model=pgsp

python -u main.py --dataset=yelp --model=lightgcn
python -u main.py --dataset=yelp --model=bpr
python -u main.py --dataset=yelp --model=pgsp
python -u main.py --dataset=yelp --model=jgcf
python -u main.py --dataset=yelp --model=apegnn
python -u main.py --dataset=yelp --model=gfcf