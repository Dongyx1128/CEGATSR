# python mains.py train --dataset_name 'Chikusei' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 32 --n_scale 2 --gpus "0,1"
# python mains.py test --dataset_name 'Chikusei' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 2 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Chikusei' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 32 --n_scale 4 --gpus "0,1"
# python mains.py test --dataset_name 'Chikusei' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 4 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Chikusei' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 32 --n_scale 8 --gpus "0,1"
# python mains.py test --dataset_name 'Chikusei' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 8 --cuda 1 --gpus "0,1"

# python mains.py train --dataset_name 'Cave' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 64 --n_scale 2 --gpus "0,1"
# python mains.py test --dataset_name 'Cave' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 2 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Cave' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 64 --n_scale 4 --gpus "0,1"
# python mains.py test --dataset_name 'Cave' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 4 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Cave' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 64 --n_scale 8 --gpus "0,1"
# python mains.py test --dataset_name 'Cave' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 8 --cuda 1 --gpus "0,1"

# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 32 --n_scale 2 --gpus "0,1"
# python mains.py test --dataset_name 'Pavia' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 2 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 4 --gpus "0,1"
# python mains.py test --dataset_name 'Pavia' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 128 --n_scale 4 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 64 --n_scale 8 --gpus "0,1"
# python mains.py test --dataset_name 'Pavia' --n_blocks 6 --n_subs 4 --n_ovls 1 --out_feats 64 --n_scale 8 --cuda 1 --gpus "0,1"


# efficiency
python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 60 --batch_size 16 --n_subs 4 --n_ovls 1 --out_feats 64 --n_scale 8 --gpus "0,1"



# python mains.py test --cuda 1 --gpus "0,1"