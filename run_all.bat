
@echo off
python train_cloak.py --game cs2       --model yolov5n --n_iter 100
python train_cloak.py --game cs2       --model nanodet --n_iter 100
python train_cloak.py --game cs2       --model rtdetr  --n_iter 30
python train_cloak.py --game cf        --model yolov5n --n_iter 100
python train_cloak.py --game cf        --model nanodet --n_iter 100
python train_cloak.py --game cf        --model rtdetr  --n_iter 30
python train_cloak.py --game valorant  --model yolov5n --n_iter 100
python train_cloak.py --game valorant  --model nanodet --n_iter 100
python train_cloak.py --game valorant  --model rtdetr  --n_iter 30
python train_cloak.py --game overwatch --model yolov5n --n_iter 100
python train_cloak.py --game overwatch --model nanodet --n_iter 100
python train_cloak.py --game overwatch --model rtdetr  --n_iter 30
echo ALL DONE!
pause