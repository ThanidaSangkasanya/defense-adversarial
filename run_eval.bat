@echo off
python train_cloak.py --game cs2       --model yolov5n
python train_cloak.py --game cs2       --model nanodet
python train_cloak.py --game cs2       --model rtdetr
python train_cloak.py --game cf        --model yolov5n
python train_cloak.py --game cf        --model nanodet
python train_cloak.py --game cf        --model rtdetr
python train_cloak.py --game valorant  --model yolov5n
python train_cloak.py --game valorant  --model nanodet
python train_cloak.py --game valorant  --model rtdetr
python train_cloak.py --game overwatch --model yolov5n
python train_cloak.py --game overwatch --model nanodet
python train_cloak.py --game overwatch --model rtdetr
echo ALL DONE!
pause