#!/bin/bash

#for i in $(seq 1 5)
#do
#	python3 -u main.py --mode train --scale ${i} --train-path dataset/train --epochs 30 --steps 400
#	cp model/stream_${i}.30.h5 model/stream_${i}.h5
#done
#python3 -u main.py --mode train --scale 1 --train-path dataset/train --epochs 30 --steps 400
#cp model/stream_1.30.h5 model/stream_1.h5

python3 -u main.py --mode train --scale 2 --train-path dataset/train --epochs 20 --steps 400
cp model/stream_2.20.h5 model/stream_2.h5

python3 -u main.py --mode train --scale 3 --train-path dataset/train --epochs 20 --steps 400
cp model/stream_3.20.h5 model/stream_3.h5

python3 -u main.py --mode train --scale 4 --train-path dataset/train --epochs 20 --steps 400
cp model/stream_4.20.h5 model/stream_4.h5

python3 -u main.py --mode train --scale 5 --train-path dataset/train --epochs 20 --steps 400 --batch-size 2
cp model/stream_5.20.h5 model/stream_5.h5

exit 0
