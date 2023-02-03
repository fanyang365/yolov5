#!/usr/bin/env bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
#CUDA_VISIBLE_DEVICES，选择GPU ID，从0开始编号
export CUDA_VISIBLE_DEVICES="0"
#export CUDA_VISIBLE_DEVICES="2,3,4,5"

# helmet
# train
#--weights 指定pt模型，通常用上一次的模型进行增量训练。
#--batch-size 根据GPU数量和能力进行调整，16~128
nohup python train.py --data helmet.yaml  --weights 'yolov5s.pt' --cfg yolov5s.yaml --batch-size 8 --epochs 20 >train.out 2>&1 &


# pt detect
#python detect.py --weights runs/train/exp6/weights/best.pt --source data/helmet

# evaluation
#python eval.py --weight runs/train/exp6/weights/best.pt --data data/helmet_VOC_val.yaml

# pt2pb
#python export_tf1.py --checkpoint best.pt --outName helmet_yolov5_official  --actType silu

# pb detect
#python tf_det.py  --nc 3
#omg --model models/helmet_yolov5_official.pb --framework 3 --output models/helmet_yolov5 --input_shape "input.0:1,640,640,3" --out_nodes "output.0:0;output.1:0;output.2:0" --input_format NHWC --insert_op_conf /root/hv_yolov3_aipp_640.conf  --fp16_high_prec 1

# ir_dataset
# train
#nohup python train.py --hyp data/hyps/hyp.scratch-high.yaml --data ir_VOC.yaml  --weights 'runs/train/exp7/weights/best.pt' --cfg yolov5x.yaml --batch-size 46 --workers 16 --local_rank 0 1>/home/ubun/dyr/projects/yolov5/runs/train/train.out 2>&1 &


# fire_and_smoke dataset
# train
#nohup python train.py --hyp data/hyps/hyp.scratch-low.yaml --data fire_smoke_VOC.yaml  --weights 'yolov5s.pt' --cfg yolov5s.yaml --batch-size 32 --workers 8 --local_rank 0 1>/home/ubun/dyr/projects/yolov5/runs/train/train.out 2>&1 &
# pt2pb
# python export_tf1.py --checkpoint fire_smoke_yolov5s_P-R-1_f-acc-0.63_s-acc-0.53.pt --outName fire_smoke_yolov5  --actType silu
#
# python tf_det.py  --nc 2 --input data/fire_smoke --modelPath runs/fire_smoke_yolov5s.pb


# floating
#nohup python train.py --hyp data/hyps/hyp.scratch-low.yaml --data floating.yaml  --weights 'yolov5s.pt' --cfg yolov5s.yaml --batch-size 32 --workers 8  --epochs 230 --local_rank 0 1>/home/ubun/dyr/projects/yolov5/runs/train/train.out 2>&1 &
#  python detect.py --weight floating_yolov5s.pt --source data/floating
# pt2pb
# python export_tf1.py --checkpoint floating_yolov5s.pt --outName floating_yolov5  --actType silu
#
# python tf_det.py  --nc 5 --input data/floating --modelPath runs/floating_yolov5.pb --labels /mnt/space-3/dyr_dataset/hyperpressure/floating/labels.txt
# 模型地址：/home/dyr/workspace/projects/yolov5/runs/train/exp15
# to atlas
# omg --model models/floating_yolov5.pb --framework 3 --output models/floating_yolov5 --input_shape "input.0:1,640,640,3" --out_nodes "output.0:0;output.1:0;output.2:0" --input_format NHWC --insert_op_conf /root/hv_yolov3_aipp_640.conf
# anchor [3,3, 10,6, 16,10, 19,17, 35,19, 38,32, 69,35, 110,57, 266,142]

# smoking
# train
# python train.py --hyp data/hyps/hyp.scratch-low.yaml --data smoking.yaml  --weights 'yolov5s.pt' --cfg yolov5s.yaml --batch-size 8 --workers 8 --local_rank 0 --epochs 9
# pt2pb
# python export_tf1.py --checkpoint smoking_yolov5s.pt --outName runs/smoking_yolov5  --actType silu
#
# python tf_det.py  --nc 1 --input data/smoking --modelPath runs/smoking_yolov5.pb


#hyperpressure

# python train.py --hyp data/hyps/hyp.scratch-low.yaml --data hyperpressure.yaml  --weights 'yolov5s.pt' --cfg yolov5s.yaml --batch-size 8 --workers 8 --local_rank 0 --epochs 55
# python export_tf1.py --checkpoint hyperpressure_yolov5.pt --outName hyperpressure_yolov5  --actType silu
# python tf_det.py  --nc 6 --input data/hyperpressure --modelPath runs/hyperpressure_yolov5.pb
# omg --model models/hyperpressure_yolov5.pb --framework 3 --output models/hyperpressure_yolov5 --input_shape "input.0:1,640,640,3" --out_nodes "output.0:0;output.1:0;output.2:0" --input_format NHWC --insert_op_conf /root/hv_yolov3_aipp_640.conf
