# GestureDetector
Gesture detector with yolov3

## Usage

**Training**

```bash
python train.py --pretrained_weights weights/darknet53.conv.74
```

**Testing**

```bash
python test.py --weights_path weights/yolov3.weights
```

**Detection**

```bash
$ python detect.py --image_folder data/samples/ --weights_path weights/yolov3.weights
```

## Reference

- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3)