# Sources and changes

Based on Thys, Van Ranst and Goedeme, "Fooling automated surveillance cameras: adversarial patches to attack person detection," CVPR Workshops 2019: https://arxiv.org/abs/1904.08653.

- Code and reference patch: https://gitlab.com/EAVISE/adversarial-yolo, revision 0665939cc733ade7289b2281dad80a89c6e0a2fe. Detector layers, weight loading and patch transforms are adapted from this project, which builds on marvis/pytorch-yolo2. See `licenses/adversarial-yolo-MIT.txt`.
- Pretrained YOLOv2 COCO weights: https://data.pjreddie.com/files/yolov2.weights, downloaded separately and checked against SHA-256 d9945162ed6f54ce1a901e3ec537bdba4d572ecae7873087bd730e5a7942df3f.
- INRIA Person data (Dalal and Triggs): ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar. The fixed subset contains 64 Train/pos images and 16 Test/pos images with bounding-box annotations, selected by hash order and minimum visible-person height, then square-padded and resized to 416 pixels. Source hashes and splits are in `assets/inria/manifest.json`.
- `assets/example/` contains the original project's published example and label, separate from the INRIA subset.
- `assets/reference/learned.png` is the published `object_score.png`; `run.json` records its source and checksum. Its Random control was generated independently.

This adaptation adds PyTorch 2.x support, explicit interpolation, a fixed training budget, frozen-detector checks, printable tiles, Colab integration and a camera interface.
Training starts from a gray 300-pixel patch and retains the objectness, printable-color and total-variation loss terms.
It uses a small data subset instead of the paper's full training schedule.

A custom OpenCV Reorg layer preserves the original PyTorch tensor order.
Inference applies an objectness threshold of 0.40 and class-agnostic NMS at 0.40, then displays person predictions.
