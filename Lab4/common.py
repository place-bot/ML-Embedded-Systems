"""Course settings and YOLOv2 inference shared by the camera and previews."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'course_config.json').read_text())
WEIGHTS_SHA256 = 'd9945162ed6f54ce1a901e3ec537bdba4d572ecae7873087bd730e5a7942df3f'
ANCHORS = np.array([[.57273,.677385],[1.87446,2.06253],[3.33843,5.47434],
                    [7.88282,3.52778],[9.77052,9.16828]], dtype=np.float32)


def configure():
    cv2.setNumThreads(CONFIG['threads'])
    np.random.seed(CONFIG['seed'])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def checked_weights():
    path = ROOT/'data/yolov2.weights'
    if not path.exists() or sha256(path) != WEIGHTS_SHA256:
        raise ValueError('Run python prepare_assets.py to download the course weights.')
    return path


def patch_source_name(source):
    """Accept source labels from earlier exports."""
    return {'student': 'trained', 'author-reference': 'reference'}.get(source, source)


class LegacyReorg:
    """Preserve the original PyTorch spatial/channel order in OpenCV."""
    def __init__(self, params, blobs):
        pass

    def getMemoryShapes(self, inputs):
        b,c,h,w = inputs[0]
        return [[b,4*c,h//2,w//2]]

    def forward(self, inputs):
        x = inputs[0]
        b,c,h,w = x.shape
        x = x.reshape(b,c,h//2,2,w//2,2).transpose(0,3,5,1,2,4)
        return [x.copy().reshape(b,4*c,h//2,w//2)]


cv2.dnn_registerLayer('Reorg', LegacyReorg)


def load_model():
    net = cv2.dnn.readNetFromDarknet(str(ROOT/'assets/cfg/yolov2.cfg'), str(checked_weights()))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def letterbox(image):
    image = image.convert('RGB')
    side = max(image.size)
    left, top = (side-image.width)//2, (side-image.height)//2
    canvas = Image.new('RGB', (side, side), (127,127,127))
    canvas.paste(image, (left, top))
    canvas = canvas.resize((CONFIG['image_size'],)*2, Image.Resampling.BILINEAR)
    array = np.asarray(canvas, dtype=np.float32).transpose(2,0,1).copy()/255
    return array, (CONFIG['image_size']/side, left, top)


def raw_predictions(net, array):
    net.setInput(np.ascontiguousarray(array, dtype=np.float32))
    return net.forward('conv_30')


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-80,80)))


def decode(raw, image_size, mapping):
    p = raw[0].reshape(5,85,raw.shape[-2],raw.shape[-1])
    h,w = p.shape[-2:]
    obj = sigmoid(p[:,4]).reshape(-1)
    classes = p[:,5:].argmax(axis=1).reshape(-1)
    maximum = float(obj[classes == 0].max(initial=0))
    yy,xx = np.mgrid[:h,:w]
    cx = (sigmoid(p[:,0])+xx)/w*CONFIG['image_size']
    cy = (sigmoid(p[:,1])+yy)/h*CONFIG['image_size']
    bw = np.exp(np.clip(p[:,2],-20,20))*ANCHORS[:,0,None,None]/w*CONFIG['image_size']
    bh = np.exp(np.clip(p[:,3],-20,20))*ANCHORS[:,1,None,None]/h*CONFIG['image_size']
    boxes = np.stack((cx-bw/2,cy-bh/2,bw,bh),axis=-1).reshape(-1,4)
    selected = np.flatnonzero(obj >= CONFIG['confidence'])
    # Original demonstration: objectness threshold, then class-agnostic NMS.
    kept = cv2.dnn.NMSBoxes(boxes[selected].tolist(), obj[selected].tolist(),
                           CONFIG['confidence'], CONFIG['nms_iou']) if len(selected) else []
    scale,left,top = mapping
    detections = []
    for i in np.asarray(kept,dtype=int).reshape(-1):
        j = selected[i]
        if classes[j] != 0:
            continue
        x,y,bw,bh = boxes[j]
        coords = [x/scale-left,y/scale-top,(x+bw)/scale-left,(y+bh)/scale-top]
        coords = [max(0,min(float(v),image_size[k%2])) for k,v in enumerate(coords)]
        detections.append({'label':'person','confidence':float(obj[j]),'xyxy':coords})
    return {'max_person_score':maximum,'person_detected':bool(detections),'detections':detections}


def detect(net, image):
    array,mapping = letterbox(image)
    return decode(raw_predictions(net,array[None]),image.size,mapping)


def annotate(image, result, footer=''):
    image = image.copy().convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    for box in result['detections']:
        xy = tuple(round(v) for v in box['xyxy'])
        draw.rectangle(xy, outline=(40, 230, 100), width=3)
        text = f"person {box['confidence']:.2f}"
        x, y = xy[0], max(0, xy[1]-24)
        draw.rectangle((x, y, min(image.width, x+140), y+24), fill=(0, 35, 15))
        draw.text((x+3, y+2), text, fill='white', font=font)
    text = f"person objectness {result['max_person_score']:.3f} | threshold {CONFIG['confidence']}"
    draw.rectangle((0, image.height-50, image.width, image.height), fill=(20, 20, 20))
    draw.text((8, image.height-47), text, fill='white', font=font)
    draw.text((8, image.height-25), footer, fill='white', font=font)
    return image
