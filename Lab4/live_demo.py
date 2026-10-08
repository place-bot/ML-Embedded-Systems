"""Camera -> frozen YOLO -> browser boxes, scores and frame-matched saves."""
import argparse
import base64
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import signal
import threading
import time
from urllib.parse import urlparse, parse_qs

from common import CONFIG, ROOT, annotate, configure, detect, load_model, sha256, patch_source_name
from camera_stream import Camera
from results_io import Results


def patch_metadata(folder, reference=False):
    folder = Path(folder)
    run = json.loads((folder/'run.json').read_text())
    if sha256(folder/'learned.png') != run['patch_sha256'] or sha256(folder/'random.png') != run['random_sha256']:
        raise ValueError('Patch files differ from their training record.')
    if run['weights_sha256'] != sha256(ROOT/'data/yolov2.weights'):
        raise ValueError('The patch was trained for different detector weights.')
    return {'patch_source': patch_source_name(run.get('patch_source', 'reference' if reference else 'trained')), 'training': run}


class Pipeline:
    def __init__(self, model, camera):
        self.camera = camera
        self.model = model
        self.condition = threading.Condition()
        self.frames = OrderedDict()
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        sequence = 0
        previous = None
        try:
            while not self.stop.is_set():
                item = self.camera.next(sequence)
                if item is None:
                    return
                sequence, captured, captured_unix, image, raw = item
                started = time.monotonic()
                result = detect(self.model, image)
                elapsed = (time.monotonic()-started)*1000
                now = time.monotonic()
                fps = 0.0 if previous is None else 1/(now-previous)
                previous = now
                footer = f'frame {sequence} | {fps:.1f} updates/s | {elapsed:.0f} ms'
                if self.camera.image_path:
                    footer = f'IMAGE REPLAY | frame {sequence}'
                annotated = annotate(image, result, footer)
                encoded = io.BytesIO()
                annotated.save(encoded, 'JPEG', quality=85)
                frame = {'id': sequence, 'captured': captured, 'captured_unix': captured_unix,
                         'raw': raw, 'annotated': encoded.getvalue(), 'result': result,
                         'inference_ms': elapsed, 'update_fps': fps}
                with self.condition:
                    self.frames[sequence] = frame
                    while len(self.frames) > 40:
                        self.frames.popitem(last=False)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def newest(self, after=0):
        with self.condition:
            self.condition.wait_for(lambda: self.error or self.stop.is_set() or
                                    (self.frames and next(reversed(self.frames)) != after), 2)
            if self.error:
                raise RuntimeError(self.error)
            if not self.frames or next(reversed(self.frames)) == after:
                return None
            frame = self.frames[next(reversed(self.frames))]
            if time.monotonic()-frame['captured'] > CONFIG['frame_max_age']:
                raise RuntimeError('Camera frames are stale. Check the Pi terminal.')
            return frame

    def get(self, frame_id):
        with self.condition:
            if self.error:
                raise ValueError(self.error)
            frame = self.frames.get(frame_id)
            if frame is None or time.monotonic()-frame['captured'] > CONFIG['frame_max_age']:
                raise ValueError('That frame has expired. Wait for the live picture and save again.')
            return frame

    def close(self):
        self.stop.set()
        self.camera.close()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=10)


def make_handler(pipeline, results, token, page, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed_host(self):
            return self.headers.get('Host') in [f'127.0.0.1:{port}', f'localhost:{port}']

        def send(self, status, content, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def json(self, status, data):
            self.send(status, json.dumps(data, allow_nan=False).encode())

        def do_GET(self):
            if not self.allowed_host():
                return self.json(403, {'error': 'Use localhost through the SSH tunnel.'})
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/':
                    return self.send(200, page, 'text/html; charset=utf-8')
                if parsed.path == '/frame':
                    after = int(parse_qs(parsed.query).get('after', ['0'])[0])
                    frame = pipeline.newest(after)
                    if frame is None:
                        return self.json(200, {'waiting': True})
                    with results.lock:
                        saved = results.status()
                    return self.json(200, {'id': frame['id'], 'image': base64.b64encode(frame['annotated']).decode(),
                                          'score': frame['result']['max_person_score'],
                                          'detected': frame['result']['person_detected'], 'saved': saved,
                                          'update_fps': frame['update_fps'], 'inference_ms': frame['inference_ms']})
                if parsed.path == '/download':
                    return self.send(200, results.download(), 'application/zip')
                self.json(404, {'error': 'Not found'})
            except (ValueError, RuntimeError) as exc:
                self.json(503, {'error': str(exc)})
            except ConnectionError:
                pass

        def do_POST(self):
            if not self.allowed_host() or urlparse(self.path).path != '/save':
                return self.json(403, {'error': 'Unsupported request.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 4096:
                    raise ValueError('Invalid save request.')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Invalid save request.')
                if not secrets.compare_digest(str(data.get('token', '')), token):
                    return self.json(403, {'error': 'Refresh this page before saving.'})
                frame = pipeline.get(int(data['frame_id']))
                saved = results.save(data['condition'], frame, data.get('note', ''))
                self.json(200, {'saved': saved, 'frame_id': frame['id']})
            except (ValueError, KeyError, TypeError) as exc:
                self.json(400, {'error': str(exc)})
            except ConnectionError:
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--patch', type=Path, default=ROOT/'results/patch')
    parser.add_argument('--reference', action='store_true', help='Use the supplied reference print pair.')
    parser.add_argument('--image', type=Path, help='Software check only; visibly labels evidence IMAGE REPLAY.')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.camera < 0:
        parser.error('Use a nonnegative camera number and a port between 1024 and 65535.')
    def stop_requested(signum, frame):
        raise KeyboardInterrupt
    for name in ['SIGTERM', 'SIGHUP']:
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_requested)
    configure()
    model = load_model()
    patch_folder = ROOT/'assets/reference' if args.reference else args.patch
    metadata = {'config': CONFIG, 'source': 'image-replay' if args.image else 'camera',
                'camera': args.camera, **patch_metadata(patch_folder, args.reference)}
    output = (args.output or ROOT/'results/live'/datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = Results(output, metadata)
    token = secrets.token_urlsafe(24)
    page = (ROOT/'web/index.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
    label = 'IMAGE REPLAY - software check only' if args.image else 'Raspberry Pi camera'
    page = page.replace('__SOURCE__', label).replace('__PATCH_SOURCE__', metadata['patch_source'])
    camera = None
    pipeline = None
    server = None
    try:
        camera = Camera(args.camera, output/'camera.log', args.image)
        pipeline = Pipeline(model, camera)
        server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                     make_handler(pipeline, results, token, page.encode(), args.port))
        server.daemon_threads = True
        print(f'Open http://127.0.0.1:{args.port} through your SSH tunnel.', flush=True)
        print(f'Saving to {output}. Ctrl+C stops the camera.', flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print('\nStopping camera.', flush=True)
    finally:
        if pipeline:
            pipeline.close()
        elif camera:
            camera.close()
        if server:
            server.server_close()


if __name__ == '__main__':
    main()
