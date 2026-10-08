"""Save frame-matched observations and the four-condition comparison."""
import csv
import io
import json
from pathlib import Path
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageDraw, ImageFont
from common import write_json

CONDITIONS = ['Clean', 'Random', 'Learned', 'Explore']


class Results:
    def __init__(self, output, metadata):
        self.output = Path(output)
        self.metadata = metadata
        self.saved = {}
        self.counter = 0
        self.lock = threading.RLock()
        write_json(self.output/'session.json', metadata)

    def status(self):
        return {label: {'max_person_score': row['max_person_score'],
                        'person_detected': row['person_detected'], 'note': row['note']}
                for label, row in self.saved.items()}

    def save(self, condition, frame, note):
        if condition not in CONDITIONS:
            raise ValueError('Choose Clean, Random, Learned or Explore.')
        if not isinstance(note, str) or len(note) > 160:
            raise ValueError('Keep the note within 160 characters.')
        if condition == 'Explore' and not note.strip():
            raise ValueError('Describe the one change made for Explore.')
        if condition == 'Clean' and not frame['result']['person_detected']:
            raise ValueError('First arrange a clear person detection for Clean.')
        with self.lock:
            if condition != 'Clean' and 'Clean' not in self.saved:
                raise ValueError('Save Clean first.')
            if condition == 'Explore' and 'Learned' not in self.saved:
                raise ValueError('Save Learned before Explore.')
            self.counter += 1
            name = f'{self.counter:03d}-{condition.lower()}'
            folder = self.output/name
            folder.mkdir()
            (folder/'raw.jpg').write_bytes(frame['raw'])
            (folder/'annotated.jpg').write_bytes(frame['annotated'])
            row = {'condition': condition, 'frame_id': frame['id'], 'note': note.strip(),
                   'captured_unix': frame['captured_unix'], 'saved_unix': time.time(),
                   'inference_ms': frame['inference_ms'], 'update_fps': frame['update_fps'],
                   'folder': name, 'source': self.metadata['source'],
                   'patch_source': self.metadata['patch_source'], **frame['result']}
            write_json(folder/'observation.json', row)
            self.saved[condition] = row
            self._summarize()
            return self.status()

    def _summarize(self):
        rows = [self.saved[label] for label in CONDITIONS if label in self.saved]
        write_json(self.output/'comparison.json', rows)
        columns = ['condition', 'person_detected', 'max_person_score', 'inference_ms', 'update_fps',
                   'note', 'patch_source', 'source', 'folder']
        with (self.output/'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        canvas = Image.new('RGB', (1280, 1040), 'white')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=20)
        for index, label in enumerate(CONDITIONS):
            x, y = (index%2)*640, (index//2)*520
            row = self.saved.get(label)
            if row:
                with Image.open(self.output/row['folder']/'annotated.jpg') as image:
                    image.thumbnail((640, 480))
                    canvas.paste(image, (x, y+40))
                caption = f'{label}: {row["note"]}' if label == 'Explore' else label
                if self.metadata['source'] != 'camera':
                    caption = 'IMAGE REPLAY / ' + caption
                draw.text((x+8, y+8), caption[:72], fill='black', font=font)
            else:
                draw.text((x+8, y+8), f'{label}: not saved', fill='black', font=font)
        canvas.save(self.output/'evidence.jpg', quality=90)

    def download(self):
        with self.lock:
            buffer = io.BytesIO()
            with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
                for path in sorted(self.output.rglob('*')):
                    if path.is_file() and path.suffix in ['.jpg', '.json', '.csv']:
                        archive.write(path, path.relative_to(self.output).as_posix())
            return buffer.getvalue()
