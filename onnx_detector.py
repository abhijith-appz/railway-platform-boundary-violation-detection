"""
Torch-free YOLOv8 backend (onnxruntime + OpenCV) with a lightweight IoU tracker.
Handy for edge devices / servers where installing PyTorch is not practical.

Export once on a machine with ultralytics:   yolo export model=yolov8s.pt format=onnx imgsz=640
Then:  python detector.py --video v.mp4 --auto-line --backend onnx --weights yolov8s.onnx
"""
from __future__ import annotations

import cv2
import numpy as np

from detector import Detection


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class OnnxYoloTracker:
    def __init__(self, weights="yolov8n.onnx", conf=0.35, iou=0.5, max_age=15, device="cpu"):
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(weights, providers=prov)
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.conf, self.iou, self.max_age = conf, iou, max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def _infer(self, frame):
        h, w = frame.shape[:2]
        r = self.size / max(h, w)
        nw, nh = int(w * r), int(h * r)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        out = self.sess.run(None, {self.name: blob})[0][0].T          # (8400, 84)
        scores = out[:, 4]                                             # class 0 = person
        keep = scores > self.conf
        out, scores = out[keep], scores[keep]
        if not len(out):
            return []
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([(cx - bw / 2) / r, (cy - bh / 2) / r, bw / r, bh / r], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf, self.iou)
        res = []
        for i in np.array(idx).flatten():
            x, y, bw_, bh_ = boxes[i]
            res.append((int(max(x, 0)), int(max(y, 0)), int(min(x + bw_, w - 1)),
                        int(min(y + bh_, h - 1)), float(scores[i])))
        return res

    def __call__(self, frame):
        dets = self._infer(frame)
        for t in self.tracks.values():
            t["age"] += 1
        out, used = [], set()
        for x1, y1, x2, y2, c in sorted(dets, key=lambda d: -d[4]):
            best, bi = 0.25, None
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                v = _iou((x1, y1, x2, y2), t["box"])
                if v > best:
                    best, bi = v, tid
            if bi is None:
                bi, self.next_id = self.next_id, self.next_id + 1
            used.add(bi)
            self.tracks[bi] = {"box": (x1, y1, x2, y2), "age": 0}
            out.append(Detection(bi, (x1, y1, x2, y2), c))
        self.tracks = {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}
        return out
