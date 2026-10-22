"""
Railway Platform Boundary Violation Detector
============================================
Detects people who step past the yellow safety line into the danger zone
(the strip between the line and the track edge) using YOLO + tracking.

Usage:
    python detector.py --video platform.mp4 --zone zone.json
    python detector.py --video platform.mp4 --zone zone.json --show --webhook https://example/alert
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

PERSON_CLASS = 0  # COCO class id for "person"

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float


@dataclass
class TrackState:
    frames_inside: int = 0
    alerted: bool = False
    last_alert_ts: float = -1e9
    in_zone_now: bool = False


@dataclass
class ViolationEvent:
    timestamp: str          # wall-clock time of processing
    video_time: str         # mm:ss.s inside the video
    frame: int
    track_id: int
    confidence: float
    snapshot: str


# --------------------------------------------------------------------------- #
# Zone helpers
# --------------------------------------------------------------------------- #
def load_zone(path: str | Path, width: int, height: int) -> np.ndarray:
    """Load the danger-zone polygon. Coordinates are stored normalised (0-1)
    so the same file works for any video resolution."""
    data = json.loads(Path(path).read_text())
    pts = np.array(data["danger_zone"], dtype=np.float32)
    if data.get("normalized", True):
        pts = pts * np.array([width, height], dtype=np.float32)
    if len(pts) < 3:
        raise ValueError("danger_zone needs at least 3 points")
    return pts.astype(np.int32)


def save_zone(path: str | Path, points_px: Iterable[tuple[int, int]], width: int, height: int) -> None:
    pts = [[round(x / width, 5), round(y / height, 5)] for x, y in points_px]
    Path(path).write_text(json.dumps({"normalized": True, "danger_zone": pts}, indent=2))


def foot_point(box: tuple[int, int, int, int]) -> tuple[int, int]:
    """Bottom-centre of the bounding box ~ where the person stands.
    Using the feet (not the box centre) is what makes the check accurate
    on a perspective camera view."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int(y2))


def in_zone(zone: np.ndarray, point: tuple[int, int]) -> bool:
    return cv2.pointPolygonTest(zone.reshape(-1, 1, 2), point, False) >= 0


# --------------------------------------------------------------------------- #
# YOLO wrapper (kept separate so the logic below can be tested without torch)
# --------------------------------------------------------------------------- #
class YoloTracker:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 imgsz: int = 960, device: Optional[str] = None,
                 tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO  # imported lazily
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device, self.tracker = conf, imgsz, device, tracker
        # Some short clips / CPU runs return detections before ByteTrack has
        # assigned IDs. Keep a lightweight IoU-based fallback so detections
        # are not silently discarded just because boxes.id is None.
        self._previous: list[tuple[int, tuple[int, int, int, int]]] = []
        self._next_id = 1

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        return inter / max(area_a + area_b - inter, 1)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.track(
            frame, persist=True, classes=[PERSON_CLASS], conf=self.conf,
            imgsz=self.imgsz, device=self.device, tracker=self.tracker, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            self._previous = []
            return []
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        raw_ids = res.boxes.id
        if raw_ids is not None:
            ids = raw_ids.cpu().numpy().astype(int)
            detections = [Detection(int(i), tuple(map(int, b)), float(c))
                          for b, i, c in zip(boxes, ids, confs)]
            self._previous = [(d.track_id, d.box) for d in detections]
            return detections

        # Fallback matching: greedily match each box to the previous frame by
        # IoU. It is intended for short/no-ID clips, not dense multi-object MOT.
        unused = set(range(len(self._previous)))
        detections = []
        current = []
        for box, conf in zip(boxes, confs):
            box_t = tuple(map(int, box))
            best_idx, best_iou = None, 0.15
            for j in unused:
                score = self._iou(box_t, self._previous[j][1])
                if score > best_iou:
                    best_idx, best_iou = j, score
            if best_idx is None:
                tid = self._next_id
                self._next_id += 1
            else:
                tid = self._previous[best_idx][0]
                unused.remove(best_idx)
            det = Detection(tid, box_t, float(conf))
            detections.append(det)
            current.append((tid, box_t))
        self._previous = current
        return detections


# --------------------------------------------------------------------------- #
# Alerting
# --------------------------------------------------------------------------- #
class Alerter:
    """Console + CSV log + snapshot + optional webhook (SMS/Telegram/control-room
    system can sit behind the webhook)."""

    def __init__(self, out_dir: Path, webhook: Optional[str] = None, beep: bool = True):
        self.out_dir = out_dir
        self.snap_dir = out_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.webhook, self.beep = webhook, beep
        self.csv_path = out_dir / "violations.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "video_time", "frame", "track_id", "confidence", "snapshot"])

    def fire(self, ev: ViolationEvent) -> None:
        print(f"\033[91m[ALERT]\033[0m Boundary violated | person #{ev.track_id} | "
              f"video time {ev.video_time} | frame {ev.frame}")
        if self.beep:
            sys.stdout.write("\a")
            sys.stdout.flush()
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ev.timestamp, ev.video_time, ev.frame,
                                    ev.track_id, f"{ev.confidence:.2f}", ev.snapshot])
        if self.webhook:
            try:
                import requests
                requests.post(self.webhook, json=ev.__dict__, timeout=3)
            except Exception as e:  # never let a network error stop detection
                print(f"[warn] webhook failed: {e}")


# --------------------------------------------------------------------------- #
# Main monitor
# --------------------------------------------------------------------------- #
class BoundaryMonitor:
    def __init__(self, detect_fn: Callable[[np.ndarray], list[Detection]],
                 zone_file: Optional[str], out_dir: str = "output",
                 confirm_frames: int = 5, cooldown_s: float = 10.0,
                 webhook: Optional[str] = None, beep: bool = True,
                 auto_line=None):
        """
        zone_file      : fixed zone JSON, or None when auto_line is used
        auto_line      : a yellow_line.YellowLineZone - finds the line every frame
        confirm_frames : consecutive frames a person must be inside the zone
                         before an alert (filters out flicker / false positives)
        cooldown_s     : minimum seconds between repeat alerts for the same person
        """
        self.detect = detect_fn
        self.zone_file, self.auto_line = zone_file, auto_line
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_frames, self.cooldown_s = confirm_frames, cooldown_s
        self.alerter = Alerter(self.out_dir, webhook, beep)
        self.tracks: dict[int, TrackState] = {}

    # -- per-frame logic ---------------------------------------------------- #
    def step(self, frame: np.ndarray, zone: np.ndarray, frame_idx: int, fps: float):
        """Returns (annotated_frame, new_events)."""
        dets = self.detect(frame)
        events: list[ViolationEvent] = []
        seen = set()

        for d in dets:
            seen.add(d.track_id)
            st = self.tracks.setdefault(d.track_id, TrackState())
            st.in_zone_now = in_zone(zone, foot_point(d.box))
            st.frames_inside = st.frames_inside + 1 if st.in_zone_now else 0

            confirmed = st.frames_inside >= self.confirm_frames
            now = time.time()
            if confirmed and (not st.alerted or now - st.last_alert_ts > self.cooldown_s):
                st.alerted, st.last_alert_ts = True, now
                events.append(self._make_event(frame, zone, d, frame_idx, fps, dets))
            if not st.in_zone_now:
                st.alerted = False  # re-arm once the person steps back

        # drop state of people who left the scene
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid].in_zone_now = False
                self.tracks[tid].frames_inside = 0

        return self._draw(frame, zone, dets), events

    def _make_event(self, frame, zone, det, frame_idx, fps, dets) -> ViolationEvent:
        snap = self._draw(frame, zone, dets)
        path = self.out_dir / "snapshots" / f"violation_id{det.track_id}_f{frame_idx}.jpg"
        cv2.imwrite(str(path), snap)
        secs = frame_idx / fps if fps else 0
        ev = ViolationEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            video_time=f"{int(secs // 60):02d}:{secs % 60:04.1f}",
            frame=frame_idx, track_id=det.track_id,
            confidence=det.conf, snapshot=str(path))
        self.alerter.fire(ev)
        return ev

    def _draw(self, frame, zone, dets) -> np.ndarray:
        out = frame.copy()
        overlay = out.copy()
        cv2.fillPoly(overlay, [zone], YELLOW)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.polylines(out, [zone], True, YELLOW, 2)

        violators = 0
        for d in dets:
            st = self.tracks.get(d.track_id)
            bad = bool(st and st.in_zone_now and st.frames_inside >= self.confirm_frames)
            violators += bad
            col = RED if bad else GREEN
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.circle(out, foot_point(d.box), 5, col, -1)
            label = f"ID {d.track_id}" + (" VIOLATION" if bad else "")
            cv2.putText(out, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        if violators:
            h, w = out.shape[:2]
            if int(time.time() * 3) % 2 == 0:  # flashing banner
                cv2.rectangle(out, (0, 0), (w, 48), RED, -1)
                cv2.putText(out, f"! BOUNDARY VIOLATION - {violators} person(s) !",
                            (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return out

    # -- video driver ------------------------------------------------------- #
    def process_video(self, source: str | int, write_video: bool = True,
                      show: bool = False,
                      progress: Optional[Callable[[int, int], None]] = None
                      ) -> list[ViolationEvent]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        zone = None if self.auto_line else load_zone(self.zone_file, w, h)

        writer = None
        if write_video:
            self.video_path = self.out_dir / "annotated.mp4"
            writer = cv2.VideoWriter(str(self.video_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        all_events: list[ViolationEvent] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.auto_line:
                    zone = self.auto_line.update(frame)
                    if zone is None:          # line not seen yet - skip, don't guess
                        if writer: writer.write(frame)
                        idx += 1
                        continue
                annotated, events = self.step(frame, zone, idx, fps)
                all_events.extend(events)
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Boundary Monitor (q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if progress:
                    progress(idx, total)
                idx += 1
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        return all_events


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Railway platform boundary violation detector")
    ap.add_argument("--video", required=True, help="video file, RTSP URL, or webcam index (0)")
    ap.add_argument("--zone", default=None, help="danger-zone JSON (see select_zone.py)")
    ap.add_argument("--auto-line", action="store_true",
                    help="detect the yellow line automatically each frame (no zone file needed)")
    ap.add_argument("--track-side", choices=["left", "right"], default="left",
                    help="with --auto-line: which side of the yellow line the rails are on")
    ap.add_argument("--margin", type=int, default=0, help="shift boundary toward track (+px)")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights (n/s/m/l/x; use s or m for CCTV)")
    ap.add_argument("--backend", choices=["ultralytics", "onnx"], default="ultralytics",
                    help="onnx = torch-free (needs an exported .onnx in --weights)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    ap.add_argument("--confirm-frames", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--out", default="output")
    ap.add_argument("--webhook", default=None, help="URL to POST alert JSON to")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--no-video", action="store_true", help="skip writing annotated video")
    ap.add_argument("--no-beep", action="store_true")
    a = ap.parse_args()

    src: str | int = int(a.video) if a.video.isdigit() else a.video
    if a.backend == "onnx":
        from onnx_detector import OnnxYoloTracker
        tracker = OnnxYoloTracker(a.weights, a.conf, device=a.device or "cpu")
    else:
        tracker = YoloTracker(a.weights, a.conf, a.imgsz, a.device)
    if not a.auto_line and not a.zone:
        ap.error("give --zone zone.json or use --auto-line")
    auto = None
    if a.auto_line:
        from yellow_line import YellowLineZone
        auto = YellowLineZone(a.track_side, margin_px=a.margin)
    mon = BoundaryMonitor(tracker, a.zone, a.out, a.confirm_frames, a.cooldown,
                          a.webhook, not a.no_beep, auto_line=auto)

    def prog(i, n):
        if n and i % 30 == 0:
            print(f"\rProcessing {i}/{n} ({100 * i // n}%)", end="", flush=True)

    events = mon.process_video(src, not a.no_video, a.show, prog)
    print("\n" + "=" * 55)
    if events:
        print(f"RESULT: BOUNDARY VIOLATED - {len(events)} alert(s), "
              f"{len({e.track_id for e in events})} person(s)")
        print(f"Log: {mon.alerter.csv_path}")
    else:
        print("RESULT: No boundary violation detected.")


if __name__ == "__main__":
    main()
