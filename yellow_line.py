"""
Automatic yellow-safety-line detection.

Instead of drawing the danger zone by hand, find the painted yellow line in every
frame and treat everything on the track side of it as the danger zone. Works with
perspective views, and with handheld / slowly moving / PTZ cameras where a fixed
polygon would drift.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class YellowLineZone:
    def __init__(self, track_side: str = "left", hsv_low=(15, 90, 120), hsv_high=(35, 255, 255),
                 min_height_frac: float = 0.15, search_from_frac: float = 0.30,
                 margin_px: int = 0, smooth: float = 0.6, extend_up_frac: float = 0.06):
        """
        track_side      : 'left' or 'right' - which side of the yellow line the rails are on
        hsv_low/high    : yellow colour range (tune for lighting; OpenCV hue is 0-179)
        search_from_frac: ignore the top part of the frame (signs, sky) when looking for the line
        margin_px       : shift the boundary toward the track side (+) or platform side (-)
        smooth          : 0 = no smoothing, 0.9 = very smooth (EMA across frames)
        """
        assert track_side in ("left", "right")
        self.side, self.low, self.high = track_side, np.array(hsv_low), np.array(hsv_high)
        self.min_h, self.search_from = min_height_frac, search_from_frac
        self.margin, self.smooth, self.extend = margin_px, smooth, extend_up_frac
        self._params: Optional[np.ndarray] = None   # (y_top, x_top, x_bottom)
        self.found = False

    def _detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2)) if w > 900 else frame
        sh, sw = small.shape[:2]
        k = w / sw
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), self.low, self.high)
        mask[: int(sh * self.search_from)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        best, best_score = None, 0
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if ch < self.min_h * sh:
                continue
            score = area * (ch / max(cw, 1))          # prefer big, long, thin-ish blobs
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return None

        ys, xs = np.where(lab == best)
        pts = []
        for y in np.unique(ys):
            row = xs[ys == y]
            pts.append((row.min() if self.side == "left" else row.max(), y))
        pts = np.array(pts, dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        if abs(vy) < 1e-3:
            return None
        x_at = lambda y: (x0 + (y - y0) * vx / vy) * k
        y_top = max(ys.min() * k - self.extend * h, self.search_from * h * 0.7)
        return np.array([y_top, x_at(y_top / k), x_at(sh)], dtype=np.float32)

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns the danger-zone polygon (int32 Nx2) or None if no line has ever been found."""
        p = self._detect(frame)
        self.found = p is not None
        if p is not None:
            self._params = p if self._params is None else \
                self.smooth * self._params + (1 - self.smooth) * p
        if self._params is None:
            return None
        h, w = frame.shape[:2]
        y_top, x_top, x_bot = self._params
        m = self.margin if self.side == "left" else -self.margin
        x_top, x_bot = x_top + m, x_bot + m
        edge = 0 if self.side == "left" else w
        poly = np.array([[edge, y_top], [x_top, y_top], [x_bot, h], [edge, h]])
        return poly.astype(np.int32)
