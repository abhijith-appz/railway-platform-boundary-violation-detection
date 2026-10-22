"""
Web dashboard:  streamlit run app.py
Upload a platform video -> set the danger zone -> run detection -> get alerts.
"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import BoundaryMonitor, YoloTracker, save_zone

st.set_page_config(page_title="Railway Boundary Monitor", page_icon="🚆", layout="wide")
st.title("🚆 Railway Platform Boundary Violation Detection")


@st.cache_resource
def get_tracker(weights, conf, imgsz):
    return YoloTracker(weights, conf, imgsz)


with st.sidebar:
    st.header("Settings")
    weights = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=1)
    conf = st.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", [640, 960, 1280], value=960)
    confirm = st.slider("Confirm frames (reduces false alarms)", 1, 30, 3)
    cooldown = st.slider("Alert cooldown (s)", 1, 60, 10)
    webhook = st.text_input("Webhook URL (optional)")

video_file = st.file_uploader("Platform video", type=["mp4", "avi", "mov", "mkv"])
if not video_file:
    st.info("Upload a CCTV clip to begin.")
    st.stop()

tmp = Path(tempfile.mkdtemp())
vpath = tmp / video_file.name
vpath.write_bytes(video_file.read())

cap = cv2.VideoCapture(str(vpath))
ok, first = cap.read()
cap.release()
if not ok:
    st.error("Could not read the video.")
    st.stop()
h, w = first.shape[:2]

# ---- danger zone --------------------------------------------------------- #
st.subheader("1. Define the danger zone")
st.caption("IMPORTANT: the zone must follow the actual track-side strip in this camera view. "
           "Coordinates are normalized [x, y] fractions (0-1). The old full-width bottom "
           "rectangle often misses people. Edit the polygon below for this video; use 3+ corners.")
# Approximate starting polygon for the example perspective (track-side is left of the
# diagonal yellow line in the supplied screenshot). Always inspect the preview and adjust.
default = [[0.20, 0.00], [0.56, 0.00], [0.72, 1.00], [0.34, 1.00]]
txt = st.text_area("danger_zone", json.dumps(default), height=110)
try:
    zone_norm = json.loads(txt)
    zone_px = (np.array(zone_norm) * [w, h]).astype(np.int32)
    assert len(zone_px) >= 3
except Exception:
    st.error("Invalid zone JSON.")
    st.stop()

prev = first.copy()
ov = prev.copy()
cv2.fillPoly(ov, [zone_px], (0, 220, 255))
prev = cv2.addWeighted(ov, 0.3, prev, 0.7, 0)
cv2.polylines(prev, [zone_px], True, (0, 220, 255), 2)
st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), caption="Zone preview", use_container_width=True)

zone_file = tmp / "zone.json"
zone_file.write_text(json.dumps({"normalized": True, "danger_zone": zone_norm}))

# ---- run ------------------------------------------------------------------ #
st.subheader("2. Run detection")
if st.button("Analyse video", type="primary"):
    with st.spinner("Loading model..."):
        tracker = get_tracker(weights, conf, imgsz)
    mon = BoundaryMonitor(tracker, str(zone_file), str(tmp / "out"),
                          confirm, cooldown, webhook or None, beep=False)
    bar = st.progress(0.0)
    events = mon.process_video(
        str(vpath), True, False,
        lambda i, n: bar.progress(min((i + 1) / n, 1.0)) if n else None)
    bar.empty()

    st.caption(f"Analysis finished. Video: {mon.video_path}")
    st.info("Inspect the annotated video: green boxes = detected people; red boxes/labels = confirmed "
            "violations. A foot-point marker outside the yellow polygon will not trigger an alert.")
    if events:
        st.error(f"🚨 BOUNDARY VIOLATED — {len(events)} alert(s), "
                 f"{len({e.track_id for e in events})} person(s)")
        df = pd.DataFrame([e.__dict__ for e in events])
        st.dataframe(df.drop(columns=["snapshot"]), use_container_width=True)
        st.download_button("Download violations.csv", df.to_csv(index=False), "violations.csv")
        cols = st.columns(min(len(events), 4))
        for c, e in zip(cols, events[:4]):
            c.image(e.snapshot, caption=f"ID {e.track_id} @ {e.video_time}")
    else:
        st.success("✅ No confirmed boundary violation detected.")
        st.warning("If people are visible but there are no green boxes, check the model/clip. "
                   "If green boxes exist, inspect whether each foot-point marker enters the yellow zone. "
                   "If it never enters, redraw the polygon to match the actual track-side area. "
                   "This result is not a guarantee that the scene is safe.")

    st.video(str(mon.video_path))
