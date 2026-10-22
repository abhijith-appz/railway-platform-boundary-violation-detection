"""Tests the violation logic without YOLO/torch: a fake detector walks a 'person'
across the yellow line. Run: python test_logic.py"""
import shutil, numpy as np, cv2
from detector import BoundaryMonitor, Detection

W, H, N = 640, 480, 90
cv2.VideoWriter  # sanity
vw = cv2.VideoWriter("synthetic.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for _ in range(N):
    vw.write(np.full((H, W, 3), 80, np.uint8))
vw.release()

# person A walks down into the zone (zone starts at y=0.78*480=374); person B stays safe
frame_no = {"i": 0}
def fake_detect(frame):
    i = frame_no["i"]; frame_no["i"] += 1
    a_foot = 250 + i * 3                      # crosses y=374 at ~frame 41
    return [Detection(1, (100, a_foot - 120, 160, a_foot), 0.9),
            Detection(2, (400, 150, 460, 300), 0.9)]

shutil.rmtree("test_out", ignore_errors=True)
mon = BoundaryMonitor(fake_detect, "zone.example.json", "test_out",
                      confirm_frames=5, cooldown_s=10, beep=False)
events = mon.process_video("synthetic.mp4", show=False)
print("events:", [(e.track_id, e.frame, e.video_time) for e in events])
assert len(events) == 1 and events[0].track_id == 1, "expected exactly one alert for person 1"
assert 44 <= events[0].frame <= 48, events[0].frame   # crossing ~41 + 5 confirm frames
print("PASS")
