"""
Draw the danger zone on a video frame.

The danger zone = the strip between the yellow safety line and the platform
edge / track. Click its corners in order around the area.

    python select_zone.py --video platform.mp4 --out zone.json

Keys:  left-click add point | u undo | r reset | Enter/s save | q quit
"""
import argparse

import cv2

from detector import save_zone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="zone.json")
    ap.add_argument("--frame", type=int, default=0, help="frame index to draw on")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from the video")
    h, w = frame.shape[:2]
    pts = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow("Select danger zone")
    cv2.setMouseCallback("Select danger zone", on_mouse)
    while True:
        view = frame.copy()
        for p in pts:
            cv2.circle(view, p, 5, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(view, [__import__("numpy").array(pts)], len(pts) > 2, (0, 220, 255), 2)
        cv2.putText(view, "Click zone corners | Enter=save  u=undo  r=reset  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Select danger zone", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("r"):
            pts.clear()
        elif k in (13, ord("s")) and len(pts) >= 3:
            save_zone(a.out, pts, w, h)
            print(f"Saved {len(pts)} points to {a.out}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
