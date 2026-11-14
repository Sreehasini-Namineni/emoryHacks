"""Step 2: open the webcam and show a live preview. Press q to quit."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera import open_webcam


def main() -> None:
    cap = open_webcam()
    print("SANITY CHECK: you should see your live camera. Wave. Then press q.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("SANITY CHECK FAIL: camera opened but no frames arrived.")

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            "Webcam OK — press q to quit",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Step 2 — Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("SANITY CHECK PASS: live frames displayed.")


if __name__ == "__main__":
    main()
