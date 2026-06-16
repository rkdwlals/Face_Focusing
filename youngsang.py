import os
import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD

# OpenMP 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 모델 수동 로딩
model_path = "C:/Users/jimin/.insightface/models/scrfd_10g_bnkps/scrfd_10g_bnkps.onnx"
detector = SCRFD(model_path)
detector.prepare(ctx_id=0)  # CPU 사용 시 ctx_id=-1

# 영상 열기
video_path = "뉴진스.mp4"
output_path = "테스트.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 30
delay = int(1000 / fps)
out = None

cv2.namedWindow("Focused Face")
cv2.createTrackbar("Zoom", "Focused Face", 50, 150, lambda x: None)
target_center = None

def draw_faces_with_index(frame, dets):
    face_list = []
    for idx, det in enumerate(dets):
        x1, y1, x2, y2, score = map(int, det[:5])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        face_list.append((x1, y1, x2 - x1, y2 - y1, cx, cy))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(idx), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return face_list

def get_nearest_face(faces, target_center):
    tx, ty = target_center
    return min(faces, key=lambda f: (f[4] - tx)**2 + (f[5] - ty)**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    dets = detector.detect(frame, input_size=(640, 640))[0]

    if len(dets) > 0:
        indexed_faces = draw_faces_with_index(frame, dets)

        if target_center is None:
            cv2.imshow("Select a Face (Press index number)", frame)
            key = cv2.waitKey(0)
            if ord('0') <= key <= ord(str(min(len(indexed_faces) - 1, 9))):
                idx = int(chr(key))
                _, _, _, _, cx, cy = indexed_faces[idx]
                target_center = (cx, cy)
                continue
            else:
                print("잘못된 입력입니다.")
                continue

        target_face = get_nearest_face(indexed_faces, target_center)
        if target_face:
            x, y, bw, bh, _, _ = target_face
            pad = cv2.getTrackbarPos("Zoom", "Focused Face")
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + bw + pad, w)
            y2 = min(y + bh + pad, h)

            focused = frame[y1:y2, x1:x2]
            fh, fw = focused.shape[:2]
            output_frame = np.zeros((h, w, 3), dtype=np.uint8)
            start_y = (h - fh) // 2
            start_x = (w - fw) // 2
            output_frame[start_y:start_y+fh, start_x:start_x+fw] = focused

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            out.write(output_frame)
            cv2.imshow("Focused Face", output_frame)
        else:
            cv2.imshow("Focused Face", frame)
    else:
        cv2.imshow("Focused Face", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
if out: out.release()
cv2.destroyAllWindows()
