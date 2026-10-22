# 얼굴 포커싱 직캠 모듈 (Face_Focusing)

여러 사람이 나오는 영상에서 한 사람을 고르면, 그 사람 얼굴을 확대한 "직캠" 영상을 만들어 저장하는 프로그램입니다.
영상처리 강의 과제로 제출했습니다.

- 시기: 2025년 5월~6월
- 언어·라이브러리: Python, OpenCV, InsightFace(SCRFD 얼굴 검출 모델), NumPy

## 동작

1. InsightFace의 SCRFD 모델(`scrfd_10g_bnkps`)로 프레임마다 얼굴을 검출하고, 얼굴마다 번호를 붙여 보여 줍니다.
2. 첫 화면에서 숫자 키로 따라갈 얼굴을 고릅니다.
3. 이후 프레임에서는 처음 고른 위치에 가장 가까운 얼굴을 찾아, 그 주변을 잘라 확대합니다.
4. `Zoom` 트랙바로 얼굴 주변 여백(확대 정도)을 조절할 수 있습니다.
5. 결과 화면을 원본과 같은 크기의 mp4로 저장합니다. `q`를 누르면 종료합니다.

## 파일

| 파일 | 내용 |
|---|---|
| `youngsang.py` | 전체 프로그램 |
| `영상처리_기말_수정본.pdf` | 과제 보고서 |

## 실행

```bash
pip install opencv-python numpy insightface onnxruntime
python youngsang.py
```

`youngsang.py` 안의 모델 경로(`model_path`)와 입력·출력 영상 경로(`video_path`, `output_path`)를 자기 환경에 맞게 바꿔야 합니다. 입력 영상은 저장소에 포함하지 않았습니다.

README 정리는 Claude(AI)의 도움을 받았습니다.
