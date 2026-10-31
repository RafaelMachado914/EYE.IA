
import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

opcoes = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(opcoes)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a webcam.")
    detector.close()
    raise SystemExit

conexoes = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

inicio = time.monotonic()
ultimo_timestamp = -1

while True:
    sucesso, imagem = camera.read()

    if not sucesso:
        break

    imagem = cv2.flip(imagem, 1)

    rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    imagem_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.monotonic() - inicio) * 1000)
    timestamp = max(timestamp, ultimo_timestamp + 1)
    ultimo_timestamp = timestamp

    resultado = detector.detect_for_video(
        imagem_mp,
        timestamp
    )

    altura, largura, _ = imagem.shape

    for mao in resultado.hand_landmarks:
        pontos = []

        for ponto in mao:
            x = int(ponto.x * largura)
            y = int(ponto.y * altura)

            pontos.append((x, y))

            cv2.circle(
                imagem, (x, y), 4, (0, 255, 0), -1
            )

        for inicio_ponto, fim_ponto in conexoes:
            cv2.line(
                imagem,
                pontos[inicio_ponto],
                pontos[fim_ponto],
                (255, 255, 255),
                2
            )

    cv2.imshow("EYE.IA - Deteccao de Maos", imagem)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
detector.close()
cv2.destroyAllWindows()
