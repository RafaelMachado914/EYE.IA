
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE = Path(__file__).resolve().parents[2]

MODELO = BASE / "hand_landmarker.task"

SINAL = "ola"
PARTICIPANTE = "p001"
GRAVACAO = "gravacao_001"

TOTAL_QUADROS = 30
FPS = 20

PASTA_VIDEOS = BASE / "dataset" / "videos" / PARTICIPANTE / SINAL
PASTA_DADOS = BASE / "dataset" / "processed" / PARTICIPANTE / SINAL

PASTA_VIDEOS.mkdir(parents=True, exist_ok=True)
PASTA_DADOS.mkdir(parents=True, exist_ok=True)

ARQUIVO_VIDEO = PASTA_VIDEOS / f"{GRAVACAO}.mp4"
ARQUIVO_DADOS = PASTA_DADOS / f"{GRAVACAO}.npy"

if ARQUIVO_VIDEO.exists() or ARQUIVO_DADOS.exists():
    raise SystemExit("Essa gravação já existe. Altere GRAVACAO.")

if not MODELO.exists():
    raise SystemExit("Arquivo hand_landmarker.task não encontrado.")


opcoes = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODELO)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(opcoes)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    detector.close()
    raise SystemExit("Não foi possível abrir a webcam.")

largura = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

gravador = cv2.VideoWriter(
    str(ARQUIVO_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (largura, altura)
)

if not gravador.isOpened():
    camera.release()
    detector.close()
    raise SystemExit("Não foi possível criar o vídeo.")

print("EYE.IA - Coletor de sinais")
print(f"Sinal: {SINAL}")
print("Prepare-se para executar o sinal.")

for segundos in range(3, 0, -1):
    print(f"Gravação em {segundos}...")
    time.sleep(1)

coordenadas = []
inicio = time.monotonic()
ultimo_timestamp = -1

try:
    while len(coordenadas) < TOTAL_QUADROS:

        sucesso, imagem = camera.read()

        if not sucesso:
            print("Falha na captura.")
            break

        imagem = cv2.flip(imagem, 1)

        rgb = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )

        imagem_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            (time.monotonic() - inicio) * 1000
        )
        timestamp = max(timestamp, ultimo_timestamp + 1)
        ultimo_timestamp = timestamp

        resultado = detector.detect_for_video(
            imagem_mp,
            timestamp
        )

        pontos = np.zeros((2, 21, 3), dtype=np.float32)

        for indice, mao in enumerate(
            resultado.hand_landmarks[:2]
        ):
            for j, ponto in enumerate(mao):
                pontos[indice, j] = [
                    ponto.x,
                    ponto.y,
                    ponto.z
                ]

        coordenadas.append(pontos.flatten())

        gravador.write(imagem)

        cv2.putText(
            imagem,
            f"Gravando: {len(coordenadas)}/{TOTAL_QUADROS}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("EYE.IA - Coletor", imagem)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    gravador.release()
    detector.close()
    cv2.destroyAllWindows()


if len(coordenadas) == TOTAL_QUADROS:

    np.save(
        ARQUIVO_DADOS,
        np.array(coordenadas, dtype=np.float32)
    )

    print("Gravação concluída!")
    print(f"Vídeo: {ARQUIVO_VIDEO}")
    print(f"Coordenadas: {ARQUIVO_DADOS}")

else:
    ARQUIVO_VIDEO.unlink(missing_ok=True)
    print("Gravação incompleta. Tente novamente.")
