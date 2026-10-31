
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: não foi possível abrir a webcam.")
    raise SystemExit

print("Webcam funcionando! Pressione ESC para sair.")

while True:
    sucesso, imagem = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem.")
        break

    imagem = cv2.flip(imagem, 1)

    cv2.imshow("EYE.IA - Webcam", imagem)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
