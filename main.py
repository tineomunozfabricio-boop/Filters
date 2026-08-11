import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_tracking import INDEX_TIP, THUMB_TIP
from geometry import render_portal, portal_width, ClosingGestureDetector
from filters import FILTROS


MODEL_PATH = "hand_landmarker.task"


def main():

    # Configuración del modelo
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Crear detector
    detector = vision.HandLandmarker.create_from_options(options)

    # Cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. "
            "Revisa el indice de camara o los permisos."
        )

    filtro_index = 0
    closing_detector = ClosingGestureDetector()

    timestamp_ms = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # Espejo
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # Convertir BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir a imagen de MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Timestamp obligatorio para VIDEO
        timestamp_ms += 33

        # Detectar manos
        results = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        left_hand = None
        right_hand = None

        # MediaPipe Tasks devuelve:
        # results.hand_landmarks
        # results.handedness

        if results.hand_landmarks and results.handedness:

            for hand_landmarks, handedness in zip(
                results.hand_landmarks,
                results.handedness
            ):

                raw_label = handedness[0].category_name

                # Debido al efecto espejo de la cámara
                label = "Right" if raw_label == "Left" else "Left"

                if label == "Left":
                    left_hand = hand_landmarks

                else:
                    right_hand = hand_landmarks

        # Si tenemos las dos manos
        if left_hand is not None and right_hand is not None:

            p1 = (
                left_hand[INDEX_TIP].x * w,
                left_hand[INDEX_TIP].y * h
            )

            p2 = (
                left_hand[THUMB_TIP].x * w,
                left_hand[THUMB_TIP].y * h
            )

            p3 = (
                right_hand[INDEX_TIP].x * w,
                right_hand[INDEX_TIP].y * h
            )

            p4 = (
                right_hand[THUMB_TIP].x * w,
                right_hand[THUMB_TIP].y * h
            )

            # Calcular ancho del portal
            width = portal_width(
                p1,
                p2,
                p3,
                p4
            )

            # Detectar gesto de cierre
            if closing_detector.update(width, w):

                filtro_index = (
                    filtro_index + 1
                ) % len(FILTROS)

            # Dibujar portal y aplicar filtro
            frame = render_portal(
                frame,
                p1,
                p2,
                p3,
                p4,
                FILTROS[filtro_index]
            )

        # Mostrar cámara
        cv2.imshow("Filters", frame)

        # Q para salir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()