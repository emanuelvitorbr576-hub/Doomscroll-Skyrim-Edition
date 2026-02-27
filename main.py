import cv2
import mediapipe as mp
import time
from pathlib import Path
import pygame

# ==============================
# CONFIG
# ==============================

TIMER = 0.5                 # resposta rápida
LOOKING_DOWN = 0.35
DEBOUNCE = 0.45

VIDEO_PATH = Path("./assets/skyrim-skeleton.mp4").resolve()
AUDIO_PATH = Path("./assets/skeleton-with-shield.mp3").resolve()

def draw_eye_boxes(frame, landmarks, color):
    h, w, _ = frame.shape
    
    # Olho esquerdo - pontos ao redor do olho
    left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # Olho direito - pontos ao redor do olho  
    right_eye_points = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    
    # Calcular bounding box para o olho esquerdo
    left_x = [landmarks[i].x * w for i in left_eye_points]
    left_y = [landmarks[i].y * h for i in left_eye_points]
    left_min_x, left_max_x = min(left_x), max(left_x)
    left_min_y, left_max_y = min(left_y), max(left_y)
    
    # Calcular bounding box para o olho direito
    right_x = [landmarks[i].x * w for i in right_eye_points]
    right_y = [landmarks[i].y * h for i in right_eye_points]
    right_min_x, right_max_x = min(right_x), max(right_x)
    right_min_y, right_max_y = min(right_y), max(right_y)
    
    # Desenhar quadrados
    cv2.rectangle(frame, (int(left_min_x)-5, int(left_min_y)-5), 
                  (int(left_max_x)+5, int(left_max_y)+5), color, 2)
    cv2.rectangle(frame, (int(right_min_x)-5, int(right_min_y)-5), 
                  (int(right_max_x)+5, int(right_max_y)+5), color, 2)

# ==============================
# MAIN
# ==============================

def main():
    # Inicializar pygame mixer
    pygame.mixer.init()
    
    if not VIDEO_PATH.exists():
        print("Vídeo não encontrado:", VIDEO_PATH)
        return
    
    if not AUDIO_PATH.exists():
        print("Áudio não encontrado:", AUDIO_PATH)
        return

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro ao abrir webcam")
        return

    # Vídeo (pré-carregado)
    video = cv2.VideoCapture(str(VIDEO_PATH))
    video_playing = False
    sound_played = False

    doomscroll_start = None

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        now = time.time()
        box_color = (0, 255, 0)

        if result.multi_face_landmarks:

            lm = result.multi_face_landmarks[0].landmark

            left_top, left_bottom = lm[145], lm[159]
            right_top, right_bottom = lm[374], lm[386]

            l_iris, r_iris = lm[468], lm[473]

            l_ratio = (l_iris.y - left_bottom.y) / (left_top.y - left_bottom.y + 1e-6)
            r_ratio = (r_iris.y - right_bottom.y) / (right_top.y - right_bottom.y + 1e-6)
            avg_ratio = (l_ratio + r_ratio) / 2

            cv2.putText(frame, f"ratio: {avg_ratio:.2f}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if video_playing:
                looking_down = avg_ratio < DEBOUNCE
            else:
                looking_down = avg_ratio < LOOKING_DOWN

            if looking_down:
                box_color = (0, 0, 255)

                if doomscroll_start is None:
                    doomscroll_start = now

                if now - doomscroll_start >= TIMER:
                    video_playing = True
                    if not sound_played:
                        try:
                            pygame.mixer.music.load(str(AUDIO_PATH))
                            pygame.mixer.music.play()
                            sound_played = True
                        except Exception as e:
                            print(f"Erro ao tocar áudio: {e}")
            else:
                doomscroll_start = None
                video_playing = False
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                sound_played = False
                pygame.mixer.music.stop()
            
            # Desenhar quadrados nos olhos
            draw_eye_boxes(frame, lm, box_color)

        else:
            doomscroll_start = None
            video_playing = False
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ==============================
        # MOSTRAR VÍDEO
        # ==============================

        if video_playing:
            vret, vframe = video.read()
            if not vret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                vframe = cv2.resize(vframe, (w, h))
                cv2.imshow("VIDEO", vframe)
        else:
            try:
                cv2.destroyWindow("VIDEO")
            except:
                pass

        # ==============================
        # UI
        # ==============================

        cv2.imshow("Lock In Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
