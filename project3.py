import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import threading
import winsound  # Windows yerleşik ses kütüphanesi

# ==============================================================================
# KULLANICI AYARLARI (Kendi fotoğraf ve .wav uzantılı ses dosyanı buraya yaz)
# ==============================================================================
IMAGE_PATH = "alarm.jpg"   # Klasördeki fotoğrafın adı (örn: "resim.png" veya "alarm.jpg")
SOUND_PATH = "alarm.wav"   # Klasördeki .wav ses dosyasının adı (Yoksa otomatik siren çalar)
# ==============================================================================

# PyAutoGUI güvenlik ve hız ayarları
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# MediaPipe Tasks bileşenleri
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

tip_ids = [4, 8, 12, 16, 20]

# Ekran ve Kamera Parametreleri
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 640, 480
FRAME_R = 100
SMOOTHING = 5

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Durum (State) Değişkenleri
mode = "CURSOR"                 # Başlangıç modu: "CURSOR" veya "COUNTING"
fist_cycle_state = "WAIT_OPEN"  # Aç-kapa takibi için: "WAIT_OPEN" -> "WAIT_CLOSE"
open_close_count = 0            # Kaç kere aç-kapa yapıldı
last_click_time = 0             # Seri tıklamayı önlemek için zamanlayıcı
alarm_active = False            # Alarmın o an çalıp çalmadığını kontrol eder

# Alarm fotoğrafını yükle (Dosya yoksa otomatik uyarı ekranı oluştur)
if os.path.exists(IMAGE_PATH):
    alarm_img = cv2.imread(IMAGE_PATH)
    alarm_img = cv2.resize(alarm_img, (600, 450))
else:
    alarm_img = np.zeros((450, 600, 3), dtype=np.uint8)
    alarm_img[:] = (0, 0, 180)
    cv2.putText(alarm_img, "DIKKAT! ALARM AKTIF!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    cv2.putText(alarm_img, f"'{IMAGE_PATH}' klasorde yok.", (90, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

# Arka planda alarm çalma fonksiyonu
def trigger_alarm_sound():
    global alarm_active
    if os.path.exists(SOUND_PATH):
        winsound.PlaySound(SOUND_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
    else:
        def beep_loop():
            while alarm_active:
                winsound.Beep(1200, 250)
                winsound.Beep(800, 250)
        threading.Thread(target=beep_loop, daemon=True).start()

def stop_alarm_sound():
    if os.path.exists(SOUND_PATH):
        winsound.PlaySound(None, winsound.SND_PURGE)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            lm_list = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * CAM_W), int(lm.y * CAM_H)
                lm_list.append([id, cx, cy])

            # El iskeletini çiz
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, (lm_list[start_idx][1], lm_list[start_idx][2]),
                                (lm_list[end_idx][1], lm_list[end_idx][2]), (0, 255, 0), 2)
            for lm in lm_list:
                cv2.circle(frame, (lm[1], lm[2]), 4, (0, 0, 255), cv2.FILLED)

            if len(lm_list) == 21:
                fingers = []

                # 1. Başparmak Kontrolü (Mesafe bazlı)
                dist_thumb_tip = ((lm_list[4][1] - lm_list[17][1])**2 + (lm_list[4][2] - lm_list[17][2])**2) ** 0.5
                dist_thumb_mcp = ((lm_list[2][1] - lm_list[17][1])**2 + (lm_list[2][2] - lm_list[17][2])**2) ** 0.5
                fingers.append(1 if dist_thumb_tip > dist_thumb_mcp else 0)

                # 2. Diğer 4 Parmak Kontrolü: fingers[1]:İşaret, fingers[2]:Orta, fingers[3]:Yüzük, fingers[4]:Serçe
                for id in range(1, 5):
                    fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)

                total_fingers = fingers.count(1)

                # Parmak ucu koordinatları
                thumb_x, thumb_y = lm_list[4][1], lm_list[4][2]   # Başparmak ucu (4)
                index_x, index_y = lm_list[8][1], lm_list[8][2]   # İşaret parmağı ucu (8)
                middle_x, middle_y = lm_list[12][1], lm_list[12][2] # Orta parmak ucu (12)

                # ==============================================================
                # ÖZEL HAREKET: BAŞPARMAK VE İŞARET PARMAĞI BİRBİRİNE DEĞİYORSA -> ALARM
                # ==============================================================
                dist_thumb_index = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

                # Mesafe 35 pikselden küçükse VE el tam yumruk değilse (diğer 3 parmaktan en az biri açıksa, örn: 👌 işareti)
                if dist_thumb_index < 35 and (fingers[2] == 1 or fingers[3] == 1 or fingers[4] == 1):
                    cv2.circle(frame, ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2), 14, (0, 0, 255), cv2.FILLED)
                    cv2.imshow("ALARM FOTOGRAFI", alarm_img)
                    if not alarm_active:
                        alarm_active = True
                        trigger_alarm_sound()
                else:
                    if alarm_active:
                        alarm_active = False
                        stop_alarm_sound()
                        try:
                            cv2.destroyWindow("ALARM FOTOGRAFI")
                        except:
                            pass

                # ==============================================================
                # MOD 1: PARMAK SAYMA MODU (COUNTING)
                # ==============================================================
                if mode == "COUNTING":
                    cv2.rectangle(frame, (20, 60), (160, 160), (245, 117, 16), cv2.FILLED)
                    cv2.putText(frame, str(total_fingers), (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
                    cv2.putText(frame, f"Cikis icin Ac-Kapa: {open_close_count}/2", (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 2 Kere Avuç Aç-Kapa Takibi (5 parmak -> 0 parmak = 1 Tur)
                    if total_fingers == 5:
                        fist_cycle_state = "WAIT_CLOSE"
                    elif total_fingers == 0 and fist_cycle_state == "WAIT_CLOSE":
                        open_close_count += 1
                        fist_cycle_state = "WAIT_OPEN"
                        if open_close_count >= 2:
                            mode = "CURSOR"
                            open_close_count = 0

                # ==============================================================
                # MOD 2: FARE KONTROL MODU (CURSOR)
                # ==============================================================
                elif mode == "CURSOR" and not alarm_active:
                    cv2.rectangle(frame, (FRAME_R, FRAME_R), (CAM_W - FRAME_R, CAM_H - FRAME_R), (255, 0, 255), 2)

                    # A) Sadece Serçe Parmağı Kaldırıldıysa -> PARMAK SAYMA MODUNA GEÇ
                    if fingers == [0, 0, 0, 0, 1]:
                        mode = "COUNTING"
                        open_close_count = 0
                        fist_cycle_state = "WAIT_OPEN"

                    # B) Sadece İşaret Parmağı Havadaysa -> İMLECİ HAREKET ETTİR
                    elif fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                        x3 = np.interp(index_x, (FRAME_R, CAM_W - FRAME_R), (0, SCREEN_W))
                        y3 = np.interp(index_y, (FRAME_R, CAM_H - FRAME_R), (0, SCREEN_H))

                        curr_x = prev_x + (x3 - prev_x) / SMOOTHING
                        curr_y = prev_y + (y3 - prev_y) / SMOOTHING

                        pyautogui.moveTo(curr_x, curr_y)
                        prev_x, prev_y = curr_x, curr_y
                        cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)

                    # C) İşaret ve Orta Parmak Havadaysa -> BİRBİRİNE DEĞİNCE SOL CLICK YAP
                    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                        dist_click = ((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2) ** 0.5
                        cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (255, 0, 255), 2)

                        if dist_click < 40:
                            cv2.circle(frame, ((index_x + middle_x) // 2, (index_y + middle_y) // 2), 12, (0, 255, 0), cv2.FILLED)
                            if time.time() - last_click_time > 0.35:
                                pyautogui.click()
                                last_click_time = time.time()

        # Üst bilgi çubuğu
        cv2.rectangle(frame, (0, 0), (CAM_W, 45), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, f"AKTIF MOD: {mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Akilli El Kontrol Sistemi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stop_alarm_sound()
cap.release()
cv2.destroyAllWindows()