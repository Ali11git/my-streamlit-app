import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np

# Video akışındaki her kareyi işleyecek fonksiyon
def process_video_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24") # Görüntüyü BGR formatında NumPy dizisine çevir

    # --- Kendi Basit Alan Kontrol Mantığımız ---

    # Görüntünün boyutlarını al
    height, width, _ = img.shape

    # Kontrol edeceğimiz alanın boyutunu belirle (örneğin 20x20 piksel)
    area_size = 20 # Karenin kenar uzunluğu
    half_area = area_size // 2

    # Alanın merkezini belirle (tam ortası)
    center_x = width // 2
    center_y = height // 2

    # Alanın köşe koordinatlarını hesapla
    start_x = max(0, center_x - half_area) # Kenarlardan taşmamak için max(0, ...) kullan
    start_y = max(0, center_y - half_area)
    end_x = min(width, center_x + half_area) # Genişlikten taşmamak için min(width, ...) kullan
    end_y = min(height, center_y + half_area)

    # Belirlenen alanı (Region of Interest - ROI) al
    roi = img[start_y:end_y, start_x:end_x]

    # Alan boş değilse (köşeler doğru hesaplandıysa ve boyut > 0 ise)
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        # Bu alandaki piksellerin ortalama BGR değerini hesapla
        average_bgr = np.mean(roi, axis=(0, 1)).astype(np.uint8) # Yükseklik ve genişlik eksenleri boyunca ortalama al

        # Hedef ortalama renk belirle (örneğin parlak kırmızı - BGR formatında (0, 0, 255))
        target_bgr = np.array([0, 0, 255], dtype=np.uint8) # NumPy array olarak tanımla

        # Ortalama rengin hedef renge yakınlığını kontrol et
        color_difference = np.sum(np.abs(average_bgr - target_bgr))
        color_tolerance = 60 # Tolerans değeri (biraz artırılabilir, ortalama daha kararlı olabilir)

        # Sonucu Streamlit'in session state'ine kaydet
        if color_difference < color_tolerance:
            st.session_state.detected_color = f"Hedef Renk Tespit Edildi! (Ortalama BGR: {average_bgr})"
            detection_color_display = (0, 255, 0) # Ekranda yeşil dikdörtgen çiz
        else:
            st.session_state.detected_color = f"Beklenen Renk Bulunamadı. (Ortalama BGR: {average_bgr})"
            detection_color_display = (0, 0, 255) # Ekranda kırmızı dikdörtgen çiz
    else:
         # ROI boşsa hata durumu veya varsayılan
        st.session_state.detected_color = "Alan Okunamadı."
        detection_color_display = (128, 128, 128) # Gri renk

    # --- Görüntüye Bilgi Ekleme (Görselleştirme) ---

    # Kontrol edilen alanın etrafına bir dikdörtgen çiz
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), detection_color_display, 3) # Alanın etrafına dikdörtgen çiz

    # Ortalama BGR değerini görüntüye yazdır
    # Metnin yeri, dikdörtgenin altına veya yakınına olabilir
    text_y_position = end_y + 25
    cv2.putText(img, f"Avg BGR: {average_bgr}", (start_x, text_y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Beyaz renkte ortalama BGR yaz

    # --- İşlenmiş Kareyi Döndürme ---

    # İşlenmiş görüntüyü av.VideoFrame formatına geri çevir
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Uygulaması ---

st.title("Basit Alan Rengi Tanıma (Webcam)")

st.write(f"Kameranıza erişim izni vererek ortadaki {area_size}x{area_size} piksellik alanın rengini kontrol edebilirsiniz.")

# Streamlit session state'ini başlat
if 'detected_color' not in st.session_state:
    st.session_state.detected_color = "Kamera Başlatılıyor..."


# streamlit-webrtc bileşenini kullanarak kamera akışını başlat
webrtc_ctx = webrtc_streamer(
    key="area-detector", # Bu bileşen için benzersiz bir anahtar (öncekiden farklı olmalı)
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_frame_callback=process_video_frame, # Kareleri işlemek için fonksiyonumuzu kullan
    async_processing=True,
)

# Session state'te saklanan renk tespit sonucunu göster
st.subheader("Tespit Sonucu:")
result_placeholder = st.empty() # Boş bir konteyner oluştur
# .get() kullanarak session state'in henüz oluşmamış olma ihtimaline karşı koruma ekleyelim
result_placeholder.write(st.session_state.get('detected_color', 'Kamera bekleniyor...'))


st.write(f"Kameranın ortasındaki {area_size}x{area_size} piksellik dikdörtgenin rengine dikkat edin.")
st.write("- Yeşil dikdörtgen: Hedef renk (kırmızı) ortalaması tolerans içinde tespit edildi.")
st.write("- Kırmızı dikdörtgen: Hedef renk ortalaması bulunamadı.")

# Bu örnekte de sonuçları temizleme butonu çok anlamlı değil ama kalabilir
# if st.button("Sonuçları Temizle"):
#     if 'detected_color' in st.session_state:
#         del st.session_state.detected_color
#     st.rerun()
