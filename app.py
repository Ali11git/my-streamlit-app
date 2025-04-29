import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np

# --- Konfigürasyon Ayarları (Global Kapsamda) ---
# Kontrol edeceğimiz alanın boyutunu belirle (örneğin 20x20 piksel)
area_size = 20 # Karenin kenar uzunluğu
half_area = area_size // 2 # Bu da dışarıda hesaplanabilir

# Ortalama renk toleransı
color_tolerance = 60

# Hex renk kodunu BGR NumPy array'e çeviren yardımcı fonksiyon
# st.color_picker hex formatında renk döndürür (#RRGGBB)
def hex_to_bgr(hex_color):
    # Hex işaretini (#) kaldır
    hex_color = hex_color.lstrip('#')
    # R, G, B bileşenlerini al (hex -> int)
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # OpenCV BGR formatı bekler, bu yüzden sırayı B, G, R yap
    return np.array([b, g, r], dtype=np.uint8)


# Video akışındaki her kareyi işleyecek fonksiyon
# Bu fonksiyon artık hedef rengi session state'ten alacak
def process_video_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24") # Görüntüyü BGR formatında NumPy dizisine çevir

    # --- Kendi Basit Alan Kontrol Mantığımız ---

    # Görüntünün boyutlarını al
    height, width, _ = img.shape

    # Alanın merkezini belirle (tam ortası)
    center_x = width // 2
    center_y = height // 2

    # Alanın köşe koordinatlarını hesapla
    start_x = max(0, center_x - half_area)
    start_y = max(0, center_y - half_area)
    end_x = min(width, center_x + half_area)
    end_y = min(height, center_y + half_area)

    # Belirlenen alanı (Region of Interest - ROI) al
    roi = img[start_y:end_y, start_x:end_x]

    # Alan boş değilse
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        # Bu alandaki piksellerin ortalama BGR değerini hesapla
        average_bgr = np.mean(roi, axis=(0, 1)).astype(np.uint8)

        # Hedef rengi session state'ten al ve BGR'ye çevir
        # Eğer session state'te yoksa varsayılan olarak kırmızı kullan (ilk yüklemede olabilir)
        target_hex_color = st.session_state.get('target_color_hex', '#FF0000') # Varsayılan: Kırmızı (hex)
        target_bgr = hex_to_bgr(target_hex_color)


        # Ortalama rengin hedef renge yakınlığını kontrol et
        color_difference = np.sum(np.abs(average_bgr - target_bgr))
        # color_tolerance global olarak tanımlandı

        # Sonucu Streamlit'in session state'ine kaydet
        if color_difference < color_tolerance:
            st.session_state.detected_color = f"Hedef Renk Tespit Edildi! (Ortalama BGR: {average_bgr})"
            detection_color_display = (0, 255, 0) # Ekranda yeşil dikdörtgen çiz (BGR)
        else:
            st.session_state.detected_color = f"Beklenen Renk Bulunamadı. (Ortalama BGR: {average_bgr})"
            detection_color_display = (0, 0, 255) # Ekranda kırmızı dikdörtgen çiz (BGR)
    else:
         # ROI boşsa
        st.session_state.detected_color = "Alan Okunamadı."
        detection_color_display = (128, 128, 128) # Gri renk (BGR)


    # --- Görüntüye Bilgi Ekleme (Görselleştirme) ---

    # Kontrol edilen alanın etrafına bir dikdörtgen çiz (BGR rengi kullan)
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), detection_color_display.tolist(), 3) # .tolist() çünkü cv2 renkleri liste/tuple ister

    # Ortalama BGR değerini görüntüye yazdır
    text_y_position = end_y + 25
    if text_y_position > height - 10: text_y_position = height - 10

    # Metin rengi olarak tespit rengini kullan (ama BGR numpy array ise listeye çevir)
    text_color_display = detection_color_display.tolist() if isinstance(detection_color_display, np.ndarray) else detection_color_display

    cv2.putText(img, f"Avg BGR: {average_bgr}", (start_x, text_y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color_display, 2) # Tespit renginde ortalama BGR yaz


    # --- İşlenmiş Kareyi Döndürme ---

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Uygulaması ---

st.title("Alan Rengi Tanıma (Webcam)")

st.write(f"Kameranıza erişim izni vererek ortadaki {area_size}x{area_size} piksellik alanın rengini kontrol edebilirsiniz.")

# Hedef renk seçimi için color picker widget'ı
# Bu widget'ın değeri doğrudan session state'e bağlanır
st.session_state.target_color_hex = st.color_picker(
    "Hedef Rengi Seçin:",
    value=st.session_state.get('target_color_hex', '#FF0000'), # Varsayılan değer kırmızı
    key='color_picker' # Widget için benzersiz anahtar
)

# Streamlit session state'ini başlat (eğer henüz başlamadıysa)
if 'detected_color' not in st.session_state:
    st.session_state.detected_color = "Kamera Başlatılıyor..."


# streamlit-webrtc bileşenini kullanarak kamera akışını başlat
webrtc_ctx = webrtc_streamer(
    key="area-detector-color", # Anahtarı yine değiştirelim
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
result_placeholder = st.empty()
result_placeholder.write(st.session_state.get('detected_color', 'Kamera bekleniyor...'))


st.write(f"Kameranın ortasındaki {area_size}x{area_size} piksellik dikdörtgenin rengine dikkat edin.")
st.write("- Yeşil dikdörtgen: Seçilen hedef renk ortalaması tolerans içinde tespit edildi.")
st.write("- Kırmızı dikdörtgen: Seçilen hedef renk ortalaması bulunamadı.")
st.write(f"Kullanılan Tolerans: {color_tolerance}")


# Bu örnekte de sonuçları temizleme butonu çok anlamlı değil
# if st.button("Sonuçları Temizle"):
#     if 'detected_color' in st.session_state:
#         del st.session_state.detected_color
#     st.rerun()
