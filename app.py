import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np # NumPy kütüphanesini ekliyoruz

# Video akışındaki her kareyi işleyecek fonksiyon
def process_video_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24") # Görüntüyü BGR formatında NumPy dizisine çevir

    # --- Kendi Basit Piksel Kontrol Mantığımız ---

    # Görüntünün boyutlarını al
    height, width, _ = img.shape

    # Kontrol edeceğimiz pikselin koordinatlarını belirle (örneğin tam orta)
    center_x = width // 2
    center_y = height // 2

    # Kontrol edeceğimiz pikselin BGR değerini al
    # OpenCV'de renk formatı BGR'dir (Mavi, Yeşil, Kırmızı)
    pixel_bgr = img[center_y, center_x]

    # Hedef renk belirle (örneğin parlak kırmızı - BGR formatında (0, 0, 255))
    target_bgr = np.array([0, 0, 255], dtype=np.uint8) # NumPy array olarak tanımlamak iyi pratiktir

    # Belirli bir tolerans içinde rengi karşılaştır
    # Tam eşitlik yerine renklerin birbirine yakın olup olmadığını kontrol etmek daha pratiktir.
    # Burada basitçe bileşenlerin farklarının toplamını kontrol edelim.
    color_difference = np.sum(np.abs(pixel_bgr - target_bgr))
    color_tolerance = 50 # Tolerans değeri (0 tam eşitlik, arttıkça daha esnek)

    # Sonucu Streamlit'in session state'ine kaydet
    # Bu sonuç ana Streamlit döngüsünde okunup gösterilecek
    if color_difference < color_tolerance:
        st.session_state.detected_color = "Hedef Kırmızı Renk Tespit Edildi!"
        detection_color_display = (0, 255, 0) # Ekranda yeşil daire çiz
    else:
        st.session_state.detected_color = "Beklenen Renk Bulunamadı."
        detection_color_display = (0, 0, 255) # Ekranda kırmızı daire çiz

    # --- Görüntüye Bilgi Ekleme (Görselleştirme) ---

    # Kontrol edilen pikselin etrafına bir daire çiz
    cv2.circle(img, (center_x, center_y), 10, detection_color_display, 3) # Merkezde daire, kalınlık 3

    # Pikselin BGR değerini görüntüye yazdır
    cv2.putText(img, f"BGR: {pixel_bgr}", (center_x + 15, center_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Beyaz renkte piksel değeri yaz

    # --- İşlenmiş Kareyi Döndürme ---

    # İşlenmiş görüntüyü av.VideoFrame formatına geri çevir
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Uygulaması ---

st.title("Basit Piksel Tanıma (Webcam)")

st.write("Kameranıza erişim izni vererek ortadaki pikselin rengini kontrol edebilirsiniz.")

# Streamlit session state'ini başlat
if 'detected_color' not in st.session_state:
    st.session_state.detected_color = "Kamera Başlatılıyor..."


# streamlit-webrtc bileşenini kullanarak kamera akışını başlat
webrtc_ctx = webrtc_streamer(
    key="pixel-detector", # Bu bileşen için benzersiz bir anahtar
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
# Sonucu bir placeholder'da gösteriyoruz ki her kare işlendiğinde güncellensin
result_placeholder = st.empty() # Boş bir konteyner oluştur
result_placeholder.write(st.session_state.get('detected_color', 'Kamera bekleniyor...'))


st.write("Kameranın ortasındaki dairenin rengine dikkat edin.")
st.write("- Yeşil daire: Hedef renk (kırmızı) tespit edildi.")
st.write("- Kırmızı daire: Hedef renk bulunamadı.")

# Bu basit örnekte sonuçları temizleme butonu çok anlamlı değil ama önceki koddan kaldı
# if st.button("Sonuçları Temizle"):
#     if 'detected_color' in st.session_state:
#         del st.session_state.detected_color
#     st.rerun() # Uygulamayı yeniden çalıştır
