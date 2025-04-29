import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode # VideoProcessor kaldırıldı
import av # AudioVideo library, used by streamlit-webrtc
import cv2
from pyzbar import pyzbar

# Video akışındaki her kareyi işleyecek fonksiyon
def process_video_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Görüntüdeki QR kodlarını çöz
    decoded_objects = pyzbar.decode(img)

    if decoded_objects:
        # QR kod sonuçlarını oturum durumu (session_state) kullanarak saklayacağız
        if 'qr_results' not in st.session_state:
             st.session_state.qr_results = set() # Tekrarları önlemek için set kullanıyoruz

        for obj in decoded_objects:
            # QR kodundan okunan veriyi al (bytes -> string)
            data = obj.data.decode("utf-8")

            # Veriyi oturum durumuna ekle
            st.session_state.qr_results.add(data)

            # QR kodunun etrafına bir kutu çiz
            (x, y, w, h) = obj.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Okunan veriyi görüntünün üzerine yaz
            cv2.putText(img, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # İşlenmiş görüntüyü av.VideoFrame formatına geri çevir
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit Uygulaması ---

st.title("QR Code Reader (Webcam)")

st.write("Kameranıza erişim izni vererek QR kodlarını tarayabilirsiniz.")

# streamlit-webrtc bileşenini kullanarak kamera akışını başlat
# client_settings kaldırıldı, rtc_configuration doğrudan webrtc_streamer'a geçti
webrtc_ctx = webrtc_streamer(
    key="qr-reader", # Bu bileşen için benzersiz bir anahtar
    mode=WebRtcMode.SENDRECV, # Hem video alıp hem işleyip geri göndereceğiz
    # client_settings = { ... } BLOĞU KALDIRILDI

    # rtc_configuration doğrudan parametre olarak eklendi
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # WebRTC için NAT geçişini sağlar

    # media_stream_constraints ayarları da doğrudan veya farklı bir parametre ile verilebilir.
    # Eski dökümantasyonlarda bu ayarlar ayrı bir parametre ile verilebiliyordu, deneyelim:
    media_stream_constraints={
         "video": True, # Sadece video akışını istiyoruz
         "audio": False,
     },

    video_frame_callback=process_video_frame, # Kareleri işlemek için fonksiyonumuzu kullan
    async_processing=True, # İşlemeyi eşzamansız yap (performans için önemli)
)

# Oturum durumunda saklanan QR kod sonuçlarını göster
if 'qr_results' in st.session_state and st.session_state.qr_results:
    st.subheader("Okunan QR Kodları:")
    for qr_data in sorted(list(st.session_state.qr_results)):
        st.markdown(f"- `{qr_data}`")
else:
     st.info("Point your camera at a QR code.")


# Sonuçları temizlemek için bir buton
if st.button("Sonuçları Temizle"):
    if 'qr_results' in st.session_state:
        del st.session_state.qr_results
    st.rerun()
