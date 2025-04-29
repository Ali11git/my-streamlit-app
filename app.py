import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessor, WebRtcMode
import av # AudioVideo library, used by streamlit-webrtc
import cv2
from pyzbar import pyzbar

# Bu sınıf, video akışındaki her kareyi işlemek için kullanılır.
class QRCodeProcessor(VideoProcessor):
    def __init__(self):
        # QR kod sonuçlarını oturum durumu (session_state) kullanarak saklayacağız
        # Çünkü VideoProcessor stateless olmalıdır.
        if 'qr_results' not in st.session_state:
             st.session_state.qr_results = set() # Tekrarları önlemek için set kullanıyoruz

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # av.VideoFrame'i OpenCV (numpy array) formatına çevir
        img = frame.to_ndarray(format="bgr24")

        # Görüntüdeki QR kodlarını çöz
        decoded_objects = pyzbar.decode(img)

        if decoded_objects:
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
# video_processor_factory parametresi ile yukarıdaki sınıfımızı kullanmasını sağlıyoruz.
webrtc_ctx = webrtc_streamer(
    key="qr-reader", # Bu bileşen için benzersiz bir anahtar
    mode=WebRtcMode.SENDRECV, # Hem video alıp hem işleyip geri göndereceğiz
    client_settings={
        "rtc_configuration": {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # WebRTC için NAT geçişini sağlar
        "media_stream_constraints": {
            "video": True, # Sadece video akışını istiyoruz
            "audio": False,
        },
    },
    video_processor_factory=QRCodeProcessor, # Kareleri işlemek için sınıfımızı kullan
    async_processing=True, # İşlemeyi eşzamansız yap (performans için önemli)
)

# Oturum durumunda saklanan QR kod sonuçlarını göster
if 'qr_results' in st.session_state and st.session_state.qr_results:
    st.subheader("Okunan QR Kodları:")
    # Set'teki verileri listeleyerek göster
    for qr_data in sorted(list(st.session_state.qr_results)): # Alfabetik sırala
        st.markdown(f"- `{qr_data}`") # Markdown ile kod formatında göster

# Sonuçları temizlemek için bir buton
if st.button("Sonuçları Temizle"):
    if 'qr_results' in st.session_state:
        del st.session_state.qr_results
    st.rerun() # Uygulamayı yeniden çalıştır (streamlit 1.10 ve sonrası için st.experimental_rerun() yerine)
