import streamlit as st
import subprocess
import json
import base64
import hashlib
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Streamlit sayfa ayarları
st.set_page_config(page_title="Steganografi Uygulaması", page_icon="🔒")
st.title("🔒 Steganografi Uygulaması")

# Yardımcı Fonksiyonlar
def generate_ai_image(prompt, width=256, height=256):
    encoded_prompt = quote_plus(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img_bytes = BytesIO(response.content)
    img_bytes.seek(0)
    return img_bytes

def encrypt_data(data_bytes, key_string, original_filename=None):
    key = hashlib.sha256(key_string.encode()).digest()
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data_bytes, AES.block_size))
    return json.dumps({
        'iv': base64.b64encode(cipher.iv).decode(),
        'ciphertext': base64.b64encode(ct_bytes).decode(),
        'filename': os.path.basename(original_filename) if original_filename else None
    })

def decrypt_data(json_str, key_string):
    key = hashlib.sha256(key_string.encode()).digest()
    data = json.loads(json_str)
    iv = base64.b64decode(data['iv'])
    ct = base64.b64decode(data['ciphertext'])
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), data.get('filename')

# Görsel Steganografi Fonksiyonları
def encode_lsb(image_file, secret_data, output_format='PNG'):
    img = Image.open(image_file).convert("RGB")
    binary_secret = ''.join([format(ord(c), '08b') for c in str(secret_data)] + ['0']*40)
    
    pixels = np.array(img)
    data_index = 0
    
    for row in pixels:
        for pixel in row:
            for i in range(3):
                if data_index < len(binary_secret):
                    pixel[i] = (pixel[i] & 0xFE) | int(binary_secret[data_index])
                    data_index += 1

    buf = BytesIO()
    img.save(buf, format=output_format)
    return buf.getvalue()

def decode_lsb(image_file):
    img = Image.open(image_file).convert("RGB")
    pixels = np.array(img)
    binary_data = []
    
    for row in pixels:
        for pixel in row:
            for channel in pixel[:3]:
                binary_data.append(str(channel & 1))
    
    return ''.join(binary_data).split('00000000'*5)[0]

# Ses Steganografi Fonksiyonları (Güncellenmiş)
def encode_lsb_audio(audio_file, secret_data, output_format='wav'):
    try:
        # 1. Convert to WAV format
        convert_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '1', 'pipe:1'
        ]
        
        p_convert = subprocess.run(
            convert_cmd,
            input=audio_file.getvalue(),
            capture_output=True
        )
        
        if p_convert.returncode != 0:
            st.error(f"Dönüşüm hatası: {p_convert.stderr.decode()}")
            return None

        # 2. Process audio data
        full_data = p_convert.stdout
        header = full_data[:44]
        audio_data = bytearray(full_data[44:])
        
        secret_str = str(secret_data)
        binary_secret = ''.join(f"{ord(c):08b}" for c in secret_str) + '0'*40
        
        data_index = 0
        for i in range(len(audio_data)):
            if data_index >= len(binary_secret):
                break
            audio_data[i] = (audio_data[i] & 0xFE) | int(binary_secret[data_index])
            data_index += 1

        # 3. Re-encode to target format
        encode_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'wav', '-i', 'pipe:0', '-f', output_format, 'pipe:1'
        ]
        
        p_encode = subprocess.run(
            encode_cmd,
            input=header + audio_data,
            capture_output=True
        )
        
        return BytesIO(p_encode.stdout) if p_encode.returncode == 0 else None

    except Exception as e:
        st.error(f"Ses kodlama hatası: {str(e)}")
        return None

def decode_lsb_audio(audio_file):
    try:
        # Convert to WAV format
        convert_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', 'pipe:0', '-f', 'wav', 'pipe:1'
        ]
        
        p_convert = subprocess.run(
            convert_cmd,
            input=audio_file.getvalue(),
            capture_output=True
        )
        
        audio_data = bytearray(p_convert.stdout[44:])
        binary_data = ''.join(str(byte & 1) for byte in audio_data)
        clean_data = binary_data.split('00000000'*5)[0]
        
        return ''.join(chr(int(clean_data[i:i+8], 2)) for i in range(0, len(clean_data), 8))
    
    except Exception as e:
        st.error(f"Ses çözme hatası: {str(e)}")
        return None

# Video Steganografi Fonksiyonları (Güncellenmiş)
def encode_lsb_video(video_file, secret_data, output_format='mp4'):
    try:
        # 1. Extract video stream
        extract_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'
        ]
        
        p_extract = subprocess.run(
            extract_cmd,
            input=video_file.getvalue(),
            capture_output=True
        )
        
        # 2. Process video frames
        binary_secret = ''.join(f"{ord(c):08b}" for c in str(secret_data)) + '0'*40
        secret_index = 0
        
        # 3. Re-encode video
        encode_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'fast',
            '-f', output_format, 'pipe:1'
        ]
        
        p_encode = subprocess.run(
            encode_cmd,
            input=processed_frames,
            capture_output=True
        )
        
        return (BytesIO(p_encode.stdout), f"output.{output_format}") if p_encode.returncode == 0 else (None, None)

    except Exception as e:
        st.error(f"Video kodlama hatası: {str(e)}")
        return None, None

def decode_lsb_video(video_file):
    try:
        # Extract raw video using pipes
        extract_cmd = [
            'ffmpeg', '-y',
            '-i', 'pipe:0',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1'
        ]
        p_extract = subprocess.Popen(
            extract_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        raw_video, _ = p_extract.communicate(input=video_file.getvalue())
        
        # Process frames in memory...
        return decoded_data
    except Exception as e:
        st.error(f"Video çözme hatası: {str(e)}")
        return None

# Streamlit UI
def main():
    operation = st.sidebar.radio("İşlem:", ["Gizle", "Çöz"])
    media_type = st.sidebar.selectbox("Medya Türü:", ["Resim", "Ses", "Video"])
    password = st.sidebar.text_input("Şifre:", type="password")

    if operation == "Gizle":
        # Gizleme arayüzü
        st.header(f" secretive Veri Gizleme ({media_type})")

        # File size limits (adjust as needed)
        MAX_CARRIER_SIZE_MB = 50 # Max size for image/audio/video file
        MAX_SECRET_SIZE_MB = 20 # Max size for secret file
        MAX_CARRIER_SIZE_BYTES = MAX_CARRIER_SIZE_MB * 1024 * 1024
        MAX_SECRET_SIZE_BYTES = MAX_SECRET_SIZE_MB * 1024 * 1024
    
    
        # Secret data input
        st.subheader("1. Gizlenecek Veri")
        secret_choice = st.radio("Ne gizlemek istiyorsunuz?", ("Metin", "Dosya"), key="secret_choice")
    
        secret_data_to_embed_bytes = None
        original_secret_filename = None # Keep track of original filename
    
        if secret_choice == "Metin":
            secret_data_input = st.text_area("Gizlenecek metni girin:", key="secret_text")
            if secret_data_input:
                 try:
                    secret_data_to_embed_bytes = secret_data_input.encode('utf-8')
                    original_secret_filename = "gizli_metin.txt" # Assign a default name for text
                    # Check size
                    if len(secret_data_to_embed_bytes) > MAX_SECRET_SIZE_BYTES:
                         st.error(f"Metin verisi çok büyük ({len(secret_data_to_embed_bytes)/(1024*1024):.2f} MB). Maksimum limit: {MAX_SECRET_SIZE_MB} MB.")
                         secret_data_to_embed_bytes = None # Reset if too large
                 except Exception as e:
                      st.error(f"Metin UTF-8'e dönüştürülürken hata: {e}")
                      secret_data_to_embed_bytes = None
            else:
                 # Provide a subtle hint if empty
                 # st.info("Gizlemek için bir metin girin.")
                 pass
        else: # Secret choice is File
            secret_file = st.file_uploader(f"Gizlenecek dosyayı yükleyin (Maksimum {MAX_SECRET_SIZE_MB} MB):", type=None, key="secret_file")
            if secret_file is not None:
                if secret_file.size > MAX_SECRET_SIZE_BYTES:
                     st.error(f"Gizlenecek dosya '{secret_file.name}' boyutu ({secret_file.size/(1024*1024):.2f} MB) limiti ({MAX_SECRET_SIZE_MB} MB) aşıyor.")
                else:
                     original_secret_filename = secret_file.name
                     secret_data_to_embed_bytes = secret_file.getvalue()
    
    
        # Carrier media input
        st.subheader("2. Taşıyıcı Medya")
    
        uploaded_media_file = None
        media_source = None # Track if AI or uploaded file
    
        if "Resim" in media_type:
            # Option for AI image generation or upload
            media_source = st.radio("Görsel kaynağı:", ("Dosya yükle", "AI ile oluştur"), key="image_source")
    
            if media_source == "AI ile oluştur":
                 st.markdown("#### AI ile Görsel Oluşturma")
                 ai_prompt = st.text_input("Görsel için açıklama (prompt):", value="Renkli soyut desen", key="ai_prompt")
                 # --- DÜZELTME BAŞLANGICI ---
                 resolution_options = ["128x128", "256x256", "384x384", "512x512"]
                 default_resolution_str = "256x256"
    
                 selected_resolution_str = st.select_slider(
                     "Görsel çözünürlüğü:",
                     options=resolution_options,
                     value=default_resolution_str, # Varsayılan değer olarak string kullanıldı
                     # format_func'a artık gerek yok, string'ler zaten açıklayıcı
                     key="ai_res_str" # Anahtar ismi değiştirildi (opsiyonel ama iyi pratik)
                 )
    
                 # Seçilen string'i (width, height) tuple'ına dönüştür
                 try:
                     width_str, height_str = selected_resolution_str.split('x')
                     ai_resolution_tuple = (int(width_str), int(height_str))
                     # Eğer başka yerde tuple'a ihtiyaç varsa session state'e kaydedilebilir
                     # st.session_state.ai_selected_resolution_tuple = ai_resolution_tuple
                 except Exception as e:
                     st.error(f"Çözünürlük ayrıştırılamadı: {e}")
                     # Hata durumunda varsayılana dön
                     ai_resolution_tuple = (256, 256)
                     # st.session_state.ai_selected_resolution_tuple = ai_resolution_tuple
                # --- DÜZELTME SONU ---
    
                 # Store AI generated image in session state to avoid regeneration on every interaction
                 # Session state anahtarlarını kontrol et/güncelle
                 if 'ai_generated_image' not in st.session_state:
                     st.session_state.ai_generated_image = None
                 if 'last_ai_prompt' not in st.session_state:
                     st.session_state.last_ai_prompt = ""
                 if 'last_ai_res_str' not in st.session_state: # Anahtar adını string'e göre güncelle
                     st.session_state.last_ai_res_str = ""
    
    
                 col1, col2 = st.columns(2)
                 with col1:
                     if st.button("Önizleme Oluştur/Yenile", key="ai_preview"):
                         if ai_prompt:
                              with st.spinner("AI görsel oluşturuluyor..."):
                                  # Dönüştürülmüş tuple'ı kullan
                                  st.session_state.ai_generated_image = generate_ai_image(ai_prompt, ai_resolution_tuple[0], ai_resolution_tuple[1])
                                  st.session_state.last_ai_prompt = ai_prompt
                                  # Session state'e string gösterimini kaydet
                                  st.session_state.last_ai_res_str = selected_resolution_str
                                  st.success("AI görsel hazır.")
                         else:
                              st.warning("Lütfen görsel için bir açıklama girin.")
    
                 # Display the generated image if available in state
                 if st.session_state.ai_generated_image:
                      with col2:
                          # Başlık için session state'den string'i al
                          caption_res = st.session_state.get('last_ai_res_str', default_resolution_str)
                          st.image(st.session_state.ai_generated_image, caption=f"Oluşturulan: '{st.session_state.last_ai_prompt}' ({caption_res})", use_container_width=True)
                          # Set the uploaded_media_file to the generated image in memory
                          st.session_state.ai_generated_image.seek(0)
                          uploaded_media_file = st.session_state.ai_generated_image
    
    
            else: # media_source == "Dosya yükle"
                 # ... (Dosya yükleme kodu aynı kalır)
                 uploaded_media_file = st.file_uploader(
                     f"Taşıyıcı görsel dosyasını yükleyin (PNG, BMP önerilir) (Maksimum {MAX_CARRIER_SIZE_MB} MB):",
                     type=["png", "bmp", "tiff", "jpg", "jpeg"],
                     key="carrier_image_upload")
    
        elif "Ses" in media_type:
             uploaded_media_file = st.file_uploader(
                 f"Taşıyıcı ses dosyasını yükleyin (WAV, FLAC vb. kayıpsız önerilir) (Maksimum {MAX_CARRIER_SIZE_MB} MB):",
                 # Allow common types, but conversion to WAV happens internally
                 type=["wav", "mp3", "flac", "aac", "ogg", "aiff"],
                 key="carrier_audio_upload")
    
        elif "Video" in media_type:
             uploaded_media_file = st.file_uploader(
                 f"Taşıyıcı video dosyasını yükleyin (Maksimum {MAX_CARRIER_SIZE_MB} MB):",
                 # Allow common types, intermediate is lossless AVI, final is MKV/MP4
                 type=["mp4", "avi", "mkv", "mov", "mpeg", "wmv"],
                 key="carrier_video_upload")
    
    
        # Check uploaded carrier file size (only if not AI generated)
        if uploaded_media_file and media_source != "AI ile oluştur":
             # For UploadedFile object, size attribute exists
             if hasattr(uploaded_media_file, 'size') and uploaded_media_file.size > MAX_CARRIER_SIZE_BYTES:
                 st.error(f"Taşıyıcı medya dosyası '{uploaded_media_file.name}' boyutu ({uploaded_media_file.size/(1024*1024):.2f} MB) limiti ({MAX_CARRIER_SIZE_MB} MB) aşıyor.")
                 uploaded_media_file = None # Reset if too large
             # For BytesIO (like AI image), check getvalue() length
             elif hasattr(uploaded_media_file, 'getvalue') and len(uploaded_media_file.getvalue()) > MAX_CARRIER_SIZE_BYTES:
                  # This case shouldn't be hit for AI currently due to resolution limits, but good practice
                  st.error(f"Oluşturulan AI görselin boyutu ({len(uploaded_media_file.getvalue())/(1024*1024):.2f} MB) beklenmedik şekilde limiti ({MAX_CARRIER_SIZE_MB} MB) aşıyor.")
                  uploaded_media_file = None # Reset
    
    
        # --- Trigger Encoding ---
        st.subheader("3. Gizleme İşlemi")
        st.markdown("---")
    
        if st.button("Veriyi Gizle ve Şifrele", key="encode_button"):
            # --- Input Validation ---
            valid_input = True
            # if not password:
            #      st.error("Lütfen bir şifre girin.")
            #      valid_input = False
            if secret_data_to_embed_bytes is None:
                 st.error("Lütfen gizlenecek bir metin girin veya geçerli bir dosya yükleyin.")
                 valid_input = False
    
            # Check carrier media
            if media_source == "AI ile oluştur":
                if st.session_state.ai_generated_image is None:
                     st.error("Lütfen önce bir AI görseli oluşturun veya 'Dosya yükle' seçeneğini kullanın.")
                     valid_input = False
                else:
                     # Use the image from session state
                     uploaded_media_file = st.session_state.ai_generated_image
                     uploaded_media_file.seek(0) # Ensure pointer is at the start
                     carrier_filename_for_output = "ai_generated_image"
            elif uploaded_media_file is None:
                st.error(f"Lütfen bir taşıyıcı {media_type.split(' ')[0].lower()} dosyası yükleyin.")
                valid_input = False
            else:
                carrier_filename_for_output = os.path.splitext(uploaded_media_file.name)[0]
    
    
            # --- Proceed if Valid ---
            if valid_input:
                with st.spinner(f"{media_type} içine veri gizleniyor ve şifreleniyor... Lütfen bekleyin..."):
                    try:
                        # 1. Encrypt the secret data (bytes)
                        print(f"Şifrelenecek veri tipi: {type(secret_data_to_embed_bytes)}, Boyut: {len(secret_data_to_embed_bytes)} bytes")
                        print(f"Şifreleme için kullanılacak dosya adı: {original_secret_filename}")
                        encrypted_json_data = encrypt_data(secret_data_to_embed_bytes, password, original_secret_filename)
    
                        if encrypted_json_data is None:
                            # Error handled within encrypt_data, just stop
                            raise ValueError("Şifreleme başarısız oldu.") # Raise specific error to be caught below
    
                        # 2. Prepare output filename base
                        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename_base = f"{now_str}_steg_{carrier_filename_for_output}"
    
                        # 3. Perform LSB encoding based on media type
                        output_bytes = None
                        final_output_filename_from_func = None # To capture filename from video func
    
                        if "Resim" in media_type:
                            # Image encoding expects PNG or BMP typically for LSB
                            output_filename = output_filename_base + ".png"
                            print(f"Resim LSB kodlama çağrılıyor. Çıktı adı: {output_filename}")
                            output_bytes = encode_lsb(uploaded_media_file, encrypted_json_data, output_filename)
                            final_output_filename_from_func = output_filename # Use the generated name
    
                        elif "Ses" in media_type:
                             # Audio encoding converts to WAV internally, output is WAV
                             output_filename = output_filename_base + ".wav"
                             print(f"Ses LSB kodlama çağrılıyor. Çıktı adı: {output_filename}")
                             output_bytes = encode_lsb_audio(uploaded_media_file, encrypted_json_data, output_filename)
                             final_output_filename_from_func = output_filename # Use the generated name
    
                        elif "Video" in media_type:
                             # Video encoding outputs MKV (preferred) or AVI
                             # The function now returns (bytes, final_filename)
                             output_filename_suggestion = output_filename_base + ".mkv" # Suggest MKV
                             print(f"Video LSB kodlama çağrılıyor. Önerilen çıktı adı: {output_filename_suggestion}")
                             output_bytes, final_output_filename_from_func = encode_lsb_video(uploaded_media_file, encrypted_json_data, output_filename_suggestion)
    
    
                        # 4. Provide download button if successful
                        if output_bytes and final_output_filename_from_func:
                             st.success("Veri başarıyla gizlendi ve şifrelendi!")
                             st.info(f"Oluşturulan Dosya: {os.path.basename(final_output_filename_from_func)}")
    
                             # Determine mime type for download button
                             mime_type = "application/octet-stream" # Default
                             fname_lower = final_output_filename_from_func.lower()
                             if fname_lower.endswith(('.png', '.bmp', '.tiff')):
                                 mime_type = f"image/{os.path.splitext(fname_lower)[1][1:]}"
                             elif fname_lower.endswith('.wav'):
                                 mime_type = "audio/wav"
                             elif fname_lower.endswith('.avi'):
                                 mime_type = "video/x-msvideo"
                             elif fname_lower.endswith('.mkv'):
                                 mime_type = "video/x-matroska"
                             elif fname_lower.endswith('.mp4'):
                                 mime_type = "video/mp4"
    
    
                             st.download_button(
                                 label=f"Gizlenmiş Dosyayı İndir ({os.path.basename(final_output_filename_from_func)})",
                                 data=output_bytes,
                                 file_name=os.path.basename(final_output_filename_from_func),
                                 mime=mime_type
                             )
                        else:
                             # Error message should have been shown in the respective encode function
                             st.error("Veri gizleme işlemi başarısız oldu. Yukarıdaki hata mesajlarını kontrol edin.")
                             print("Encode fonksiyonundan geçerli byte veya dosya adı dönmedi.")
    
                    except ValueError as ve: # Catch specific error from encryption
                         st.error(f"İşlem Hatası: {ve}")
                         print(f"ValueError: {ve}")
                    except Exception as e:
                        st.error(f"Gizleme işlemi sırasında beklenmedik bir hata oluştu: {e}")
                        import traceback
                        print(f"Beklenmedik Hata (Encode Butonu): {e}\n{traceback.format_exc()}")
                        st.info("Girdi dosyalarınızın formatını ve boyutunu kontrol edin. Gerekli programlar (ffmpeg, ffprobe) sisteminizde kurulu mu?")

        pass
    else:
        # Çözme arayüzü
        st.header(f" secretive Veri Çözme ({media_type})")

        # Supported types for decoding based on encoding output
        decode_file_types = []
        if "Resim" in media_type:
             decode_file_types = ["png", "bmp", "tiff"] # Match encode output/common LSB types
             st.info("Yalnızca PNG, BMP, TIFF gibi kayıpsız formatlarda gizlenmiş veriler güvenilir şekilde çözülebilir.")
        elif "Ses" in media_type:
             decode_file_types = ["wav"] # Encode function outputs WAV
             st.info("Ses çözme işlemi yalnızca '.wav' formatındaki dosyaları destekler (gizleme işlemi sırasında bu formata dönüştürülür).")
        elif "Video" in media_type:
             decode_file_types = ["avi", "mkv"] # Encode function outputs AVI (lossless intermediate) or MKV (muxed)
             st.info("Video çözme işlemi genellikle gizleme sonrası oluşturulan '.avi' veya '.mkv' dosyalarını destekler.")
    
    
        steg_media_file = st.file_uploader(
            f"İçinde gizli veri olan {media_type.split(' ')[0].lower()} dosyasını yükleyin:",
            type=decode_file_types,
            key="steg_file_upload"
        )
    
        st.markdown("---")
    
        if st.button("Veriyi Çöz", key="decode_button"):
             # --- Input Validation ---
             valid_input = True
             # if not password:
             #     st.error("Lütfen şifreyi girin.")
             #     valid_input = False
             if steg_media_file is None:
                 st.error(f"Lütfen çözülecek bir {media_type.split(' ')[0].lower()} dosyası yükleyin.")
                 valid_input = False
    
             # --- Proceed if Valid ---
             if valid_input:
                  with st.spinner(f"{media_type} içinden veri çıkarılıyor ve şifre çözülüyor..."):
                       try:
                           # 1. Extract the hidden JSON data using LSB decode
                           extracted_json_str = None
                           if "Resim" in media_type:
                               extracted_json_str = decode_lsb(steg_media_file)
                           elif "Ses" in media_type:
                               extracted_json_str = decode_lsb_audio(steg_media_file)
                           elif "Video" in media_type:
                               extracted_json_str = decode_lsb_video(steg_media_file)
    
                           if extracted_json_str:
                               print(f"Çıkarılan JSON String (ilk 100 char): {extracted_json_str[:100]}")
                               # 2. Decrypt the extracted data
                               decrypted_bytes, retrieved_filename = decrypt_data(extracted_json_str, password)
    
                               if decrypted_bytes is not None:
                                   st.success("Veri başarıyla çıkarıldı ve şifresi çözüldü!")
    
                                   # 3. Try decoding as text, if fails, offer as file download
                                   try:
                                       decoded_text = decrypted_bytes.decode('utf-8')
                                       st.subheader("Çözülen Metin:")
                                       st.text_area("Metin:", decoded_text, height=150, key="decoded_text_area")
                                   except UnicodeDecodeError:
                                       # It's likely a file
                                       st.subheader("Çözülen Dosya:")
                                       now = datetime.datetime.now()
                                       timestamp = now.strftime("%Y%m%d_%H%M%S")
    
                                       # Construct filename using retrieved filename if available
                                       if retrieved_filename:
                                            # Basic check if it has an extension
                                            if '.' in retrieved_filename:
                                                 file_name_to_download = f"{timestamp}_decrypted_{retrieved_filename}"
                                            else: # No extension, maybe just a name part
                                                 file_name_to_download = f"{timestamp}_decrypted_{retrieved_filename}.bin" # Add default bin
                                       else:
                                           file_name_to_download = f"{timestamp}_decrypted_file.bin" # Default filename
    
    
                                       # Determine mime type for download button
                                       mime_type = "application/octet-stream" # Default
                                       if retrieved_filename:
                                           try:
                                                # Use mimetypes module for better guess
                                                import mimetypes
                                                mime_guess = mimetypes.guess_type(retrieved_filename)[0]
                                                if mime_guess:
                                                     mime_type = mime_guess
                                                print(f"Tahmin edilen MIME türü ({retrieved_filename}): {mime_type}")
                                           except Exception as mime_e:
                                                print(f"MIME türü tahmin edilirken hata: {mime_e}")
    
    
                                       st.download_button(
                                           label=f"Çözülen Dosyayı İndir ({os.path.basename(file_name_to_download)})",
                                           data=decrypted_bytes,
                                           file_name=os.path.basename(file_name_to_download),
                                           mime=mime_type
                                       )
                               else:
                                   # Decryption failed - error shown in decrypt_data
                                   # st.error("Şifre çözme başarısız. Şifre yanlış veya veri bozuk.") # Redundant
                                   pass
                           else:
                               # Extraction failed - error likely shown in decode_lsb_*
                               st.error("Dosyadan gizli veri çıkarılamadı. Dosya formatı doğru mu? Bu dosya içine veri gizlenmiş miydi?")
                               print("LSB decode fonksiyonu None veya boş string döndürdü.")
    
                       except Exception as e:
                            st.error(f"Çözme işlemi sırasında beklenmedik bir hata oluştu: {e}")
                            import traceback
                            print(f"Beklenmedik Hata (Decode Butonu): {e}\n{traceback.format_exc()}")
                            st.info("İpucu: Dosya türü ve şifrenizi kontrol edin. Yüklediğiniz dosyanın gerçekten gizlenmiş veri içerdiğinden emin olun.")

    # --- Footer/Info ---
    st.sidebar.markdown("---")
    st.sidebar.info("Bu uygulama LSB (Least Significant Bit) steganografi tekniğini ve AES şifrelemesini kullanır.")
    st.sidebar.warning("Büyük dosyalarla çalışmak zaman alabilir ve yüksek bellek kullanımı gerektirebilir.")
    st.sidebar.markdown("Geliştirici: Ali11git\nBST Python ile Algoritma")

if __name__ == "__main__":
    main()
