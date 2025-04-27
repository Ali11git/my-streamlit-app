import streamlit as st
st.set_page_config(
    page_title="Steganografi Uygulaması",
    page_icon="🔒"
)
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import hashlib
import os
import subprocess
import json
from PIL import Image
import wave
import cv2
import io
import datetime
def encode_lsb(image_file, secret_data, output_filename):
    img = Image.open(image_file).convert("RGB")
    encoded = img.copy()
    width, height = img.size
    index = 0
    secret_data_str = str(secret_data)
    binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
    binary_secret += '00000000' * 5
    data_len = len(binary_secret)
    total_pixels = width * height
    if data_len > total_pixels * 3:
        st.warning(f"Uyarı: Gizlenecek veri ({data_len} bit) resmin kapasitesini ({total_pixels * 3} bit) aşıyor. Tüm veri gizlenemeyebilir.")
    for y in range(height):
        for x in range(width):
            if index < data_len:
                r, g, b = img.getpixel((x, y))
                r = int(r)
                g = int(g)
                b = int(b)
                if index < data_len:
                    r = (r & ~1) | int(binary_secret[index])
                    index += 1
                if index < data_len:
                    g = (g & ~1) | int(binary_secret[index])
                    index += 1
                if index < data_len:
                    b = (b & ~1) | int(binary_secret[index])
                    index += 1
                encoded.putpixel((x, y), (r, g, b))
            else:
                break
        if index >= data_len:
            break
    img_byte_arr = io.BytesIO()
    encoded.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr
def decode_lsb(image_file):
    img = Image.open(image_file).convert("RGB")
    binary_data = ""
    terminator_bits = '00000000' * 5
    found_terminator = False
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = img.getpixel((x, y))
            r = int(r)
            g = int(g)
            b = int(b)
            binary_data += str(r & 1)
            if len(binary_data) >= len(terminator_bits) and binary_data[-len(terminator_bits):] == terminator_bits:
                found_terminator = True
                binary_data = binary_data[:-len(terminator_bits)]
                break
            binary_data += str(g & 1)
            if len(binary_data) >= len(terminator_bits) and binary_data[-len(terminator_bits):] == terminator_bits:
                found_terminator = True
                binary_data = binary_data[:-len(terminator_bits)]
                break
            binary_data += str(b & 1)
            if len(binary_data) >= len(terminator_bits) and binary_data[-len(terminator_bits):] == terminator_bits:
                found_terminator = True
                binary_data = binary_data[:-len(terminator_bits)]
                break
        if found_terminator:
            break
    if not found_terminator:
        st.warning("Uyarı: Terminator bulunamadı. Tüm dosya okundu, ancak gizli veri tamamlanmamış olabilir.")
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte_str in all_bytes:
        if len(byte_str) == 8:
            try:
                decoded_data += chr(int(byte_str, 2))
            except ValueError:
                 pass
    return decoded_data
def encode_lsb_audio(audio_file, secret_data, output_filename):
    st.warning("Ses Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{audio_file.name}"
    temp_output_path_converted = "temp_steg_converted.wav" # Daha açıklayıcı isim
    output_bytes = None
    try:
        with open(temp_input_path, "wb") as f:
            f.write(audio_file.getvalue())
        audio_convert_cmd = f"ffmpeg -i {temp_input_path} -acodec pcm_s16le {temp_output_path_converted} -y"
        exit_code = os.system(audio_convert_cmd)
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if exit_code != 0 or not os.path.exists(temp_output_path_converted):
             st.error(f"Hata: Ses dönüştürme başarısız oldu veya '{temp_output_path_converted}' dosyası oluşturulamadı. ffmpeg kurulu ve erişilebilir mi?")
             print(f"Hata: Ses dönüştürme başarısız oldu veya '{temp_output_path_converted}' dosyası oluşturulamadı. ffmpeg kurulu ve erişilebilir mi?")
             return None
        secret_data_str = str(secret_data)
        binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
        binary_secret += '00000000' * 5  # Sonlandırıcı işaret
        with wave.open(temp_output_path_converted, 'rb') as wf:
            params = wf.getparams()
            n_frames = wf.getnframes()
            audio_bytes = bytearray(wf.readframes(n_frames))
        data_index = 0
        data_len = len(binary_secret)
        total_bits_possible = len(audio_bytes) * 8
        if data_len > total_bits_possible:
             st.warning(f"Uyarı: Gömülecek veri boyutu ({data_len} bit), ses dosyasının tahmini kapasitesini ({total_bits_possible} bit) aşıyor. Tüm veri gömülemeyebilir.")
             print(f"Uyarı: Gömülecek veri boyutu ({data_len} bit), ses dosyasının tahmini kapasitesini ({total_bits_possible} bit) aşıyor. Tüm veri gömülemeyebilir.")
        progress_text = "Ses işleniyor... Lütfen bekleyin."
        progress_bar = st.progress(0, text=progress_text)
        for i in range(len(audio_bytes)):
            if data_index < data_len:
                audio_bytes[i] = (audio_bytes[i] & 0xFE) | int(binary_secret[data_index])
                data_index += 1
            else:
                break
            if total_bits_possible > 0:
                 progress = min(data_index / data_len, 1.0) if data_len > 0 else 1.0
                 progress_bar.progress(progress, text=f"Bit {data_index}/{data_len} işleniyor...")
        if 'progress_bar' in locals(): progress_bar.empty()
        print(f"Ses işleme tamamlandı. Toplam {data_index} bit işlendi.")
        if data_index < data_len:
             st.warning(f"Uyarı: Tüm veri sese sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
             print(f"Uyarı: Tüm veri sese sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
        temp_final_output_path = "temp_final_output.wav"
        with wave.open(temp_final_output_path, 'wb') as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(audio_bytes)
        print(f"Veri geçici olarak '{temp_final_output_path}' dosyasına yazıldı.")
        if os.path.exists(temp_final_output_path):
            with open(temp_final_output_path, "rb") as f:
                output_bytes = f.read()
            print(f"'{temp_final_output_path}' dosyasından bayt verisi okundu.")
        return output_bytes
    except FileNotFoundError:
        st.error(f"Hata: Gerekli dosya bulunamadı. İşlem sırasında bir sorun oluştu.")
        print(f"Hata: Gerekli dosya bulunamadı. İşlem sırasında bir sorun oluştu.")
        return None
    except wave.Error as e:
        st.error(f"WAV dosyası işlenirken hata oluştu: {e}")
        print(f"WAV dosyası işlenirken hata oluştu: {e}")
        return None
    except ValueError as e:
        st.error(f"Değer hatası oluştu: {e}")
        print(f"Değer hatası oluştu: {e}")
        return None
    except Exception as e:
        st.error(f"Beklenmedik bir hata oluştu: {e}")
        print(f"Beklenmedik bir hata oluştu: {e}")
        print(f"Hata detayı: {e}")
        return None
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path_converted):
            os.remove(temp_output_path_converted)
        if 'temp_final_output_path' in locals() and os.path.exists(temp_final_output_path):
             os.remove(temp_final_output_path)
             print(f"Geçici dosya '{temp_final_output_path}' temizlendi.")
def decode_lsb_audio(audio_file):
    audio_byte_arr = io.BytesIO(audio_file.getvalue())
    try:
        with wave.open(audio_byte_arr, 'rb') as wf:
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
        binary_data = ""
        terminator_bits = '00000000' * 5
        found_terminator = False
        for byte in audio_bytes:
            binary_data += str(byte & 1)
            if len(binary_data) >= len(terminator_bits) and binary_data[-len(terminator_bits):] == terminator_bits:
                binary_data = binary_data[:-len(terminator_bits)]
                found_terminator = True
                break
        if not found_terminator:
             st.warning("Uyarı: Terminator bulunamadı, tüm dosya okundu.")
        all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
        decoded_data = ""
        for byte_str in all_bytes:
            if len(byte_str) == 8:
                 try:
                    decoded_data += chr(int(byte_str, 2))
                 except ValueError:
                    pass
        return decoded_data
    except FileNotFoundError:
        st.error(f"Hata: Dosya bulunamadı.")
        return None
    except wave.Error as e:
        st.error(f"WAV dosyası hatası: {e}")
        return None
    except Exception as e:
        st.error(f"Beklenmedik bir hata oluştu: {e}")
        return None
def encode_lsb_video(video_file, secret_data, output_filename):
    st.warning("Video Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{video_file.name}"
    temp_output_path_video_only = "temp_steg_video_only.avi"
    temp_audio_aac = "temp_audio.aac"
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    final_output_path = f"{timestamp}_{output_filename}"
    output_video_bytes = None
    try:
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getvalue())
        print(f"Geçici giriş dosyası oluşturuldu: '{temp_input_path}'")
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Hata: Giriş video dosyası '{temp_input_path}' açılamadı. Dosya formatı destekleniyor mu?")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')
        out = cv2.VideoWriter(temp_output_path_video_only, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error(f"Hata: Çıkış video dosyası '{temp_output_path_video_only}' yazılamadı. Codec/dosya uzantısı uyumlu mu? Codec: HFYU")
            cap.release()
            return None
        print(f"Geçici çıkış video dosyası için VideoWriter oluşturuldu: '{temp_output_path_video_only}'")
        secret_data_str = str(secret_data)
        binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
        binary_secret += '00000000' * 5
        data_index = 0
        data_len = len(binary_secret)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_bits_possible = width * height * 3 * total_frames_in_video
        if data_len > total_bits_possible:
            st.warning(f"Uyarı: Gömülecek veri boyutu ({data_len} bit) videonun tahmini kapasitesini ({total_bits_possible} bit) aşıyor. Tüm veri gömülemeyebilir.")
        embedded = False
        progress_text = "Video kareleri işleniyor ve veri gömülüyor... Lütfen bekleyin."
        progress_bar = st.progress(0, text=progress_text)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if total_frames_in_video > 0:
                progress = min(frame_count / total_frames_in_video, 1.0)
                progress_bar.progress(progress, text=f"Kare {frame_count}/{total_frames_in_video} işleniyor...")
            if data_index < data_len:
                for y in range(height):
                    for x in range(width):
                        pixel = frame[y, x]
                        for c in range(3):
                            if data_index < data_len:
                                pixel[c] = (pixel[c] & 0xFE) | int(binary_secret[data_index])
                                data_index += 1
                            else:
                                embedded = True
                                break
                        if embedded: break
                    if embedded: break
            out.write(frame)
            if embedded and data_index >= data_len:
                print(f"Veri {frame_count}. karede tamamen gömüldü. Kalan kareler kopyalanıyor.")
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(frame)
                    frame_count += 1
                    if total_frames_in_video > 0:
                        progress = min(frame_count / total_frames_in_video, 1.0)
                        progress_bar.progress(progress, text=f"Kare {frame_count}/{total_frames_in_video} kopyalanıyor...")
                break
        progress_bar.empty()
        print(f"Video kare işleme tamamlandı. Toplam {frame_count} kare işlendi.")
        if data_index < data_len:
            st.warning(f"Uyarı: Tüm veri videoya sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("OpenCV kaynakları serbest bırakıldı.")
        st.info("Giriş dosyasında ses akışı kontrol ediliyor...")
        audio_exists = False
        try:
            ffprobe_cmd = f"ffprobe -hide_banner -show_streams -select_streams a {temp_input_path}"
            result = subprocess.run(ffprobe_cmd, shell=True, capture_output=True, text=True, check=False)
            if result.returncode == 0 and "codec_type=audio" in result.stdout:
                audio_exists = True
                st.info("Giriş dosyasında ses akışı bulundu.")
            else:
                st.info("Giriş dosyasında ses akışı bulunamadı.")
                print(f"ffprobe çıktısı (ses kontrolü): {result.stdout}\nffprobe hatası: {result.stderr}")
        except FileNotFoundError:
             st.warning("ffprobe bulunamadı. Ses kontrolü yapılamadı.")
             audio_exists = False
        if audio_exists:
            st.info("Orijinal ses akışı çıkarılıyor...")
            audio_extract_cmd = f"ffmpeg -i {temp_input_path} -vn -acodec copy {temp_audio_aac} -y"
            print(f"Ses çıkarma komutu: {audio_extract_cmd}")
            extract_exit_code = os.system(audio_extract_cmd)
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if extract_exit_code != 0 or not os.path.exists(temp_audio_aac):
                st.error(f"Hata: Ses çıkarma başarısız oldu veya '{temp_audio_aac}' dosyası oluşturulamadı. Giriş dosyasının ses formatı destekleniyor mu?")
                if os.path.exists(temp_audio_aac): os.remove(temp_audio_aac)
                return None
            print(f"Ses çıkarma tamamlandı. Çıkış kodu: {extract_exit_code}")
            st.info("LSB uygulanmış video ile orijinal ses birleştiriliyor...")
            video_mux_cmd = f"ffmpeg -i {temp_output_path_video_only} -i {temp_audio_aac} -c:v copy -c:a copy -shortest {final_output_path} -y"
            print(f"Birleştirme komutu: {video_mux_cmd}")
            mux_exit_code = os.system(video_mux_cmd)
            if os.path.exists(temp_output_path_video_only):
                os.remove(temp_output_path_video_only)
            if os.path.exists(temp_audio_aac):
                os.remove(temp_audio_aac)
            if mux_exit_code != 0 or not os.path.exists(final_output_path):
                st.error(f"Hata: Video ve ses birleştirme (muxing) başarısız oldu veya '{final_output_path}' dosyası oluşturulamadı. FFmpeg komutunu kontrol edin.")
                if os.path.exists(final_output_path): os.remove(final_output_path)
                return None
            print(f"Birleştirme tamamlandı. Çıkış kodu: {mux_exit_code}")
            st.success(f"Veri başarıyla videoya gizlendi ve orijinal ses eklendi: '{final_output_path}'")
            with open(final_output_path, "rb") as f:
                 output_video_bytes = f.read()
            print(f"Nihai çıktı dosyası '{final_output_path}' bayt olarak okundu.")
        else:
            st.warning("Giriş dosyasında ses akışı bulunamadı. Sadece LSB uygulanmış video döndürülecektir.")
            if os.path.exists(temp_output_path_video_only):
                 with open(temp_output_path_video_only, "rb") as f:
                     output_video_bytes = f.read()
                 print(f"LSB uygulanmış video dosyası '{temp_output_path_video_only}' bayt olarak okundu.")
            else:
                 st.error(f"Hata: Ses akışı bulunamadı ve LSB uygulanmış video dosyası ('{temp_output_path_video_only}') bulunamadı.")
                 return None
        return output_video_bytes
    except Exception as e:
        st.error(f"İşlem sırasında beklenmedik bir hata oluştu: {e}")
        print(f"Hata detayı: {e}")
        return None
    finally:
        st.info("Geçici dosyalar temizleniyor.")
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            print(f"'{temp_input_path}' temizlendi.")
        if os.path.exists(temp_output_path_video_only):
            os.remove(temp_output_path_video_only)
            print(f"'{temp_output_path_video_only}' temizlendi.")
        if os.path.exists(temp_audio_aac):
            os.remove(temp_audio_aac)
            print(f"'{temp_audio_aac}' temizlendi.")
        if os.path.exists(final_output_path):
             os.remove(final_output_path)
             print(f"'{final_output_path}' temizlendi.")
def decode_lsb_video(video_file):
    st.warning("Video Steganografi çözümleme işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{video_file.name}"
    with open(temp_input_path, "wb") as f:
        f.write(video_file.getvalue())
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error(f"Hata: '{temp_input_path}' açılamadı.")
        os.remove(temp_input_path)
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    binary_data = ""
    terminator_bits = '00000000' * 5
    found_terminator = False
    progress_text = "Video çözümleniyor... Lütfen bekleyin."
    progress_bar = st.progress(0, text=progress_text)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress, text=f"Kare {frame_count}/{total_frames} çözümleniyor...")
        for y in range(height):
            for x in range(width):
                pixel = frame[y, x]
                for c in range(3):
                    binary_data += str(pixel[c] & 1)
                    if len(binary_data) >= len(terminator_bits) and binary_data[-len(terminator_bits):] == terminator_bits:
                        found_terminator = True
                        binary_data = binary_data[:-len(terminator_bits)]
                        break
                if found_terminator: break
            if found_terminator: break
        if found_terminator: break
    progress_bar.empty()
    cap.release()
    cv2.destroyAllWindows()
    os.remove(temp_input_path)
    if not found_terminator:
        st.warning("Uyarı: Terminator bulunamadı!")
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte_str in all_bytes:
        if len(byte_str) == 8:
            try:
                decoded_data += chr(int(byte_str, 2))
            except ValueError:
                pass
    return decoded_data
def encrypt_data(data, key_string, file_extension=None):
    key = hashlib.sha256(key_string.encode('utf-8')).digest()
    cipher = AES.new(key, AES.MODE_CBC)
    data_bytes = data if isinstance(data, bytes) else str(data).encode('utf-8')
    ct_bytes = cipher.encrypt(pad(data_bytes, AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    result = {'iv': iv, 'ciphertext': ct}
    if file_extension is not None:
        result['extension'] = file_extension
    return json.dumps(result)
def decrypt_data(json_input, key_string):
    try:
        key = hashlib.sha256(key_string.encode('utf-8')).digest()
        b64 = json.loads(json_input)
        iv = base64.b64decode(b64['iv'])
        ct = base64.b64decode(b64['ciphertext'])
        retrieved_extension = b64.get('extension')
        cipher = AES.new(key, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return pt, retrieved_extension # Decrypted bytes
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        st.error(f"Şifre çözme hatası: {e}")
        return None, None
    except Exception as e:
        st.error(f"Beklenmedik bir şifre çözme hatası oluştu: {e}")
        return None, None
st.title("🔒 Steganografi Uygulaması")
operation = st.sidebar.radio("Yapmak istediğiniz işlemi seçin:", ("Gizle (Encode)", "Çöz (Decode)"))
media_type = st.selectbox("Gizleme/Çözme yapılacak medya türünü seçin:", ("Resim (Image)", "Ses (Audio)", "Video (Video)"))
password = st.text_input("Şifreyi girin:", type="password")
if operation == "Gizle (Encode)":
    MAX_FILE_SIZE_MB = 8
    st.header("Gizleme (Encode)")
    secret_choice = st.radio("Ne gizlemek istiyorsunuz?", ("Metin", "Dosya"))
    if secret_choice == "Metin":
        secret_data_input = st.text_area("Gizlenecek metni girin:")
        if secret_data_input:
            secret_data_to_embed = secret_data_input.encode('utf-8')
            filename = None
        else:
             secret_data_to_embed = None
    else:
        secret_file = st.file_uploader(f"Gizlenecek dosyayı yükleyin(Maksimum {MAX_FILE_SIZE_MB * 2} MB):")
        if secret_file is not None:
            filename = secret_file.name
            root, file_extension = os.path.splitext(filename)
            secret_data_to_embed = secret_file.getvalue()
        else:
            secret_data_to_embed = None
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    uploaded_media_file = st.file_uploader(f"Gizleme yapılacak {media_type.split(' ')[0].lower()} dosyasını yükleyin(Maksimum {MAX_FILE_SIZE_MB} MB):", type=["png", "bmp", "jpg", "Jpeg"] if "Resim" in media_type else ["mp3","wav","aac","flac","wma","aiff","pcm","alac","dsd"] if "Ses" in media_type else ["mp4", "avi", "mkv", "mpeg4"])
    if st.button("Gizle"):
        if uploaded_media_file is not None and secret_data_to_embed is not None:
            file_size = uploaded_media_file.size
            file_name = uploaded_media_file.name
            if file_size > MAX_FILE_SIZE_BYTES and secret_file.size > (MAX_FILE_SIZE_BYTES * 2):
                if file_size > MAX_FILE_SIZE_BYTES:
                    st.error(f"Hata: '{file_name}' dosyası boyutu {MAX_FILE_SIZE_MB} MB limitini aşıyor. Lütfen daha küçük bir dosya yükleyin.")
                    uploaded_media_file = None
                if secret_file.size > (MAX_FILE_SIZE_BYTES * 4):
                    st.error(f"Hata: '{filename}' dosyası boyutu {MAX_FILE_SIZE_MB * 2} MB limitini aşıyor. Lütfen daha küçük bir dosya yükleyin.")
                    secret_file = None
            else:
                with st.spinner("Veri gizleniyor..."):
                    try:
                        encrypted_secret_data = encrypt_data(secret_data_to_embed, password, filename)
                        only_name, _ = os.path.splitext(uploaded_media_file.name)
                        output_filename = f"encrypted_{only_name}"
                        output_bytes = None
                        if "Resim" in media_type:
                            if not output_filename.lower().endswith(('.png', '.bmp')):
                                output_filename += '.png'
                            output_bytes = encode_lsb(uploaded_media_file, encrypted_secret_data, output_filename)
                        elif "Ses" in media_type:
                            if not output_filename.lower().endswith('.wav'):
                                output_filename += '.wav'
                            output_bytes = encode_lsb_audio(uploaded_media_file, encrypted_secret_data, output_filename)
                        elif "Video" in media_type:
                            if not output_filename.lower().endswith('.avi'):
                                output_filename += '.avi'
                            output_bytes = encode_lsb_video(uploaded_media_file, encrypted_secret_data, output_filename)
                        if output_bytes:
                            st.success("Veri başarıyla gizlendi!")
                            st.download_button(
                                label=f"Gizlenmiş Dosyayı İndir ({output_filename.split('/')[-1]})",
                                data=output_bytes,
                                file_name=output_filename.split('/')[-1],
                                mime="image/png" if "Resim" in media_type else "audio/wav" if "Ses" in media_type else "video/avi"
                            )
                        else:
                             st.error("Veri gizleme başarısız oldu.")
                    except Exception as e:
                        st.error(f"Gizleme sırasında bir hata oluştu: {e}")
        else:
            st.warning("Lütfen tüm alanları doldurun ve dosyaları yükleyin.")
elif operation == "Çöz (Decode)":
    st.header("Çözme (Decode)")
    steg_media_file = st.file_uploader(f"Çözme yapılacak gizlenmiş {media_type.split(' ')[0].lower()} dosyasını yükleyin:", type=["png"] if "Resim" in media_type else ["wav"] if "Ses" in media_type else ["avi"])
    if st.button("Çöz"):
        if steg_media_file is not None:
            with st.spinner("Veri çözümleniyor..."):
                try:
                    extracted_json = None
                    if "Resim" in media_type:
                         extracted_json = decode_lsb(steg_media_file)
                    elif "Ses" in media_type:
                         extracted_json = decode_lsb_audio(steg_media_file)
                    elif "Video" in media_type:
                         extracted_json = decode_lsb_video(steg_media_file)
                    if extracted_json:
                        decrypted_bytes, retrieved_ext = decrypt_data(extracted_json, password)
                        if decrypted_bytes is not None:
                            try:
                                decoded_text = decrypted_bytes.decode('utf-8')
                                st.success("Veri başarıyla çözüldü (Metin):"+decoded_text)
                            except UnicodeDecodeError:
                                st.success("Veri başarıyla çözüldü (Dosya):")
                                st.download_button(
                                    label="Çözülen Dosyayı İndir",
                                    data=decrypted_bytes,
                                    file_name=f"decrypted_{retrieved_ext.split('/')[-1]}",
                                    mime=f"decrypted_{retrieved_ext}"
                                )
                        else:
                            st.error("Şifre yanlış veya veri bozuk.")
                    else:
                         st.error("Gizlenmiş dosyadan veri çıkarılamadı.")
                except Exception as e:
                    st.error(f"Çözme sırasında bir hata oluştu: {e}")
        else:
            st.warning("Lütfen gizlenmiş dosyayı yükleyin ve şifreyi girin.")
