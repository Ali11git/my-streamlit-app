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
                # Ensure pixel values are integers before bitwise operations
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
    st.warning("Audio Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{audio_file.name}"
    with open(temp_input_path, "wb") as f:
        f.write(audio_file.getvalue())
    temp_output_path_video_only = "temp_steg_video_only.wav"
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    final_output_path = f"{timestamp}{output_filename}"
    audio_convert_cmd = f"ffmpeg -i {temp_input_path} -acodec pcm_s16le {temp_output_path_video_only}"
    os.system(audio_convert_cmd)
    os.remove(temp_input_path)
    if not os.path.exists(temp_output_path_video_only):
        st.error(f"Hata: '{temp_output_path_video_only}' açılamadı.")
        # os.remove(temp_input_path) # Clean up temp file
        return None
    try:
        secret_data_str = str(secret_data)
        binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
        binary_secret += '00000000' * 5  # Terminator
        with wave.open(temp_output_path_video_only, 'rb') as wf:
            params = wf.getparams()
            n_frames = wf.getnframes()
            audio_bytes = bytearray(wf.readframes(n_frames))
        data_index = 0
        data_len = len(binary_secret)
        total_bits_possible = len(audio_bytes)
        if data_len > total_bits_possible:
            st.warning(f"Uyarı: Veri boyutu ({data_len} bit) videonun tahmini kapasitesini ({total_bits_possible} bit) aşabilir. Tüm veri gömülemeyebilir.")
        progress_text = "Ses işleniyor... Lütfen bekleyin."
        progress_bar = st.progress(0, text=progress_text)
        for i in range(total_bits_possible):
            if data_index < data_len:
                audio_bytes[i] = (audio_bytes[i] & 0xFE) | int(binary_secret[data_index])
                data_index += 1
            else:
                break
            if total_bits_possible > 0:
                progress = min(data_index / total_bits_possible, 1.0)
                progress_bar.progress(progress, text=f"Katman {data_index}/{total_bits_possible} işleniyor...")
        progress_bar.empty()
        print(f"Ses işleme tamamlandı. Toplam {data_index} katman işlendi.")
        if data_index < data_len:
            st.warning(f"Uyarı: Tüm veri sese sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
        with wave.open(final_output_path, 'wb') as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(audio_bytes)
        print(f"Veri başarıyla '{final_output_path}' dosyasına gizlendi.")
        with open(final_output_path, "rb") as f:
            output_video_bytes = f.read()
        return output_video_bytes
    except FileNotFoundError:
        print(f"Hata: '{temp_output_path_video_only}' bulunamadı.")
        return None
    except wave.Error as e:
        print(f"WAV dosyası hatası: {e}")
        return None
    except ValueError as e:
        print(f"Hata: {e}")
        return None
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        return None
    finally:
        if os.path.exists(temp_input_path): os.remove(temp_input_path)
        if os.path.exists(temp_output_path_video_only): os.remove(temp_output_path_video_only)
        if os.path.exists(final_output_path):
            pass
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
    with open(temp_input_path, "wb") as f:
        f.write(video_file.getvalue())
    temp_output_path_video_only = "temp_steg_video_only.avi"
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    final_output_path = f"{timestamp}{output_filename}"
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error(f"Hata: '{temp_input_path}' açılamadı.")
        os.remove(temp_input_path)
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    out = cv2.VideoWriter(temp_output_path_video_only, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error(f"Hata: Çıkış video dosyası '{temp_output_path_video_only}' yazılamadı. Codec/dosya uzantısı uyumlu mu? Codec: HFYU")
        cap.release()
        os.remove(temp_input_path)
        return None
    secret_data_str = str(secret_data)
    binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
    binary_secret += '00000000' * 5
    data_index = 0
    data_len = len(binary_secret)
    total_bits_possible = width * height * 3 * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if data_len > total_bits_possible:
        st.warning(f"Uyarı: Veri boyutu ({data_len} bit) videonun tahmini kapasitesini ({total_bits_possible} bit) aşabilir. Tüm veri gömülemeyebilir.")
    embedded = False
    progress_text = "Video işleniyor... Lütfen bekleyin."
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
            progress_bar.progress(progress, text=f"Kare {frame_count}/{total_frames} işleniyor...")
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
            print(f"Veri {frame_count}. karede tamamen gömüldü.")
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
                frame_count += 1
                if total_frames > 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress, text=f"Kare {frame_count}/{total_frames} işleniyor...")
            break
    progress_bar.empty()
    print(f"Video işleme tamamlandı. Toplam {frame_count} kare işlendi.")
    if data_index < data_len:
        st.warning(f"Uyarı: Tüm veri videoya sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.info("Sese tekrar ekleniyor... Bu biraz zaman alabilir.")
    audio_extract_cmd = f"ffmpeg -i {temp_input_path} -vn -acodec copy temp_audio.aac"
    video_mux_cmd = f"ffmpeg -i {temp_output_path_video_only} -i temp_audio.aac -c:v copy -c:a copy -shortest {final_output_path}"
    try:
        os.system(audio_extract_cmd)
        os.system(video_mux_cmd)
        st.success(f"Veri başarıyla videoya gizlendi ve ses eklendi: '{final_output_path}'")
        with open(final_output_path, "rb") as f:
            output_video_bytes = f.read()
        return output_video_bytes
    except Exception as e:
        st.error(f"Ses ekleme veya birleştirme hatası oluştu. ffmpeg kurulu mu? Hata: {e}")
        return None
    finally:
        if os.path.exists(temp_input_path): os.remove(temp_input_path)
        if os.path.exists(temp_output_path_video_only): os.remove(temp_output_path_video_only)
        if os.path.exists("temp_audio.aac"): os.remove("temp_audio.aac")
        if os.path.exists(final_output_path):
            pass
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
        secret_file = st.file_uploader("Gizlenecek dosyayı yükleyin:")
        if secret_file is not None:
            filename = secret_file.name
            root, file_extension = os.path.splitext(filename)
            secret_data_to_embed = secret_file.getvalue()
        else:
            secret_data_to_embed = None
    uploaded_media_file = st.file_uploader(f"Gizleme yapılacak {media_type.split(' ')[0].lower()} dosyasını yükleyin:", type=["png", "bmp"] if "Resim" in media_type else ["mp3","wav","aac","flac","wma","aiff","pcm","alac","dsd"] if "Ses" in media_type else ["mp4", "avi", "mkv"])
    if st.button("Gizle"):
        if uploaded_media_file is not None and secret_data_to_embed is not None and password:
            with st.spinner("Veri gizleniyor..."):
                try:
                    encrypted_secret_data = encrypt_data(secret_data_to_embed, password, filename)
                    output_filename = f"steg_output_{uploaded_media_file.name}"
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
                            mime="image/png" if "Resim" in media_type else "audio/wav" if "Ses" in media_type else "video/mp4"
                        )
                    else:
                         st.error("Veri gizleme başarısız oldu.")
                except Exception as e:
                    st.error(f"Gizleme sırasında bir hata oluştu: {e}")
        else:
            st.warning("Lütfen tüm alanları doldurun ve dosyaları yükleyin.")
elif operation == "Çöz (Decode)":
    st.header("Çözme (Decode)")
    steg_media_file = st.file_uploader(f"Çözme yapılacak gizlenmiş {media_type.split(' ')[0].lower()} dosyasını yükleyin:", type=["png", "bmp"] if "Resim" in media_type else ["wav"] if "Ses" in media_type else ["mp4", "avi", "mkv"])
    if st.button("Çöz"):
        if steg_media_file is not None and password:
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
                                    file_name=f"decrypted_{retrieved_ext}",
                                )
                        else:
                            st.error("Şifre yanlış veya veri bozuk.")
                    else:
                         st.error("Gizlenmiş dosyadan veri çıkarılamadı.")
                except Exception as e:
                    st.error(f"Çözme sırasında bir hata oluştu: {e}")
        else:
            st.warning("Lütfen gizlenmiş dosyayı yükleyin ve şifreyi girin.")
