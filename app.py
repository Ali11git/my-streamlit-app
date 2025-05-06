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
import numpy as np
from io import BytesIO
import random


# AI görsel oluşturma için basit bir model
# import requests
from huggingface_hub import InferenceClient
HF_TOKEN = st.secrets['HF_TOKEN']
client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)
def generate_ai_image(prompt, width=256, height=256):
    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-3.5-large",
    )
    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output
    # API_URL = "https://huggingface.co/collections/stabilityai/stable-diffusion-35-671785cca799084f71fa2838"
    # headers = {
    #     "Authorization": f"Bearer {HF_TOKEN}",
    #     "Content-Type": "application/json"
    # }
    # data = {
    #     "inputs": prompt
    # }

    # response = requests.post(API_URL, headers=headers, json=data)

    # if response.status_code != 200:
    #     raise Exception(f"AI token error: {response.status_code} - {response.text}")

    # content_type = response.headers.get("content-type")

    # if "image" in content_type:
    #     # Görsel doğrudan binary olarak geldiyse
    #     img = Image.open(BytesIO(response.content))
    # else:
    #     # JSON döndüyse (örneğin {"generated_image": "url"})
    #     result = response.json()
    #     if isinstance(result, list) and "generated_image" in result[0]:
    #         image_url = result[0]["generated_image"]
    #     elif "image" in result:
    #         image_url = result["image"]
    #     elif isinstance(result, list) and isinstance(result[0], str):
    #         image_url = result[0]
    #     else:
    #         raise Exception("Beklenmeyen yanıt formatı")

    #     image_response = requests.get(image_url)
    #     img = Image.open(BytesIO(image_response.content))

    # output = BytesIO()
    # img.save(output, format="PNG")
    # output.seek(0)
    # return output
# def generate_ai_image(prompt, width=256, height=256):
#     """
#     Verilen metne göre basit bir yapay görsel oluşturur.
#     Bu basit model, prompt'tan hash oluşturarak rastgele ama tekrarlanabilir desenler üretir.

#     Args:
#         prompt (str): Görsel için kullanılacak açıklama metni
#         width (int): Oluşturulacak görselin genişliği
#         height (int): Oluşturulacak görselin yüksekliği

#     Returns:
#         BytesIO: PNG formatında oluşturulan görsel
#     """
#     # Prompt'tan tekrarlanabilir bir seed oluştur
#     seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 10000
#     np.random.seed(seed)

#     # Rastgele renk kanalları oluştur
#     r = np.random.randint(0, 255, (height, width), dtype=np.uint8)
#     g = np.random.randint(0, 255, (height, width), dtype=np.uint8)
#     b = np.random.randint(0, 255, (height, width), dtype=np.uint8)

#     # Prompt'un ilk karakterini kullanarak basit bir desen oluştur
#     if len(prompt) > 0:
#         pattern_type = ord(prompt[0]) % 5

#         if pattern_type == 0:  # Yatay çizgiler
#             for i in range(0, height, 10):
#                 r[i:i + 3, :] = np.random.randint(100, 255)
#                 g[i:i + 3, :] = np.random.randint(100, 255)
#                 b[i:i + 3, :] = np.random.randint(100, 255)

#         elif pattern_type == 1:  # Dikey çizgiler
#             for i in range(0, width, 10):
#                 r[:, i:i + 3] = np.random.randint(100, 255)
#                 g[:, i:i + 3] = np.random.randint(100, 255)
#                 b[:, i:i + 3] = np.random.randint(100, 255)

#         elif pattern_type == 2:  # Daireler
#             num_circles = min(len(prompt), 10)
#             for i in range(num_circles):
#                 center_x = np.random.randint(0, width)
#                 center_y = np.random.randint(0, height)
#                 radius = np.random.randint(10, 50)

#                 y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]
#                 mask = x * x + y * y <= radius * radius

#                 r[mask] = np.random.randint(100, 255)
#                 g[mask] = np.random.randint(100, 255)
#                 b[mask] = np.random.randint(100, 255)

#         elif pattern_type == 3:  # Gradyan
#             for i in range(height):
#                 val_r = int(i * 255 / height)
#                 val_g = int((width - i) * 255 / width)
#                 val_b = int((i + width) % 255)
#                 r[i, :] = val_r
#                 g[i, :] = val_g
#                 b[i, :] = val_b

#         else:  # Kareler
#             square_size = 20
#             for i in range(0, height, square_size):
#                 for j in range(0, width, square_size):
#                     if (i + j) % 2 == 0:
#                         r[i:i + square_size, j:j + square_size] = np.random.randint(100, 255)
#                         g[i:i + square_size, j:j + square_size] = np.random.randint(100, 255)
#                         b[i:i + square_size, j:j + square_size] = np.random.randint(100, 255)

#     # RGB kanallarını birleştir
#     image_array = np.stack((r, g, b), axis=-1)

#     # NumPy dizisini PIL Image'e dönüştür
#     img = Image.fromarray(image_array)

#     # BytesIO nesnesine kaydet
#     output = BytesIO()
#     img.save(output, format="PNG")
#     output.seek(0)

#     return output


def encode_lsb(image_file, secret_data, output_filename):
    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Resim dosyası açılamadı veya dönüştürülemedi: {e}")
        return None
    encoded = img.copy()
    width, height = img.size
    index = 0
    secret_data_str = str(secret_data) # Already JSON string
    binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
    binary_secret += '00000000' * 5 # Terminator
    data_len = len(binary_secret)
    total_pixels = width * height
    total_capacity_bits = total_pixels * 3

    if data_len > total_capacity_bits:
        st.warning(
            f"Uyarı: Gizlenecek veri ({data_len} bit) resmin kapasitesini ({total_capacity_bits} bit) aşıyor. Veri kesilebilir.")

    pixel_access = encoded.load() # Daha hızlı piksel erişimi için

    for y in range(height):
        for x in range(width):
            if index < data_len:
                r, g, b = pixel_access[x, y]

                # Embed bits
                if index < data_len:
                    r = (r & ~1) | int(binary_secret[index])
                    index += 1
                if index < data_len:
                    g = (g & ~1) | int(binary_secret[index])
                    index += 1
                if index < data_len:
                    b = (b & ~1) | int(binary_secret[index])
                    index += 1

                pixel_access[x, y] = (r, g, b)
            else:
                # Veri tamamen gömüldüğünde döngüden çık
                img_byte_arr = io.BytesIO()
                encoded.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                return img_byte_arr
        # Bu satır teknik olarak gereksiz ama açıklık için kalabilir
        # if index >= data_len:
        #    break

    # Eğer döngü bitti ve veri bitmediyse uyarı ver (nadiren olmalı ama kontrol edelim)
    if index < data_len:
         st.warning(f"Uyarı: Döngü tamamlandı ancak verinin tamamı ({index}/{data_len} bit) gömülemedi. Bu beklenmedik bir durum.")


    img_byte_arr = io.BytesIO()
    encoded.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def decode_lsb(image_file):
    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Resim dosyası açılamadı veya dönüştürülemedi: {e}")
        return None

    binary_data = ""
    terminator_bits = '00000000' * 5
    found_terminator = False
    pixel_access = img.load() # Daha hızlı piksel erişimi
    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixel_access[x, y]

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
        st.warning("Uyarı: Terminator bulunamadı. Tüm dosya okundu, ancak gizli veri tamamlanmamış olabilir veya dosya LSB ile değiştirilmemiş olabilir.")

    # Ensure binary_data length is a multiple of 8
    if len(binary_data) % 8 != 0:
         st.warning(f"Uyarı: Çıkarılan bit sayısı ({len(binary_data)}) 8'in katı değil. Son eksik bayt atlanıyor.")
         binary_data = binary_data[:-(len(binary_data) % 8)]


    all_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte_str in all_bytes:
         # Zaten yukarıda 8'in katı olmasını sağladık ama kontrol kalabilir
        if len(byte_str) == 8:
            try:
                # Directly try to decode to catch potential issues early if it's JSON
                decoded_data += chr(int(byte_str, 2))
            except ValueError:
                 # Should not happen if byte_str contains only '0' or '1'
                 st.warning(f"Geçersiz bayt dizisi bulundu: {byte_str}. Atlanıyor.")
                 pass # Skip invalid byte sequence
            except Exception as e:
                 st.warning(f"Bayt dönüştürme hatası: {e}. Byte: {byte_str}")
                 pass
    return decoded_data


def encode_lsb_audio(audio_file, secret_data, output_filename):
    st.warning("Ses Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{datetime.datetime.now().timestamp()}_{audio_file.name}"
    temp_output_path_converted = f"temp_steg_converted_{datetime.datetime.now().timestamp()}.wav"
    temp_final_output_path = f"temp_final_output_{datetime.datetime.now().timestamp()}.wav"
    output_bytes = None

    try:
        # Write uploaded file to a temporary input file
        with open(temp_input_path, "wb") as f:
            f.write(audio_file.getvalue())
        print(f"Geçici giriş dosyası oluşturuldu: {temp_input_path}")

        # Convert audio to WAV PCM S16LE using ffmpeg
        # Ensure ffmpeg is in PATH or provide full path
        audio_convert_cmd = f'ffmpeg -i "{temp_input_path}" -acodec pcm_s16le -ar 44100 -ac 1 "{temp_output_path_converted}" -y'
        print(f"FFmpeg dönüştürme komutu çalıştırılıyor: {audio_convert_cmd}")
        process = subprocess.run(audio_convert_cmd, shell=True, capture_output=True, text=True)

        if process.returncode != 0 or not os.path.exists(temp_output_path_converted):
            st.error(f"Hata: Ses dönüştürme başarısız oldu (ffmpeg çıkış kodu: {process.returncode}). '{temp_output_path_converted}' oluşturulamadı.")
            st.error(f"FFmpeg Hata Mesajı: {process.stderr}")
            print(f"Hata: Ses dönüştürme başarısız oldu. FFmpeg stderr: {process.stderr}")
            return None
        print(f"Ses başarıyla WAV formatına dönüştürüldü: {temp_output_path_converted}")

        # Prepare secret data
        secret_data_str = str(secret_data) # Already JSON string
        binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
        binary_secret += '00000000' * 5  # Terminator

        # Read converted WAV file
        with wave.open(temp_output_path_converted, 'rb') as wf:
            params = wf.getparams()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth() # Sample width (bytes)
            nchannels = wf.getnchannels()
            print(f"Okunan WAV parametreleri: {params}")

        if sampwidth != 2:
             st.error(f"Hata: Beklenmedik örnek genişliği ({sampwidth} bayt). Yalnızca 16-bit (2 bayt) PCM desteklenir.")
             return None


        audio_bytes = bytearray(audio_data)
        data_index = 0
        data_len = len(binary_secret)
        total_bits_possible = len(audio_bytes) # Each byte's LSB can be used

        if data_len > total_bits_possible:
            st.warning(
                f"Uyarı: Gömülecek veri boyutu ({data_len} bit), ses dosyasının kapasitesini ({total_bits_possible} bit) aşıyor. Tüm veri gömülemeyebilir.")
            print(
                f"Uyarı: Gömülecek veri boyutu ({data_len} bit), ses dosyasının kapasitesini ({total_bits_possible} bit) aşıyor.")

        progress_text = "Ses baytları işleniyor ve veri gömülüyor..."
        progress_bar = st.progress(0.0, text=progress_text)

        # Embed data into LSB of each byte
        for i in range(len(audio_bytes)):
            if data_index < data_len:
                audio_bytes[i] = (audio_bytes[i] & 0xFE) | int(binary_secret[data_index])
                data_index += 1
            else:
                break # All data embedded

            # Update progress bar periodically to avoid slowing down too much
            if i % 10000 == 0: # Update every 10000 bytes
                 progress = min(data_index / data_len, 1.0) if data_len > 0 else 1.0
                 try:
                     progress_bar.progress(progress, text=f"{progress_text} ({data_index}/{data_len} bit)")
                 except st.errors.StreamlitAPIException: # Handle potential error if element disappears
                     pass


        if 'progress_bar' in locals(): progress_bar.empty()
        print(f"Ses işleme tamamlandı. Toplam {data_index} bit işlendi.")

        if data_index < data_len:
            st.warning(f"Uyarı: Tüm veri sese sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
            print(f"Uyarı: Tüm veri sese sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")

        # Write modified bytes to a final temporary WAV file
        with wave.open(temp_final_output_path, 'wb') as wf_out:
            wf_out.setparams(params) # Use original parameters
            wf_out.writeframes(audio_bytes)
        print(f"Veri geçici olarak '{temp_final_output_path}' dosyasına yazıldı.")

        # Read the final output file into bytes
        if os.path.exists(temp_final_output_path):
            with open(temp_final_output_path, "rb") as f:
                output_bytes = f.read()
            print(f"'{temp_final_output_path}' dosyasından bayt verisi okundu.")
        else:
             st.error(f"Hata: Nihai çıktı WAV dosyası '{temp_final_output_path}' oluşturulamadı veya bulunamadı.")
             print(f"Hata: Nihai çıktı WAV dosyası '{temp_final_output_path}' oluşturulamadı veya bulunamadı.")
             return None


        return output_bytes

    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosya veya komut (ffmpeg?) bulunamadı: {e}")
        print(f"Hata: Gerekli dosya bulunamadı. {e}")
        return None
    except wave.Error as e:
        st.error(f"WAV dosyası işlenirken hata oluştu: {e}")
        print(f"WAV dosyası işlenirken hata oluştu: {e}")
        return None
    except ValueError as e:
         st.error(f"Veri veya parametre hatası: {e}")
         print(f"ValueError: {e}")
         return None
    except Exception as e:
        st.error(f"Ses kodlama sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(f"Beklenmedik Hata: {e}\n{traceback.format_exc()}")
        return None
    finally:
        # Clean up temporary files
        print("Geçici dosyalar temizleniyor...")
        for temp_file in [temp_input_path, temp_output_path_converted, temp_final_output_path]:
             if os.path.exists(temp_file):
                 try:
                     os.remove(temp_file)
                     print(f"Temizlendi: {temp_file}")
                 except OSError as e:
                     print(f"Hata: Geçici dosya silinemedi '{temp_file}': {e}")
                     st.warning(f"Geçici dosya '{temp_file}' silinemedi.")


def decode_lsb_audio(audio_file):
    # Input is already a BytesIO object or similar from Streamlit uploader
    # No need to save to disk for decoding WAV if wave module supports BytesIO
    audio_byte_arr = io.BytesIO(audio_file.getvalue())
    try:
        with wave.open(audio_byte_arr, 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
            print(f"Okunan WAV örnek genişliği: {sampwidth} bayt")

        # if sampwidth != 2:
        #     # Allow decoding even if sampwidth is not 2, but warn
        #     st.warning(f"Uyarı: Ses dosyasının örnek genişliği ({sampwidth} bayt) 16-bit değil. LSB çıkarma işlemi yine de deneniyor ancak sonuç hatalı olabilir.")
        #     # Proceed with caution

        audio_bytes = bytearray(audio_data)
        binary_data = ""
        terminator_bits = '00000000' * 5
        found_terminator = False

        progress_text = "Ses baytları çözümleniyor..."
        progress_bar = st.progress(0.0, text=progress_text)
        total_bytes = len(audio_bytes)

        for i, byte in enumerate(audio_bytes):
            binary_data += str(byte & 1)
            if len(binary_data) >= len(terminator_bits) and binary_data[-len(terminator_bits):] == terminator_bits:
                binary_data = binary_data[:-len(terminator_bits)]
                found_terminator = True
                break

            # Update progress bar periodically
            if i % 10000 == 0: # Update every 10000 bytes
                progress = min((i+1) / total_bytes, 1.0) if total_bytes > 0 else 1.0
                try:
                    progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_bytes} bayt)")
                except st.errors.StreamlitAPIException:
                    pass # Ignore if element is gone

        if 'progress_bar' in locals(): progress_bar.empty()

        if not found_terminator:
            st.warning("Uyarı: Terminator bulunamadı. Tüm dosya okundu, ancak gizli veri tamamlanmamış olabilir veya dosya LSB ile değiştirilmemiş olabilir.")

        # Ensure binary_data length is a multiple of 8
        if len(binary_data) % 8 != 0:
             st.warning(f"Uyarı: Çıkarılan bit sayısı ({len(binary_data)}) 8'in katı değil. Son eksik bayt atlanıyor.")
             binary_data = binary_data[:-(len(binary_data) % 8)]


        all_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
        decoded_data = ""
        for byte_str in all_bytes:
            if len(byte_str) == 8:
                try:
                    decoded_data += chr(int(byte_str, 2))
                except ValueError:
                    st.warning(f"Geçersiz bayt dizisi bulundu: {byte_str}. Atlanıyor.")
                    pass # Skip invalid byte sequence
                except Exception as e:
                    st.warning(f"Bayt dönüştürme hatası: {e}. Byte: {byte_str}")
                    pass

        return decoded_data

    except wave.Error as e:
        st.error(f"WAV dosyası okunurken veya işlenirken hata oluştu: {e}. Dosya geçerli bir WAV dosyası mı?")
        print(f"WAV dosyası hatası: {e}")
        return None
    except Exception as e:
        st.error(f"Ses çözme sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(f"Beklenmedik Hata: {e}\n{traceback.format_exc()}")
        return None


def encode_lsb_video(video_file, secret_data, output_filename):
    st.warning("Video Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır. Bu işlem uzun sürebilir.")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f") # Add microseconds for uniqueness
    temp_input_path = f"temp_input_{timestamp}_{video_file.name}"
    # Use a lossless intermediate format like HuffYUV in AVI
    temp_output_path_video_only = f"temp_steg_video_only_{timestamp}.avi"
    # Extract audio to a common format, AAC is good, but check source or use WAV
    temp_audio_extracted = f"temp_audio_{timestamp}.aac" # Or .wav if preferred
    final_output_path = f"{timestamp}_{output_filename}" # Provided name includes timestamp now
    output_video_bytes = None

    try:
        # 1. Save uploaded video to temp file
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getvalue())
        print(f"Geçici giriş dosyası oluşturuldu: '{temp_input_path}'")

        # 2. Check for audio stream and extract if present
        audio_exists = False
        audio_codec_to_use = 'copy' # Default to copy if possible
        try:
            # Use ffprobe to get info about streams
            ffprobe_cmd = f'ffprobe -v error -show_entries stream=index,codec_type,codec_name -of csv=p=0 "{temp_input_path}"'
            print(f"FFprobe komutu çalıştırılıyor: {ffprobe_cmd}")
            result = subprocess.run(ffprobe_cmd, shell=True, capture_output=True, text=True, check=True)
            streams = result.stdout.strip().split('\n')
            print(f"FFprobe çıktı (streamler): {streams}")
            for stream in streams:
                if "audio" in stream:
                    audio_exists = True
                    audio_codec_name = stream.split(',')[2]
                    print(f"Giriş dosyasında ses akışı bulundu. Codec: {audio_codec_name}")
                    # If original codec is problematic for simple copy in target container, consider re-encoding
                    # For simplicity now, we'll stick with copy or a common format like AAC
                    # audio_codec_to_use = 'aac -b:a 128k' # Example re-encode
                    break # Assume one audio stream for simplicity
            if not audio_exists:
                st.info("Giriş videosunda ses akışı bulunamadı.")
                print("Giriş videosunda ses akışı bulunamadı.")
        except subprocess.CalledProcessError as e:
            st.warning(f"ffprobe çalıştırılamadı veya hata verdi (kod: {e.returncode}). Ses kontrolü yapılamadı. Hata: {e.stderr}")
            print(f"ffprobe hatası: {e.stderr}")
            audio_exists = False # Assume no audio if ffprobe fails
        except FileNotFoundError:
            st.warning("ffprobe komutu bulunamadı. PATH ortam değişkeninizi kontrol edin. Ses akışı kopyalanamayacak.")
            print("ffprobe bulunamadı.")
            audio_exists = False

        # 3. Extract audio if it exists
        if audio_exists:
            st.info("Orijinal ses akışı çıkarılıyor...")
            # Use the determined codec (copy or re-encode)
            audio_extract_cmd = f'ffmpeg -i "{temp_input_path}" -vn -acodec {audio_codec_to_use} -y "{temp_audio_extracted}"'
            print(f"Ses çıkarma komutu: {audio_extract_cmd}")
            process = subprocess.run(audio_extract_cmd, shell=True, capture_output=True, text=True)
            if process.returncode != 0 or not os.path.exists(temp_audio_extracted):
                st.error(f"Hata: Ses çıkarma başarısız oldu (ffmpeg çıkış kodu: {process.returncode}). '{temp_audio_extracted}' oluşturulamadı.")
                st.error(f"FFmpeg Hata Mesajı: {process.stderr}")
                print(f"Hata: Ses çıkarma başarısız oldu. FFmpeg stderr: {process.stderr}")
                # Decide if we should continue without audio or stop
                st.warning("Ses çıkarılamadığı için işleme ses olmadan devam edilecek.")
                audio_exists = False # Mark audio as not available for muxing
                # return None # Option to stop completely
            else:
                 print(f"Ses başarıyla çıkarıldı: {temp_audio_extracted}")
                 st.success("Ses akışı başarıyla çıkarıldı.")


        # 4. Process video frames with OpenCV
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Hata: Giriş video dosyası '{temp_input_path}' OpenCV ile açılamadı.")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video özellikleri: {width}x{height} @ {fps:.2f} fps, Toplam Kare: {total_frames_in_video}")

        # Use HuffYUV (lossless) in AVI container for intermediate video
        # Note: Requires ffmpeg build with huffyuv encoder, or choose another lossless like ffv1
        # fourcc = cv2.VideoWriter_fourcc(*'HFYU') # HuffYUV
        # If HFYU is not available, try FFV1 (generally available)
        fourcc = cv2.VideoWriter_fourcc(*'FFV1') # FFV1 Lossless
        out = cv2.VideoWriter(temp_output_path_video_only, fourcc, fps, (width, height))

        if not out.isOpened():
            st.error(f"Hata: Geçici çıkış video dosyası '{temp_output_path_video_only}' yazılamadı. Codec ('FFV1') destekleniyor mu?")
            cap.release()
            return None
        print(f"Geçici çıkış video dosyası için VideoWriter oluşturuldu: '{temp_output_path_video_only}' (Codec: FFV1)")

        # Prepare secret data
        secret_data_str = str(secret_data) # Already JSON string
        binary_secret = ''.join([format(ord(i), '08b') for i in secret_data_str])
        binary_secret += '00000000' * 5 # Terminator
        data_index = 0
        data_len = len(binary_secret)

        # Estimate capacity (approximate)
        bits_per_frame = width * height * 3 # 3 color channels, 1 bit per channel LSB
        total_bits_possible = bits_per_frame * total_frames_in_video if total_frames_in_video > 0 else float('inf')

        if data_len > total_bits_possible and total_frames_in_video > 0 :
             st.warning(
                 f"Uyarı: Gömülecek veri boyutu ({data_len} bit), videonun tahmini kapasitesini (~{total_bits_possible} bit) aşıyor. Tüm veri gömülemeyebilir.")

        embedded_fully = False
        progress_text = "Video kareleri işleniyor ve veri gömülüyor..."
        progress_bar = st.progress(0.0, text=progress_text)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video akışının sonuna ulaşıldı.")
                break # End of video stream

            frame_count += 1
            # Update progress based on frame count
            if total_frames_in_video > 0:
                 progress = min(frame_count / total_frames_in_video, 1.0)
                 current_text = progress_text if not embedded_fully else f"Kalan kareler kopyalanıyor ({frame_count}/{total_frames_in_video})..."
                 try:
                     progress_bar.progress(progress, text=f"{current_text}")
                 except st.errors.StreamlitAPIException:
                     pass # Ignore if element is gone


            if not embedded_fully and data_index < data_len:
                # Iterate through pixels and color channels to embed data
                # Using nested loops is slow in Python; NumPy operations are faster
                # but harder to implement for sequential bit embedding with early exit.
                # Stick with loops for clarity, accept performance hit.
                frame_modified = False
                for y in range(height):
                    for x in range(width):
                        # frame[y, x] is BGR in OpenCV
                        for c in range(3): # B, G, R channels
                            if data_index < data_len:
                                # Get current LSB
                                current_lsb = frame[y, x, c] & 1
                                # Get desired bit
                                desired_bit = int(binary_secret[data_index])
                                # Modify pixel if LSB needs changing
                                if current_lsb != desired_bit:
                                     frame[y, x, c] = (frame[y, x, c] & 0xFE) | desired_bit
                                     frame_modified = True
                                data_index += 1
                            else:
                                # All data embedded
                                embedded_fully = True
                                print(f"Veri {frame_count}. karede tamamen gömüldü.")
                                break # Break color channel loop
                        if embedded_fully: break # Break x loop
                    if embedded_fully: break # Break y loop

            # Write the frame (original or modified) to the output video
            out.write(frame)

            # Optimization: If data is fully embedded, stop processing pixels for remaining frames
            if embedded_fully:
                 # No need to process pixels anymore, just copy remaining frames
                 # The current loop structure already handles this by skipping the embedding block
                 pass


        if 'progress_bar' in locals(): progress_bar.empty()
        print(f"Video kare işleme tamamlandı. Toplam {frame_count} kare işlendi/yazıldı.")

        if data_index < data_len:
            st.warning(f"Uyarı: Tüm veri videoya sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")
            print(f"Uyarı: Tüm veri videoya sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")

        # Release video resources
        cap.release()
        out.release()
        # cv2.destroyAllWindows() # Not needed in script context
        print("OpenCV Kaynakları (VideoCapture, VideoWriter) serbest bırakıldı.")

        # 5. Mux video and audio (if audio exists) using ffmpeg
        st.info("Son çıktı dosyası oluşturuluyor...")
        if audio_exists and os.path.exists(temp_audio_extracted):
            st.info("LSB uygulanmış video ile çıkarılan ses birleştiriliyor...")
            # Use a common output container like MP4 or MKV. AVI with AAC can be problematic.
            # Let's target MP4 for better compatibility. FFV1 video might need re-encoding for MP4.
            # Option 1: Keep lossless video (if target player supports FFV1 in MP4) -c:v copy
            # Option 2: Re-encode video to H.264 (lossy but compatible) -c:v libx264
            # Option 3: Output MKV which handles FFV1 better - target MKV container
            # Let's try MKV first as it's flexible.
            final_output_path = os.path.splitext(final_output_path)[0] + ".mkv" # Change extension
            # -shortest: Finish encoding when the shortest input stream ends (video or audio)
            video_mux_cmd = (f'ffmpeg -i "{temp_output_path_video_only}" -i "{temp_audio_extracted}" '
                             f'-c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest '
                             f'-y "{final_output_path}"')
            print(f"Birleştirme (muxing) komutu: {video_mux_cmd}")
            process = subprocess.run(video_mux_cmd, shell=True, capture_output=True, text=True)

            if process.returncode != 0 or not os.path.exists(final_output_path):
                st.error(f"Hata: Video ve ses birleştirme (muxing) başarısız oldu (ffmpeg çıkış kodu: {process.returncode}).")
                st.error(f"FFmpeg Hata Mesajı: {process.stderr}")
                print(f"Hata: Muxing başarısız oldu. FFmpeg stderr: {process.stderr}")
                # Fallback: Provide the video-only file?
                st.warning("Birleştirme başarısız olduğu için sadece LSB uygulanmış video döndürülüyor (eğer mevcutsa).")
                if os.path.exists(temp_output_path_video_only):
                    with open(temp_output_path_video_only, "rb") as f:
                        output_video_bytes = f.read()
                    st.info("Sadece video dosyası indirilebilir.")
                    final_output_path = temp_output_path_video_only # Update path for download button
                else:
                    return None # Critical failure
            else:
                print(f"Birleştirme tamamlandı. Nihai çıktı: {final_output_path}")
                st.success(f"Veri başarıyla videoya gizlendi ve orijinal ses eklendi: '{os.path.basename(final_output_path)}'")
                with open(final_output_path, "rb") as f:
                    output_video_bytes = f.read()
                print(f"Nihai çıktı dosyası '{final_output_path}' bayt olarak okundu.")

        else:
             # No audio stream or extraction failed
             st.warning("Ses akışı bulunmadığı veya çıkarılamadığı için sadece LSB uygulanmış video döndürülüyor.")
             # Rename the intermediate video file to the final name (or copy)
             # Ensure the extension matches the intermediate format (AVI in this case)
             final_output_path = os.path.splitext(final_output_path)[0] + ".avi" # Match intermediate container
             try:
                 if os.path.exists(temp_output_path_video_only):
                      # Rename might fail across different filesystems/drives, copy is safer
                      # os.rename(temp_output_path_video_only, final_output_path)
                      import shutil
                      shutil.copy2(temp_output_path_video_only, final_output_path)
                      print(f"Video-only çıktı dosyası '{final_output_path}' olarak kopyalandı/yeniden adlandırıldı.")
                      with open(final_output_path, "rb") as f:
                          output_video_bytes = f.read()
                      st.success(f"Veri başarıyla videoya gizlendi (sadece video): '{os.path.basename(final_output_path)}'")
                 else:
                     st.error(f"Hata: Ses akışı yok ve LSB uygulanmış video dosyası ('{temp_output_path_video_only}') bulunamadı.")
                     return None
             except Exception as e_rename:
                  st.error(f"Hata: Video-only çıktı dosyası kopyalanamadı/yeniden adlandırılamadı: {e_rename}")
                  # Fallback to reading the temp file directly if rename/copy failed
                  if os.path.exists(temp_output_path_video_only):
                      with open(temp_output_path_video_only, "rb") as f:
                           output_video_bytes = f.read()
                      final_output_path = temp_output_path_video_only # Use temp path for download
                      st.warning("Dosya kopyalanamadı/yeniden adlandırılamadı, geçici dosya kullanılıyor.")
                  else:
                       return None


        # Ensure final_output_path is set correctly for the download button
        if output_video_bytes:
            # Pass the final path and bytes to the main part of the script
             return output_video_bytes, final_output_path # Return bytes and the final filename
        else:
             return None, None


    except cv2.error as e:
         st.error(f"OpenCV hatası oluştu: {e}")
         print(f"OpenCV Hatası: {e}")
         return None, None
    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosya veya komut (ffmpeg?, ffprobe?) bulunamadı: {e}")
        print(f"Hata: Dosya bulunamadı. {e}")
        return None, None
    except subprocess.CalledProcessError as e:
         st.error(f"ffmpeg/ffprobe komutu çalıştırılırken hata oluştu (kod: {e.returncode}): {e.stderr}")
         print(f"Subprocess Hatası: {e.stderr}")
         return None, None
    except Exception as e:
        st.error(f"Video kodlama sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(f"Beklenmedik Hata: {e}\n{traceback.format_exc()}")
        return None, None
    finally:
        # Clean up all temporary files
        print("Geçici dosyalar temizleniyor...")
        # Release CV resources if not already done (belt and suspenders)
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals() and out.isOpened(): out.release()

        files_to_clean = [temp_input_path, temp_output_path_video_only, temp_audio_extracted, final_output_path]
        # Add the actual final output path only if it wasn't the intended file to keep (e.g., if muxing failed and we returned the temp video)
        # However, since we return the bytes, we SHOULD clean the final file path from disk.
        # The caller (Streamlit button) handles the download from bytes.

        for temp_file in files_to_clean:
             if temp_file and os.path.exists(temp_file): # Check if path is not None
                 try:
                     os.remove(temp_file)
                     print(f"Temizlendi: {temp_file}")
                 except OSError as e:
                     print(f"Hata: Geçici dosya silinemedi '{temp_file}': {e}")
                     st.warning(f"Geçici dosya '{temp_file}' silinemedi.")

# Helper function for decode_lsb_video
def extract_lsb_from_frame(frame, binary_data_list, terminator_bits):
    """Extracts LSBs from a single frame until terminator is found or frame ends."""
    height, width, _ = frame.shape
    found_terminator_in_frame = False
    for y in range(height):
        for x in range(width):
            # Extract LSB from B, G, R channels
            for c in range(3):
                binary_data_list.append(str(frame[y, x, c] & 1))
                # Check for terminator efficiently
                if len(binary_data_list) >= len(terminator_bits):
                     # Check last N bits directly without string conversion/slicing if possible
                     # For simplicity, stick to string check
                     current_suffix = "".join(binary_data_list[-len(terminator_bits):])
                     if current_suffix == terminator_bits:
                         found_terminator_in_frame = True
                         # Remove terminator bits
                         del binary_data_list[-len(terminator_bits):]
                         return found_terminator_in_frame # Found it
            if found_terminator_in_frame: break # Should not be reached if return is used
        if found_terminator_in_frame: break # Should not be reached if return is used
    return found_terminator_in_frame # Did not find terminator in this frame


def decode_lsb_video(video_file):
    st.warning("Video Steganografi çözümleme işlemi disk üzerinde geçici dosyalar oluşturabilir.")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    temp_input_path = f"temp_input_decode_{timestamp}_{video_file.name}"
    decoded_data = None

    try:
        # 1. Save uploaded video to temp file
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getvalue())
        print(f"Geçici giriş dosyası (çözme) oluşturuldu: '{temp_input_path}'")

        # 2. Open video with OpenCV
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Hata: Giriş video dosyası '{temp_input_path}' OpenCV ile açılamadı. Dosya formatı (örn: AVI, MKV) destekleniyor mu?")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Çözülecek video özellikleri: {width}x{height}, Toplam Kare: {total_frames}")

        # 3. Extract LSBs frame by frame
        binary_data_list = [] # Use list for efficient appending
        terminator_bits = '00000000' * 5
        found_terminator = False
        progress_text = "Video kareleri çözümleniyor..."
        progress_bar = st.progress(0.0, text=progress_text)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video akışının sonuna ulaşıldı (çözme).")
                break # End of video

            frame_count += 1
            # Update progress
            if total_frames > 0:
                 progress = min(frame_count / total_frames, 1.0)
                 try:
                      progress_bar.progress(progress, text=f"{progress_text} (Kare {frame_count}/{total_frames})")
                 except st.errors.StreamlitAPIException:
                      pass # Ignore if element is gone

            # Extract LSBs from this frame
            found_terminator_in_frame = extract_lsb_from_frame(frame, binary_data_list, terminator_bits)

            if found_terminator_in_frame:
                found_terminator = True
                print(f"Terminator {frame_count}. karede bulundu.")
                break # Stop processing frames

        if 'progress_bar' in locals(): progress_bar.empty()
        cap.release()
        # cv2.destroyAllWindows()
        print("OpenCV VideoCapture kaynağı serbest bırakıldı (çözme).")

        if not found_terminator:
            st.warning("Uyarı: Terminator bulunamadı. Tüm video okundu, ancak gizli veri tamamlanmamış olabilir veya dosya LSB ile değiştirilmemiş olabilir.")

        # 4. Convert extracted bits to data
        binary_data = "".join(binary_data_list)
        print(f"Toplam {len(binary_data)} bit çıkarıldı.")

        # Ensure binary_data length is a multiple of 8
        remainder = len(binary_data) % 8
        if remainder != 0:
            st.warning(f"Uyarı: Çıkarılan bit sayısı ({len(binary_data)}) 8'in katı değil. Son {remainder} bit atlanıyor.")
            binary_data = binary_data[:-remainder]

        all_bytes_str = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
        # Convert to characters/bytes
        # Important: Assume the result is the JSON string from encryption
        decoded_json_str = ""
        try:
            byte_list = []
            for byte_s in all_bytes_str:
                 byte_list.append(int(byte_s, 2))

            # Attempt to decode as UTF-8 first to see if it's valid JSON text
            decoded_json_str = bytearray(byte_list).decode('utf-8')
            # Validate if it's JSON (minimal check)
            if not (decoded_json_str.startswith('{') and decoded_json_str.endswith('}')):
                 st.warning("Çıkarılan veri UTF-8 metin gibi görünüyor ancak geçerli JSON yapısı (başlangıç/bitiş { }) beklenmiyor.")
            print("Çıkarılan veri UTF-8 olarak başarıyla çözüldü (JSON bekleniyor).")
            decoded_data = decoded_json_str # Return the JSON string

        except UnicodeDecodeError:
             st.error("Hata: Çıkarılan baytlar geçerli UTF-8 (JSON) olarak çözülemedi. Veri bozuk veya farklı bir formatta olabilir.")
             print("Hata: Çıkarılan baytlar UTF-8 değil.")
             # Optionally, return the raw bytes if decoding fails? Risky.
             decoded_data = None # Indicate failure
        except ValueError as e:
             # This might happen if int(byte_s, 2) fails, though unlikely with '0'/'1'
             st.error(f"Hata: İkili dize bayta dönüştürülürken hata oluştu: {e}")
             print(f"Hata: int(byte_s, 2) hatası: {e}")
             decoded_data = None
        except Exception as e:
             st.error(f"Çıkarılan veriyi işlerken beklenmedik hata: {e}")
             import traceback
             print(f"Beklenmedik Hata (veri işleme): {e}\n{traceback.format_exc()}")
             decoded_data = None

        return decoded_data

    except cv2.error as e:
         st.error(f"OpenCV hatası oluştu (çözme): {e}")
         print(f"OpenCV Hatası (çözme): {e}")
         return None
    except Exception as e:
        st.error(f"Video çözme sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(f"Beklenmedik Hata (çözme): {e}\n{traceback.format_exc()}")
        return None
    finally:
        # Clean up temporary input file
        if 'cap' in locals() and cap.isOpened(): cap.release() # Ensure release
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
                print(f"Temizlendi (çözme): {temp_input_path}")
            except OSError as e:
                print(f"Hata: Geçici dosya silinemedi (çözme) '{temp_input_path}': {e}")
                st.warning(f"Geçici dosya '{temp_input_path}' silinemedi.")


def encrypt_data(data_bytes, key_string, original_filename=None):
    """Encrypts bytes using AES-CBC and returns a JSON string."""
    if not isinstance(data_bytes, bytes):
         st.error("Şifreleme hatası: Girdi 'bytes' türünde olmalı.")
         print("Şifreleme hatası: Girdi 'bytes' türünde değil.")
         # Attempt conversion assuming UTF-8 text if it's string
         if isinstance(data_bytes, str):
              try:
                   data_bytes = data_bytes.encode('utf-8')
                   print("Girdi str idi, utf-8 olarak encode edildi.")
              except Exception as e:
                   st.error(f"Girdi str->bytes dönüştürme hatası: {e}")
                   return None
         else:
              st.error("Desteklenmeyen girdi türü şifreleme için.")
              return None


    try:
        key = hashlib.sha256(key_string.encode('utf-8')).digest() # 32 bytes key
        cipher = AES.new(key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data_bytes, AES.block_size))
        iv = base64.b64encode(cipher.iv).decode('utf-8')
        ct = base64.b64encode(ct_bytes).decode('utf-8')
        result = {'iv': iv, 'ciphertext': ct}
        # Store the original filename within the JSON payload
        if original_filename:
             # Basic sanitization: remove path, keep only filename
             result['filename'] = os.path.basename(original_filename)
             print(f"Şifrelenmiş veriye dosya adı eklendi: {result['filename']}")

        return json.dumps(result) # Return JSON string
    except Exception as e:
         st.error(f"Şifreleme sırasında hata: {e}")
         import traceback
         print(f"Şifreleme Hatası: {e}\n{traceback.format_exc()}")
         return None


def decrypt_data(json_input_str, key_string):
    """Decrypts data from a JSON string using AES-CBC."""
    try:
        key = hashlib.sha256(key_string.encode('utf-8')).digest()
        b64 = json.loads(json_input_str) # Load JSON from string
        iv = base64.b64decode(b64['iv'])
        ct = base64.b64decode(b64['ciphertext'])
        retrieved_filename = b64.get('filename') # Get filename if stored
        cipher = AES.new(key, AES.MODE_CBC, iv)
        pt_bytes = unpad(cipher.decrypt(ct), AES.block_size)
        print(f"Şifre çözme başarılı. Çıkarılan dosya adı: {retrieved_filename}")
        return pt_bytes, retrieved_filename  # Return decrypted bytes and original filename

    except (ValueError, KeyError) as e:
        # Common errors: PaddingError (wrong key?), KeyError (bad JSON), Base64 error
        st.error(f"Şifre çözme hatası: Veri bozuk veya şifre yanlış olabilir. Hata: {e}")
        print(f"Şifre çözme ValueError/KeyError: {e}")
        return None, None
    except json.JSONDecodeError as e:
         st.error(f"Şifre çözme hatası: Girdi geçerli bir JSON değil. Hata: {e}")
         print(f"Şifre çözme JSONDecodeError: {e}")
         return None, None
    except Exception as e:
        st.error(f"Beklenmedik bir şifre çözme hatası oluştu: {e}")
        import traceback
        print(f"Beklenmedik Şifre Çözme Hatası: {e}\n{traceback.format_exc()}")
        return None, None


# --- Streamlit UI ---

st.title("🔒 Steganografi Uygulaması")
st.markdown("Verilerinizi resim, ses veya video dosyaları içine gizleyin ve şifreleyin.")
st.markdown("---")

# Sidebar for main options
operation = st.sidebar.radio("Yapmak istediğiniz işlemi seçin:", ("Gizle (Encode)", "Çöz (Decode)"))
media_type = st.sidebar.selectbox("Medya türünü seçin:",
                                  ("Resim (Image)", "Ses (Audio)", "Video (Video)"))
password = st.sidebar.text_input("Şifreyi girin (Gizleme ve Çözme için gerekli):", type="password")


# --- Encode Operation ---
if operation == "Gizle (Encode)":
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
             image_paths = []
             rndpath = ""
             for rndimg in os.listdir("images"):
                 image_paths.append(f"images/{rndimg}")
             # rndpath =  random.choice(image_paths)
             # if os.path.exists(rndpath):
             #     st.image(rndpath, caption=f"Varsayılan: {os.path.basename(rndpath)}", use_container_width=True)
             #     if st.button("Resim Değiştir"):
             #         rndpath =  random.choice(image_paths)
             # image_path = BytesIO()
             # img = Image.open(rndpath)
             # img.save(image_path, format="PNG")
             # image_path.seek(0)
             if 'image_path' not in st.session_state:
                 st.session_state.image_path = None
             if 'rndimage' not in st.session_state:
                 st.session_state.rndimage = ""
             col_1, col_2 = st.columns(2)
             with col_1:
                 if st.button("Resim Oluştur/Değiştir"):
                     rndpath =  random.choice(image_paths)
                     image_path = BytesIO()
                     img = Image.open(rndpath)
                     img.save(image_path, format="PNG")
                     image_path.seek(0)
                     st.session_state.image_path = image_path
                     st.session_state.rndimage = os.path.basename(rndpath)
             if st.session_state.image_path:
                 with col_2:
                     st.image(st.session_state.image_path, caption=f"Varsayılan: {st.session_state.rndimage}", use_container_width=True)
                     st.session_state.image_path.seek(0)
                     uploaded_media_file = st.session_state.image_path
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
            if st.session_state.ai_generated_image is None and st.session_state.image_path is None:
                 st.error("Lütfen önce bir AI görseli oluşturun veya 'Dosya yükle' seçeneğini kullanın.")
                 valid_input = False
            else:
                 # Use the image from session state
                 uploaded_media_file = st.session_state.image_path or st.session_state.ai_generated_image
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


# --- Decode Operation ---
elif operation == "Çöz (Decode)":
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
