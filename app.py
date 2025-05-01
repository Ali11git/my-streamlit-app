# -*- coding: utf-8 -*-
import streamlit as st
import asyncio
import sys

# asyncio fix for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(
    page_title="Steganografi Uygulaması",
    page_icon="🔒"
)

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad # Still needed for GCM if data isn't block aligned, though GCM handles padding internally conceptually.
from Crypto.Protocol.KDF import PBKDF2 # Use PBKDF2 for key derivation
from Crypto.Hash import SHA256 # Use SHA256 within PBKDF2
from Crypto.Random import get_random_bytes
import base64
import hashlib # Keep for other potential uses if any, but not for key derivation directly
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

# --- Constants ---
SALT_SIZE = 16 # Size of the salt for PBKDF2
KEY_SIZE = 32 # AES-256 key size
NONCE_SIZE = 16 # AES-GCM nonce size
TAG_SIZE = 16 # AES-GCM authentication tag size
PBKDF2_ITERATIONS = 100000 # Number of iterations for PBKDF2
TERMINATOR = b'\x00' * 10 # Use bytes terminator, slightly longer
TERMINATOR_BITS = ''.join([format(byte, '08b') for byte in TERMINATOR])
MAX_FILE_SIZE_MB_MEDIA = 15 # Max size for cover media
MAX_FILE_SIZE_MB_SECRET = 30 # Max size for secret file (allow larger as it might compress well)

# --- Manage App ---
# AI görsel oluşturma için basit bir model
def generate_ai_image(prompt, width=256, height=256):
    """
    Verilen metne göre basit bir yapay görsel oluşturur.
    Bu basit model, prompt'tan hash oluşturarak rastgele ama tekrarlanabilir desenler üretir.

    Args:
        prompt (str): Görsel için kullanılacak açıklama metni
        width (int): Oluşturulacak görselin genişliği
        height (int): Oluşturulacak görselin yüksekliği

    Returns:
        BytesIO: PNG formatında oluşturulan görsel
    """
    # Prompt'tan tekrarlanabilir bir seed oluştur
    seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 10000
    np.random.seed(seed)

    # Rastgele renk kanalları oluştur
    r = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    g = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    b = np.random.randint(0, 255, (height, width), dtype=np.uint8)

    # Prompt'un ilk karakterini kullanarak basit bir desen oluştur
    if len(prompt) > 0:
        pattern_type = ord(prompt[0]) % 5

        if pattern_type == 0:  # Yatay çizgiler
            for i in range(0, height, 10):
                r[i:i + 3, :] = np.random.randint(100, 255)
                g[i:i + 3, :] = np.random.randint(100, 255)
                b[i:i + 3, :] = np.random.randint(100, 255)

        elif pattern_type == 1:  # Dikey çizgiler
            for i in range(0, width, 10):
                r[:, i:i + 3] = np.random.randint(100, 255)
                g[:, i:i + 3] = np.random.randint(100, 255)
                b[:, i:i + 3] = np.random.randint(100, 255)

        elif pattern_type == 2:  # Daireler
            num_circles = min(len(prompt), 10)
            for i in range(num_circles):
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
                radius = np.random.randint(10, 50)

                y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]
                mask = x * x + y * y <= radius * radius

                r[mask] = np.random.randint(100, 255)
                g[mask] = np.random.randint(100, 255)
                b[mask] = np.random.randint(100, 255)

        elif pattern_type == 3:  # Gradyan
            for i in range(height):
                val_r = int(i * 255 / height)
                val_g = int((width - i) * 255 / width)
                val_b = int((i + width) % 255)
                r[i, :] = val_r
                g[i, :] = val_g
                b[i, :] = val_b

        else:  # Kareler
            square_size = 20
            for i in range(0, height, square_size):
                for j in range(0, width, square_size):
                    if (i + j) % 2 == 0:
                        r[i:i + square_size, j:j + square_size] = np.random.randint(100, 255)
                        g[i:i + square_size, j:j + square_size] = np.random.randint(100, 255)
                        b[i:i + square_size, j:j + square_size] = np.random.randint(100, 255)

    # RGB kanallarını birleştir
    image_array = np.stack((r, g, b), axis=-1)

    # NumPy dizisini PIL Image'e dönüştür
    img = Image.fromarray(image_array)

    # BytesIO nesnesine kaydet
    output = BytesIO()
    img.save(output, format="PNG")
    output.seek(0)

    return output

# --- LSB Image Functions ---
def encode_lsb_image(image_file, secret_data_bytes, output_filename):
    """Encodes secret_data_bytes into an image using LSB."""
    try:
        img = Image.open(image_file).convert("RGB")
        encoded = img.copy()
        width, height = img.size
        total_pixels = width * height
        max_bits = total_pixels * 3

        binary_secret = ''.join([format(byte, '08b') for byte in secret_data_bytes])
        binary_secret += TERMINATOR_BITS # Append terminator
        data_len = len(binary_secret)

        if data_len > max_bits:
            st.warning(f"Uyarı: Gizlenecek veri ({data_len} bit) resmin kapasitesini ({max_bits} bit) aşıyor. Tüm veri gizlenemeyebilir.")
            # Truncate data if too large? Or just warn? Current: warn only.
            # binary_secret = binary_secret[:max_bits] # Option to truncate
            # data_len = len(binary_secret)

        data_index = 0
        progress_bar = st.progress(0)
        pixels_processed = 0

        for y in range(height):
            for x in range(width):
                if data_index < data_len:
                    r, g, b = img.getpixel((x, y))
                    # Ensure pixel values are mutable (list)
                    pixel = [r, g, b]
                    for i in range(3): # R, G, B channels
                        if data_index < data_len:
                            # Modify LSB
                            pixel[i] = (pixel[i] & ~1) | int(binary_secret[data_index])
                            data_index += 1
                        else:
                            break
                    encoded.putpixel((x, y), tuple(pixel))
                else:
                    break # Break outer loops if data is fully embedded
            pixels_processed += width
            progress = data_index / data_len if data_len > 0 else 1.0
            progress_bar.progress(min(progress, 1.0)) # Ensure progress doesn't exceed 1.0
            if data_index >= data_len:
                break

        progress_bar.empty()

        if data_index < data_len:
             st.warning(f"Uyarı: Tüm veri resme sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")

        img_byte_arr = io.BytesIO()
        # Always save as PNG for lossless LSB storage
        output_format = 'PNG'
        if not output_filename.lower().endswith('.png'):
             output_filename += '.png'
        encoded.save(img_byte_arr, format=output_format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr, output_filename

    except Exception as e:
        st.error(f"Resim LSB kodlama hatası: {e}")
        return None, output_filename

def decode_lsb_image(image_file):
    """Decodes secret data from an image using LSB."""
    try:
        img = Image.open(image_file).convert("RGB")
        binary_data = ""
        terminator_len = len(TERMINATOR_BITS)
        found_terminator = False

        progress_bar = st.progress(0)
        pixels_processed = 0
        total_pixels = img.width * img.height

        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                for pixel_val in [r, g, b]:
                    binary_data += str(pixel_val & 1)
                    # Check for terminator efficiently
                    if len(binary_data) >= terminator_len and binary_data.endswith(TERMINATOR_BITS):
                        found_terminator = True
                        binary_data = binary_data[:-terminator_len] # Remove terminator
                        break
                if found_terminator: break
            pixels_processed += img.width
            if total_pixels > 0:
                 progress_bar.progress(min(pixels_processed / total_pixels, 1.0))
            if found_terminator: break

        progress_bar.empty()

        if not found_terminator:
            st.warning("Uyarı: Sonlandırıcı (terminator) bulunamadı. Dosyanın tamamı okundu, ancak veri eksik veya bozuk olabilir.")

        # Convert binary string to bytes
        all_bytes = bytearray()
        for i in range(0, len(binary_data), 8):
            byte_str = binary_data[i:i+8]
            if len(byte_str) == 8:
                try:
                    all_bytes.append(int(byte_str, 2))
                except ValueError:
                    # Handle potential conversion errors if non-binary chars ended up here
                    st.warning(f"Geçersiz bit dizisi bulundu: {byte_str}")
                    pass # Or handle more robustly
            # else: ignore incomplete byte at the end if terminator wasn't perfect

        return bytes(all_bytes) # Return raw bytes

    except Exception as e:
        st.error(f"Resim LSB çözümleme hatası: {e}")
        return None

# --- LSB Audio Functions ---
def encode_lsb_audio(audio_file, secret_data_bytes, output_filename_base):
    """Encodes secret_data_bytes into an audio file using LSB."""
    st.warning("Ses Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{audio_file.name}"
    temp_output_wav = "temp_steg_output.wav" # Intermediate WAV output
    final_output_filename = output_filename_base
    if not final_output_filename.lower().endswith('.wav'):
        final_output_filename += '.wav' # Output will be WAV

    output_bytes = None
    try:
        # 1. Save uploaded file temporarily
        with open(temp_input_path, "wb") as f:
            f.write(audio_file.getvalue())

        # 2. Convert input audio to PCM WAV for LSB processing
        temp_input_wav = "temp_input_converted.wav"
        convert_cmd = [
            'ffmpeg', '-i', temp_input_path,
            '-acodec', 'pcm_s16le', # Use signed 16-bit PCM Little Endian
            '-map_metadata', '-1', # Strip metadata
            temp_input_wav, '-y'
        ]
        st.info(f"Ses dönüştürülüyor: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Hata: Ses dönüştürme (ffmpeg) başarısız oldu. Çıktı:\n{result.stderr}")
            return None, final_output_filename
        if not os.path.exists(temp_input_wav):
            st.error(f"Hata: Dönüştürülmüş WAV dosyası '{temp_input_wav}' oluşturulamadı.")
            return None, final_output_filename

        # 3. Perform LSB encoding on the WAV file
        binary_secret = ''.join([format(byte, '08b') for byte in secret_data_bytes])
        binary_secret += TERMINATOR_BITS
        data_len = len(binary_secret)

        with wave.open(temp_input_wav, 'rb') as wf_in:
            params = wf_in.getparams()
            n_frames = wf_in.getnframes()
            sampwidth = wf_in.getsampwidth()
            n_channels = wf_in.getnchannels()
            audio_data = bytearray(wf_in.readframes(n_frames))

        max_bits = len(audio_data) # Each byte can store 1 bit
        if data_len > max_bits:
            st.warning(f"Uyarı: Gizlenecek veri ({data_len} bit) ses dosyasının kapasitesini ({max_bits} bit) aşıyor.")
            # Truncate or just warn? Current: Warn only.

        data_index = 0
        progress_bar = st.progress(0)
        for i in range(len(audio_data)):
            if data_index < data_len:
                # Modify LSB of the audio byte
                audio_data[i] = (audio_data[i] & 0xFE) | int(binary_secret[data_index])
                data_index += 1
            else:
                break
            if i % 1000 == 0: # Update progress bar periodically
                 progress = data_index / data_len if data_len > 0 else 1.0
                 progress_bar.progress(min(progress, 1.0))

        progress_bar.progress(1.0)
        progress_bar.empty()

        if data_index < data_len:
            st.warning(f"Uyarı: Tüm veri sese sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")

        # 4. Write the modified audio data to a new WAV file
        with wave.open(temp_output_wav, 'wb') as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(audio_data)

        # 5. Read the final bytes for download
        if os.path.exists(temp_output_wav):
            with open(temp_output_wav, "rb") as f:
                output_bytes = f.read()
            st.success(f"Veri başarıyla sese gizlendi: '{final_output_filename}'")
        else:
            st.error("Hata: Sonuç WAV dosyası oluşturulamadı.")

        return output_bytes, final_output_filename

    except wave.Error as e:
        st.error(f"WAV dosyası işlenirken hata oluştu: {e}")
        return None, final_output_filename
    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosya bulunamadı ({e}). ffmpeg kurulu mu?")
        return None, final_output_filename
    except Exception as e:
        st.error(f"Ses LSB kodlama sırasında beklenmedik hata: {e}")
        return None, final_output_filename
    finally:
        # Clean up temporary files
        for temp_file in [temp_input_path, temp_input_wav, temp_output_wav]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    st.warning(f"Geçici dosya silinemedi: {temp_file} ({e})")


def decode_lsb_audio(audio_file):
    """Decodes secret data from an audio file using LSB. Assumes WAV input."""
    audio_byte_arr = io.BytesIO(audio_file.getvalue())
    try:
        with wave.open(audio_byte_arr, 'rb') as wf:
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            audio_bytes = wf.readframes(n_frames)

        binary_data = ""
        terminator_len = len(TERMINATOR_BITS)
        found_terminator = False

        progress_bar = st.progress(0)
        total_bytes = len(audio_bytes)

        for i, byte in enumerate(audio_bytes):
            binary_data += str(byte & 1)
            if len(binary_data) >= terminator_len and binary_data.endswith(TERMINATOR_BITS):
                binary_data = binary_data[:-terminator_len]
                found_terminator = True
                break
            if i % 1000 == 0: # Update progress periodically
                progress_bar.progress(min(i / total_bytes, 1.0))

        progress_bar.progress(1.0)
        progress_bar.empty()

        if not found_terminator:
             st.warning("Uyarı: Sonlandırıcı (terminator) bulunamadı, tüm dosya okundu.")

        # Convert binary string to bytes
        all_bytes = bytearray()
        for i in range(0, len(binary_data), 8):
            byte_str = binary_data[i:i+8]
            if len(byte_str) == 8:
                try:
                    all_bytes.append(int(byte_str, 2))
                except ValueError:
                    st.warning(f"Geçersiz bit dizisi bulundu: {byte_str}")
                    pass

        return bytes(all_bytes)

    except wave.Error as e:
        st.error(f"WAV dosyası hatası: {e}. Lütfen geçerli bir WAV dosyası yükleyin.")
        return None
    except Exception as e:
        st.error(f"Ses LSB çözümleme hatası: {e}")
        return None

# --- LSB Video Functions ---
def encode_lsb_video(video_file, secret_data_bytes, output_filename_base):
    """Encodes secret_data_bytes into a video file using LSB."""
    st.warning("Video Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{video_file.name}"
    temp_output_video_only = "temp_steg_video_only.mkv" # Use MKV for lossless intermediate + flexibility
    temp_audio_extract = "temp_audio_extracted" # Base name for extracted audio
    temp_audio_extract_path = None # Will hold the actual path like temp_audio_extracted.aac
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # Use MKV as the default output container - more flexible than AVI
    final_output_path = f"{timestamp}_{output_filename_base}.mkv"
    output_video_bytes = None

    try:
        # 1. Save uploaded file temporarily
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getvalue())
        print(f"Geçici giriş dosyası oluşturuldu: '{temp_input_path}'")

        # 2. Check for audio stream and extract if present
        st.info("Giriş dosyasında ses akışı kontrol ediliyor...")
        audio_codec = None
        try:
            # Use ffprobe to get audio codec info
            ffprobe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
                temp_input_path
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                audio_codec = result.stdout.strip()
                st.info(f"Ses akışı bulundu (Codec: {audio_codec}). Çıkarılıyor...")
                # Determine appropriate extension based on codec if possible, default to .aac
                # This is a simplification; mapping codecs to extensions can be complex.
                audio_ext = ".aac" # Default extension
                if audio_codec in ['aac', 'mp3', 'opus', 'vorbis', 'flac', 'wav']: # Add more as needed
                   audio_ext = f".{audio_codec}" if audio_codec != 'wav' else '.wav' # ffmpeg might use .wav container for pcm

                temp_audio_extract_path = temp_audio_extract + audio_ext
                audio_extract_cmd = [
                    'ffmpeg', '-i', temp_input_path, '-vn', '-acodec', 'copy',
                    temp_audio_extract_path, '-y'
                ]
                print(f"Ses çıkarma komutu: {' '.join(audio_extract_cmd)}")
                extract_result = subprocess.run(audio_extract_cmd, capture_output=True, text=True)
                if extract_result.returncode != 0 or not os.path.exists(temp_audio_extract_path):
                    st.warning(f"Uyarı: Ses çıkarma başarısız oldu. Video sessiz olarak işlenecek. Hata:\n{extract_result.stderr}")
                    audio_codec = None # Mark as no audio
                    if os.path.exists(temp_audio_extract_path): os.remove(temp_audio_extract_path)
                    temp_audio_extract_path = None
                else:
                    print(f"Ses çıkarma tamamlandı: {temp_audio_extract_path}")
            else:
                st.info("Giriş dosyasında ses akışı bulunamadı veya alınamadı.")
                print(f"ffprobe çıktısı (ses kontrolü): {result.stdout}\nffprobe hatası: {result.stderr}")
        except FileNotFoundError:
             st.warning("ffprobe bulunamadı. Ses kontrolü yapılamadı. Video sessiz olarak işlenecek.")
             audio_codec = None
        except Exception as e:
             st.warning(f"Ses kontrolü sırasında hata: {e}. Video sessiz olarak işlenecek.")
             audio_codec = None


        # 3. Process video frames with LSB
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Hata: Giriş video dosyası '{temp_input_path}' açılamadı.")
            return None, final_output_path # Return early

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_in_video <= 0:
             st.warning("Video kare sayısı okunamadı, kapasite tahmini yapılamıyor.")
             max_bits_possible = float('inf') # Assume infinite for warning logic
        else:
             max_bits_possible = width * height * 3 * total_frames_in_video

        # Use FFV1 codec within MKV container for lossless intermediate video
        # FFV1 is generally well-supported and efficient lossless codec.
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(temp_output_video_only, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error(f"Hata: Çıkış video yazıcı ('{temp_output_video_only}') başlatılamadı. FFV1 codec destekleniyor mu?")
            cap.release()
            return None, final_output_path
        print(f"Geçici çıkış video dosyası için VideoWriter (FFV1/MKV) oluşturuldu: '{temp_output_video_only}'")

        binary_secret = ''.join([format(byte, '08b') for byte in secret_data_bytes])
        binary_secret += TERMINATOR_BITS
        data_len = len(binary_secret)

        if data_len > max_bits_possible:
            st.warning(f"Uyarı: Gömülecek veri boyutu ({data_len} bit) videonun tahmini kapasitesini ({max_bits_possible} bit) aşıyor.")

        data_index = 0
        embedded_fully = False
        progress_bar = st.progress(0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            frame_count += 1

            # Embed data if there's still data left to embed
            if data_index < data_len:
                frame_modified = False
                for y in range(height):
                    for x in range(width):
                        # frame[y, x] is B, G, R
                        pixel = frame[y, x]
                        for c in range(3): # B, G, R channels
                            if data_index < data_len:
                                pixel[c] = (pixel[c] & 0xFE) | int(binary_secret[data_index])
                                data_index += 1
                                frame_modified = True # Mark frame as modified
                            else:
                                embedded_fully = True
                                break # Break channel loop
                        if embedded_fully: break # Break x loop
                    if embedded_fully: break # Break y loop

            # Write the frame (original or modified)
            out.write(frame)

            # Update progress
            if total_frames_in_video > 0:
                progress = frame_count / total_frames_in_video
                progress_bar.progress(min(progress, 1.0))

            # If embedding finished, copy remaining frames directly (optional optimization)
            # Note: The current loop structure already handles this correctly by just writing unmodified frames.
            if embedded_fully and data_index >= data_len:
                 print(f"Veri {frame_count}. karede tamamen gömüldü.")
                 # Let the loop continue to write remaining frames without LSB processing

        progress_bar.empty()
        print(f"Video kare işleme tamamlandı. Toplam {frame_count} kare işlendi.")
        if data_index < data_len:
            st.warning(f"Uyarı: Tüm veri videoya sığmadı! Sadece {data_index}/{data_len} bit gömüldü.")

        # Release video resources
        cap.release()
        out.release()
        cv2.destroyAllWindows() # Good practice, though maybe not strictly necessary in streamlit
        print("OpenCV kaynakları serbest bırakıldı.")

        # 4. Mux video and audio (if audio exists) using ffmpeg
        if audio_codec and temp_audio_extract_path and os.path.exists(temp_audio_extract_path):
            st.info(f"LSB uygulanmış video ile orijinal ses ('{temp_audio_extract_path}') birleştiriliyor...")
            mux_cmd = [
                'ffmpeg',
                '-i', temp_output_video_only, # Input LSB video (FFV1 in MKV)
                '-i', temp_audio_extract_path, # Input extracted audio
                '-c:v', 'copy',       # Copy video stream (lossless)
                '-c:a', 'copy',       # Copy audio stream (original codec)
                '-map', '0:v:0',      # Map video from first input
                '-map', '1:a:0',      # Map audio from second input
                '-shortest',          # Finish encoding when the shortest input stream ends
                final_output_path,    # Output file (MKV)
                '-y'                  # Overwrite output if exists
            ]
            print(f"Birleştirme (muxing) komutu: {' '.join(mux_cmd)}")
            mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)

            if mux_result.returncode != 0 or not os.path.exists(final_output_path):
                st.error(f"Hata: Video ve ses birleştirme (muxing) başarısız oldu. FFmpeg çıktısı:\n{mux_result.stderr}")
                # Fallback: Offer the video-only file? Or just fail? Let's fail for now.
                if os.path.exists(final_output_path): os.remove(final_output_path) # Clean up failed output
                return None, final_output_path
            else:
                print(f"Birleştirme tamamlandı: '{final_output_path}'")
                st.success(f"Veri başarıyla videoya gizlendi ve orijinal ses eklendi: '{final_output_path}'")
                with open(final_output_path, "rb") as f:
                     output_video_bytes = f.read()
        else:
            # No audio or audio extraction failed, just use the video-only file
            st.warning("Ses akışı yok veya işlenemedi. Sadece LSB uygulanmış video döndürülüyor (MKV formatında).")
            if os.path.exists(temp_output_video_only):
                 # Rename the intermediate video file to be the final output file
                 try:
                      os.rename(temp_output_video_only, final_output_path)
                      print(f"Video dosyası yeniden adlandırıldı: '{final_output_path}'")
                      with open(final_output_path, "rb") as f:
                          output_video_bytes = f.read()
                 except OSError as e:
                      st.error(f"Video dosyası yeniden adlandırılamadı: {e}. LSB video '{temp_output_video_only}' olarak kaldı.")
                      # Try reading the original temp file as a last resort
                      with open(temp_output_video_only, "rb") as f:
                           output_video_bytes = f.read()
                           final_output_path = temp_output_video_only # Update filename if rename failed
            else:
                 st.error(f"Hata: LSB uygulanmış video dosyası ('{temp_output_video_only}') bulunamadı.")
                 return None, final_output_path

        return output_video_bytes, final_output_path

    except cv2.error as e:
        st.error(f"OpenCV hatası: {e}")
        return None, final_output_path
    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosya veya komut bulunamadı ({e}). ffmpeg/ffprobe kurulu ve PATH içinde mi?")
        return None, final_output_path
    except Exception as e:
        st.error(f"Video LSB kodlama sırasında beklenmedik hata: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback to console/log for debugging
        return None, final_output_path
    finally:
        st.info("Geçici dosyalar temizleniyor...")
        # Ensure cleanup happens even if functions returned early
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals() and out.isOpened(): out.release()
        cv2.destroyAllWindows()
        # Clean up temporary files
        files_to_clean = [temp_input_path, temp_output_video_only, temp_audio_extract_path, final_output_path]
        # Don't delete final_output_path if it's the intended result and bytes were read successfully
        if output_video_bytes is not None and os.path.exists(final_output_path):
             files_to_clean.remove(final_output_path) # Keep the final file if successful

        for temp_file in files_to_clean:
            if temp_file and os.path.exists(temp_file): # Check if path is not None
                try:
                    os.remove(temp_file)
                    print(f"'{temp_file}' temizlendi.")
                except OSError as e:
                    st.warning(f"Geçici dosya silinemedi: {temp_file} ({e})")


def decode_lsb_video(video_file):
    """Decodes secret data from a video file using LSB."""
    st.warning("Video Steganografi çözümleme işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_decode_{video_file.name}"
    decoded_bytes = None
    try:
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getvalue())

        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Hata: Video dosyası '{temp_input_path}' açılamadı.")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: st.info("Video kare sayısı okunamadı.")


        binary_data = ""
        terminator_len = len(TERMINATOR_BITS)
        found_terminator = False

        progress_bar = st.progress(0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Update progress less frequently for speed
            if frame_count % 10 == 0 and total_frames > 0:
                 progress_bar.progress(min(frame_count / total_frames, 1.0))

            for y in range(height):
                for x in range(width):
                    # frame[y, x] is B, G, R
                    pixel = frame[y, x]
                    for c in range(3): # B, G, R channels
                        binary_data += str(pixel[c] & 1)
                        if len(binary_data) >= terminator_len and binary_data.endswith(TERMINATOR_BITS):
                            found_terminator = True
                            binary_data = binary_data[:-terminator_len]
                            break # Break channel loop
                    if found_terminator: break # Break x loop
                if found_terminator: break # Break y loop
            if found_terminator: break # Break frame reading loop

        progress_bar.progress(1.0)
        progress_bar.empty()

        if not found_terminator:
            st.warning("Uyarı: Sonlandırıcı (terminator) bulunamadı!")

        # Convert binary string to bytes
        all_bytes = bytearray()
        for i in range(0, len(binary_data), 8):
            byte_str = binary_data[i:i+8]
            if len(byte_str) == 8:
                try:
                    all_bytes.append(int(byte_str, 2))
                except ValueError:
                    st.warning(f"Geçersiz bit dizisi bulundu: {byte_str}")
                    pass
        decoded_bytes = bytes(all_bytes)

    except cv2.error as e:
        st.error(f"OpenCV hatası: {e}")
        return None
    except Exception as e:
        st.error(f"Video LSB çözümleme hatası: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError as e:
                 st.warning(f"Geçici dosya silinemedi: {temp_input_path} ({e})")

    return decoded_bytes


# --- Encryption/Decryption Functions (AES-GCM + PBKDF2) ---

def encrypt_data(data_bytes, key_string, file_extension=None):
    """Encrypts data using AES-GCM with a key derived from key_string using PBKDF2."""
    try:
        # 1. Generate Salt
        salt = get_random_bytes(SALT_SIZE)

        # 2. Derive Key using PBKDF2
        key = PBKDF2(key_string.encode('utf-8'), salt, dkLen=KEY_SIZE, count=PBKDF2_ITERATIONS, hmac_hash_module=SHA256)

        # 3. Encrypt using AES-GCM
        cipher = AES.new(key, AES.MODE_GCM) # Nonce is generated automatically
        nonce = cipher.nonce # Get the generated nonce
        ciphertext, tag = cipher.encrypt_and_digest(data_bytes) # Encrypt and get auth tag

        # 4. Prepare JSON output
        # Encode binary data to Base64 strings for JSON compatibility
        encrypted_package = {
            'salt': base64.b64encode(salt).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'tag': base64.b64encode(tag).decode('utf-8')
        }
        if file_extension:
            encrypted_package['extension'] = file_extension

        return json.dumps(encrypted_package) # Return JSON string

    except Exception as e:
        st.error(f"Şifreleme hatası: {e}")
        return None

def decrypt_data(json_input_str, key_string):
    """Decrypts AES-GCM encrypted data using a key derived via PBKDF2."""
    try:
        # 1. Load JSON data
        try:
            encrypted_package = json.loads(json_input_str)
        except json.JSONDecodeError:
            st.error("Şifre çözme hatası: Geçersiz veri yapısı (JSON bekleniyordu).")
            return None, None

        # 2. Decode Base64 data
        salt = base64.b64decode(encrypted_package['salt'])
        nonce = base64.b64decode(encrypted_package['nonce'])
        ciphertext = base64.b64decode(encrypted_package['ciphertext'])
        tag = base64.b64decode(encrypted_package['tag'])
        retrieved_extension = encrypted_package.get('extension') # Optional

        # 3. Derive Key using PBKDF2 (must use the *same* salt and iterations)
        key = PBKDF2(key_string.encode('utf-8'), salt, dkLen=KEY_SIZE, count=PBKDF2_ITERATIONS, hmac_hash_module=SHA256)

        # 4. Decrypt and Verify using AES-GCM
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag) # Decrypts and checks integrity

        return decrypted_bytes, retrieved_extension

    except (ValueError, KeyError) as e:
        # ValueError can be raised by decrypt_and_verify if tag is invalid (wrong key or tampered data)
        st.error(f"Şifre çözme hatası: Şifre yanlış veya veri bozulmuş/değiştirilmiş. (Hata: {e})")
        return None, None
    except Exception as e:
        st.error(f"Beklenmedik bir şifre çözme hatası oluştu: {e}")
        return None, None


# --- Stable Diffusion ---
@st.cache_resource # Cache the pipeline object for performance
def load_sd_pipeline():
    """Loads the Stable Diffusion pipeline."""
    try:
        model_path = "runwayml/stable-diffusion-v1-5" # Or another model like CompVis/stable-diffusion-v1-4
        hf_token = os.getenv("HF_TOKEN") # Get token from environment variable
        if not hf_token:
             st.warning("Hugging Face Token (HF_TOKEN) ortam değişkeni bulunamadı. Özel modeller yüklenemeyebilir.")

        if not torch.cuda.is_available():
            st.warning("CUDA desteklenmiyor veya PyTorch CUDA için kurulmamış. CPU üzerinde çalışacak (yavaş).")
            device = "cpu"
        else:
            device = "cuda"
            st.info("CUDA bulundu, GPU kullanılacak.")
            # Check for accelerate library for potentially better performance/memory usage
            try:
                import accelerate
                st.info("Accelerate kütüphanesi bulundu.")
            except ImportError:
                 st.info("Opsiyonel 'accelerate' kütüphanesi bulunamadı (`pip install accelerate`).")


        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            token=hf_token # Use token=hf_token or use_auth_token=hf_token depending on diffusers version
        )
        pipe = pipe.to(device)
        return pipe
    except ImportError:
        st.error("Stable Diffusion için gerekli kütüphaneler (diffusers, transformers, torch) yüklenemedi.")
        return None
    except Exception as e:
        st.error(f"Stable Diffusion modeli yüklenirken hata oluştu: {e}")
        return None

def generate_image_from_prompt(pipe, prompt):
    """Generates an image from a text prompt using the loaded Stable Diffusion pipeline."""
    if pipe is None:
        st.error("Stable Diffusion modeli yüklenemedi, görsel üretilemiyor.")
        return None
    try:
        with st.spinner(f"'{prompt}' için görsel üretiliyor... Bu işlem biraz zaman alabilir."):
            # Optional: Add negative prompts, control steps, guidance scale etc.
            image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Görsel üretimi sırasında bir hata oluştu: {e}")
        return None

# --- Streamlit UI ---
st.title("🔒 Steganografi Uygulaması (LSB + AES-GCM)")

# Sidebar
operation = st.sidebar.radio("Yapmak istediğiniz işlemi seçin:", ("Gizle (Encode)", "Çöz (Decode)"))
st.sidebar.markdown("---")
media_type = st.sidebar.selectbox("Medya türünü seçin:", ("Resim (Image)", "Ses (Audio)", "Video (Video)"))
st.sidebar.markdown("---")
password = st.sidebar.text_input("Şifreyi girin (Gizleme ve Çözme için Gerekli):", type="password")
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Limitler:**
- Medya Dosyası: {MAX_FILE_SIZE_MB_MEDIA} MB
- Gizlenecek Dosya: {MAX_FILE_SIZE_MB_SECRET} MB
""")
st.sidebar.markdown("---")
st.sidebar.info("Not: Ses ve Video işlemleri için `ffmpeg` ve `ffprobe` sisteminizde kurulu olmalıdır.")


# Main Area based on Operation
if operation == "Gizle (Encode)":
    st.header("Veri Gizleme (Encode)")

    # --- Secret Data Input ---
    st.subheader("1. Gizlenecek Veri")
    secret_choice = st.radio("Ne gizlemek istiyorsunuz?", ("Metin", "Dosya"), horizontal=True)

    secret_data_to_embed = None
    original_filename = None # Store original filename (for file option)
    file_extension = None # Store extension (for file option)

    if secret_choice == "Metin":
        secret_text_input = st.text_area("Gizlenecek metni girin:")
        if secret_text_input:
            secret_data_to_embed = secret_text_input.encode('utf-8') # Encode text to bytes
            original_filename = "gizli_metin.txt" # Assign a default name
            file_extension = ".txt"
    else:
        secret_file = st.file_uploader(f"Gizlenecek dosyayı yükleyin (Maksimum {MAX_FILE_SIZE_MB_SECRET} MB):")
        if secret_file is not None:
            file_size_secret = len(secret_file.getvalue())
            if file_size_secret > MAX_FILE_SIZE_MB_SECRET * 1024 * 1024:
                 st.error(f"Hata: Gizlenecek dosya boyutu ({file_size_secret / (1024*1024):.2f} MB) limiti ({MAX_FILE_SIZE_MB_SECRET} MB) aşıyor.")
            else:
                 original_filename = secret_file.name
                 _, file_extension = os.path.splitext(original_filename)
                 secret_data_to_embed = secret_file.getvalue() # Get data as bytes

    if secret_data_to_embed:
        st.success(f"Gizlenecek veri hazırlandı ({len(secret_data_to_embed)} bytes).")
    else:
        st.info("Lütfen gizlenecek metni girin veya bir dosya yükleyin.")


    # --- Cover Media Input ---
    st.subheader("2. Taşıyıcı Medya")
    uploaded_media_file = None
    generated_image_bytesio = None

    # Image specific options
    if media_type == "Resim (Image)":
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
             # File uploader for image
             allowed_types_img = ["png", "bmp", "tiff", "jpg", "jpeg"]
             st.info(f"Not: Yüklenen resimler ({', '.join(allowed_types_img)}) LSB için PNG formatına dönüştürülecektir.")
             uploaded_media_file = st.file_uploader(
                 f"Gizleme yapılacak RESİM dosyasını yükleyin (Maksimum {MAX_FILE_SIZE_MB_MEDIA} MB):",
                 type=allowed_types_img
             )

    # Audio specific uploader
    elif media_type == "Ses (Audio)":
         allowed_types_audio = ["wav", "mp3", "flac", "ogg", "aac", "m4a"] # Common types ffmpeg usually handles
         st.info(f"Not: Yüklenen ses dosyaları ({', '.join(allowed_types_audio)}) LSB için WAV formatına dönüştürülecektir.")
         uploaded_media_file = st.file_uploader(
             f"Gizleme yapılacak SES dosyasını yükleyin (Maksimum {MAX_FILE_SIZE_MB_MEDIA} MB):",
             type=allowed_types_audio
         )

    # Video specific uploader
    elif media_type == "Video (Video)":
         allowed_types_video = ["mp4", "avi", "mkv", "mov", "wmv", "flv"] # Common types ffmpeg usually handles
         st.info(f"Not: Video kareleri kayıpsız işlenecek, ses (varsa) korunacaktır. Çıktı MKV formatında olacaktır.")
         uploaded_media_file = st.file_uploader(
             f"Gizleme yapılacak VİDEO dosyasını yükleyin (Maksimum {MAX_FILE_SIZE_MB_MEDIA} MB):",
             type=allowed_types_video
         )

    if uploaded_media_file and media_source != "AI ile oluştur":
         # Check media file size
         uploaded_media_file.seek(0, io.SEEK_END) # Go to end to get size
         file_size_media = uploaded_media_file.tell()
         uploaded_media_file.seek(0) # Go back to start
         if file_size_media > MAX_FILE_SIZE_MB_MEDIA * 1024 * 1024:
             st.error(f"Hata: Medya dosyası boyutu ({file_size_media / (1024*1024):.2f} MB) limiti ({MAX_FILE_SIZE_MB_MEDIA} MB) aşıyor.")
             uploaded_media_file = None # Invalidate file
         else:
              st.success(f"Taşıyıcı medya '{uploaded_media_file.name}' yüklendi ({file_size_media / (1024*1024):.2f} MB).")


    # --- Execute Encoding ---
    st.subheader("3. Gizleme İşlemi")
    if st.button("Gizle ve Şifrele"):
        if uploaded_media_file is not None and secret_data_to_embed is not None and password:
            with st.spinner("Veri şifreleniyor ve gizleniyor..."):
                try:
                    # 1. Encrypt the secret data (using AES-GCM + PBKDF2)
                    encrypted_json_str = encrypt_data(secret_data_to_embed, password, file_extension)

                    if not encrypted_json_str:
                        st.error("Şifreleme başarısız oldu.")
                        # Use st.stop() in newer streamlit versions if needed
                        raise Exception("Encryption failed") # Stop processing


                    encrypted_data_bytes = encrypted_json_str.encode('utf-8') # Encrypt returns JSON string, encode to bytes for LSB

                    # 2. Get base name for output file
                    base_name, _ = os.path.splitext(uploaded_media_file.name)
                    output_filename_base = f"gizlenmis_{base_name}"
                    output_bytes = None
                    final_output_filename = "" # Will be set by the encode function


                    # 3. Perform LSB encoding based on media type
                    if media_type == "Resim (Image)":
                        output_bytes, final_output_filename = encode_lsb_image(uploaded_media_file, encrypted_data_bytes, output_filename_base)
                        mime_type = "image/png"
                    elif media_type == "Ses (Audio)":
                        output_bytes, final_output_filename = encode_lsb_audio(uploaded_media_file, encrypted_data_bytes, output_filename_base)
                        mime_type = "audio/wav"
                    elif media_type == "Video (Video)":
                        output_bytes, final_output_filename = encode_lsb_video(uploaded_media_file, encrypted_data_bytes, output_filename_base)
                        mime_type = "video/x-matroska" # MKV mime type


                    # 4. Provide download link if successful
                    if output_bytes and final_output_filename:
                        st.success(f"Veri başarıyla gizlendi!")
                        st.download_button(
                            label=f"Gizlenmiş Dosyayı İndir ({os.path.basename(final_output_filename)})",
                            data=output_bytes,
                            file_name=os.path.basename(final_output_filename), # Use only the filename part
                            mime=mime_type
                        )
                    else:
                        st.error("Veri gizleme işlemi başarısız oldu veya sonuç dosyası oluşturulamadı.")

                except Exception as e:
                    st.error(f"Gizleme sırasında bir hata oluştu: {e}")
                    import traceback
                    st.error(f"Detay: {traceback.format_exc()}") # Show more details on error

        elif not uploaded_media_file:
            st.warning("Lütfen bir taşıyıcı medya dosyası yükleyin.")
        elif not secret_data_to_embed:
             st.warning("Lütfen gizlenecek metni girin veya dosya yükleyin.")

# --- Decode Operation ---
elif operation == "Çöz (Decode)":
    st.header("Veri Çözme (Decode)")

    st.subheader("1. Gizlenmiş Medya Dosyası")
    # Define expected file types based on the encoding output format
    if media_type == "Resim (Image)":
        expected_types = ["png"] # We always save as PNG
        st.info("LSB ile gizlenmiş PNG dosyası yükleyin.")
    elif media_type == "Ses (Audio)":
        expected_types = ["wav"] # We always save as WAV
        st.info("LSB ile gizlenmiş WAV dosyası yükleyin.")
    elif media_type == "Video (Video)":
        expected_types = ["mkv", "avi"] # We output MKV now, accept legacy AVI just in case
        st.info("LSB ile gizlenmiş MKV (veya eski AVI) dosyası yükleyin.")
    steg_media_file = st.file_uploader(
        f"Çözme yapılacak gizlenmiş {media_type.split(' ')[0].lower()} dosyasını yükleyin:",
        type=expected_types
    )
    st.subheader("2. Çözme İşlemi")
    if st.button("Çöz ve Şifreyi Aç"):
        if steg_media_file is not None and password:
            with st.spinner("Veri çözümleniyor ve şifre açılıyor..."):
                try:
                    extracted_lsb_bytes = None
                    if media_type == "Resim (Image)":
                         extracted_lsb_bytes = decode_lsb_image(steg_media_file)
                    elif media_type == "Ses (Audio)":
                         extracted_lsb_bytes = decode_lsb_audio(steg_media_file)
                    elif media_type == "Video (Video)":
                         extracted_lsb_bytes = decode_lsb_video(steg_media_file)
                    if not extracted_lsb_bytes:
                        st.error("Gizlenmiş dosyadan LSB veri çıkarılamadı. Dosya türü doğru mu veya dosya bozuk mu?")
                        raise Exception("LSB Extraction failed")
                    try:
                        extracted_json_str = extracted_lsb_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                         st.error("Çıkarılan LSB verisi geçerli metin (JSON) formatında değil. Dosya bozuk veya yanlış türde olabilir.")
                         raise Exception("LSB data is not valid UTF-8 JSON")
                    decrypted_payload_bytes, retrieved_ext = decrypt_data(extracted_json_str, password)
                    if decrypted_payload_bytes is not None:
                        st.success("Veri başarıyla çözüldü ve şifresi açıldı!")
                        try:
                            decoded_text = decrypted_payload_bytes.decode('utf-8')
                            st.subheader("Çözülen Metin:")
                            st.text_area("Metin", decoded_text, height=200)
                        except UnicodeDecodeError:
                            st.subheader("Çözülen Dosya:")
                            if retrieved_ext:
                                download_filename = f"cozulen_dosya{retrieved_ext}"
                            else:
                                download_filename = "cozulen_dosya.bin"
                                st.warning("Orijinal dosya uzantısı alınamadı, '.bin' olarak kaydedilecek.")
                            st.download_button(
                                label=f"Çözülen Dosyayı İndir ({download_filename})",
                                data=decrypted_payload_bytes,
                                file_name=download_filename,
                                mime="application/octet-stream"
                            )
                except Exception as e:
                    st.error(f"Çözme sırasında bir hata oluştu: {e}")
        elif not steg_media_file:
            st.warning("Lütfen çözülecek gizlenmiş dosyayı yükleyin.")
