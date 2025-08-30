# app.py
import streamlit as st
st.set_page_config(page_title="Steganografi Uygulaması", page_icon="🔒")

# crypto / utils
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
import requests
from urllib.parse import quote_plus
import mimetypes

# -------------------------
# Helper: AI image generation
# -------------------------
def generate_ai_image(prompt, width=256, height=256):
    encoded_prompt = quote_plus(prompt)
    random_seed = random.randint(0, 999999999999)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={random_seed}&model=turbo&nologo=true&transparent=true" # 
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img_bytes = BytesIO(response.content)
    img_bytes.seek(0)
    return img_bytes

# -------------------------
# Image LSB methods: Simple / LSB-matching / Adaptive
# -------------------------
def encode_lsb_simple(image_file, secret_data_bytes):
    """Standard sequential LSB embed. Returns PNG bytes."""
    img = Image.open(image_file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()
    capacity = flat.size  # bits (1 LSB per byte)
    payload_bits = np.unpackbits(np.frombuffer(secret_data_bytes, dtype=np.uint8))
    payload_bitlen = payload_bits.size
    len_bits = np.unpackbits(np.frombuffer(payload_bitlen.to_bytes(4, 'big'), dtype=np.uint8))
    all_bits = np.concatenate([len_bits, payload_bits]).astype(np.uint8)
    if all_bits.size > capacity:
        raise ValueError(f"Payload too large: need {all_bits.size} bits but capacity {capacity} bits.")
    flat[: all_bits.size] = (flat[: all_bits.size] & ~1) | all_bits
    stego = flat.reshape(arr.shape)
    out = Image.fromarray(stego)
    bio = BytesIO(); out.save(bio, format="PNG"); return bio.getvalue()

def encode_lsb_lsb_matching(image_file, secret_data_bytes):
    """
    LSB-matching: if LSB != desired then +1 or -1 applied.
    Returns PNG bytes.
    """
    img = Image.open(image_file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()
    capacity = flat.size
    payload_bits = np.unpackbits(np.frombuffer(secret_data_bytes, dtype=np.uint8))
    payload_bitlen = payload_bits.size
    len_bits = np.unpackbits(np.frombuffer(payload_bitlen.to_bytes(4, 'big'), dtype=np.uint8))
    all_bits = np.concatenate([len_bits, payload_bits]).astype(np.uint8)
    if all_bits.size > capacity:
        raise ValueError(f"Payload too large: need {all_bits.size} bits but capacity {capacity} bits.")
    for i, bit in enumerate(all_bits):
        cur = int(flat[i])
        if (cur & 1) != int(bit):
            if cur == 255:
                flat[i] = cur - 1
            elif cur == 0:
                flat[i] = cur + 1
            else:
                if (i % 2) == 0:
                    flat[i] = cur + 1
                else:
                    flat[i] = cur - 1
    stego = flat.reshape(arr.shape)
    out = Image.fromarray(stego)
    bio = BytesIO(); out.save(bio, format="PNG"); return bio.getvalue()

def encode_lsb_adaptive(image_file, secret_data_bytes, canny_low=100, canny_high=200):
    """
    Adaptive: write first 32 bits (length) into flat[:32], remaining bits prioritized into edge pixel channels.
    Returns PNG bytes.
    """
    img = Image.open(image_file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr.flatten()
    capacity = flat.size

    payload_bits = np.unpackbits(np.frombuffer(secret_data_bytes, dtype=np.uint8))
    payload_bitlen = payload_bits.size
    len_bits = np.unpackbits(np.frombuffer(payload_bitlen.to_bytes(4, 'big'), dtype=np.uint8))
    all_payload_bits = np.concatenate([len_bits, payload_bits]).astype(np.uint8)
    if all_payload_bits.size > capacity:
        raise ValueError(f"Payload too large: need {all_payload_bits.size} bits but capacity {capacity} bits.")

    # write first 32 bits into fixed header
    flat[:32] = (flat[:32] & ~1) | all_payload_bits[:32]

    # prepare positions for remaining bits (exclude first 32 indices)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    ys, xs = np.where(edges > 0)
    # base indices for edge pixels (RGB -> 3 positions)
    edge_base = (ys * w + xs) * 3
    edge_positions = np.concatenate([edge_base + offset for offset in (0,1,2)]) if edge_base.size>0 else np.array([], dtype=np.int64)

    # all base indices
    y_all = np.repeat(np.arange(h), w)
    x_all = np.tile(np.arange(w), h)
    all_base = (y_all * w + x_all) * 3
    if edge_base.size > 0:
        non_base = np.setdiff1d(all_base, edge_base, assume_unique=True)
    else:
        non_base = all_base
    non_positions = np.concatenate([non_base + offset for offset in (0,1,2)]) if non_base.size>0 else np.array([], dtype=np.int64)

    combined = np.concatenate([edge_positions, non_positions]).astype(np.int64)
    combined = combined[combined >= 32]  # do not override header

    needed = all_payload_bits.size - 32
    if needed > combined.size:
        raise ValueError("Adaptive positions insufficient; capacity mismatch.")

    target_positions = combined[:needed]
    flat[target_positions] = (flat[target_positions] & ~1) | all_payload_bits[32:32+needed]

    stego = flat.reshape(arr.shape)
    out = Image.fromarray(stego)
    bio = BytesIO(); out.save(bio, format="PNG"); return bio.getvalue()

def decode_lsb_simple_or_matching(image_file):
    """Extraction common for simple and LSB-matching (sequential LSBs). Returns payload bytes."""
    img = Image.open(image_file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten().astype(np.uint8)
    total_bits = flat.size
    if total_bits < 32:
        raise ValueError("Carrier too small.")
    len_bits = flat[:32] & 1
    len_bytes = np.packbits(len_bits).tobytes()
    payload_bitlen = int.from_bytes(len_bytes, byteorder='big')
    if payload_bitlen == 0:
        return b""
    if 32 + payload_bitlen > total_bits:
        payload_bitlen = max(0, total_bits - 32)
    data_bits = flat[32:32+payload_bitlen] & 1
    rem = data_bits.size % 8
    if rem != 0:
        data_bits = data_bits[:-rem]
    packed = np.packbits(data_bits)
    return packed.tobytes()

def decode_lsb_adaptive(image_file, canny_low=100, canny_high=200):
    """Extraction for adaptive embedding. Returns payload bytes."""
    img = Image.open(image_file).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr.flatten().astype(np.uint8)
    total_bits = flat.size
    if total_bits < 32:
        raise ValueError("Carrier too small.")
    len_bits = flat[:32] & 1
    len_bytes = np.packbits(len_bits).tobytes()
    payload_bitlen = int.from_bytes(len_bytes, byteorder='big')
    if payload_bitlen == 0:
        return b""

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    ys, xs = np.where(edges > 0)
    edge_base = (ys * w + xs) * 3
    edge_positions = np.concatenate([edge_base + offset for offset in (0,1,2)]) if edge_base.size>0 else np.array([], dtype=np.int64)
    y_all = np.repeat(np.arange(h), w)
    x_all = np.tile(np.arange(w), h)
    all_base = (y_all * w + x_all) * 3
    if edge_base.size > 0:
        non_base = np.setdiff1d(all_base, edge_base, assume_unique=True)
    else:
        non_base = all_base
    non_positions = np.concatenate([non_base + offset for offset in (0,1,2)]) if non_base.size>0 else np.array([], dtype=np.int64)
    combined = np.concatenate([edge_positions, non_positions]).astype(np.int64)
    combined = combined[combined >= 32]
    needed = payload_bitlen
    remaining_bits_count = needed
    data_bits = []
    if remaining_bits_count > 0:
        read_count = min(remaining_bits_count, combined.size)
        pos = combined[:read_count]
        if pos.size > 0:
            bits_arr = flat[pos] & 1
            data_bits = bits_arr.tolist()
        else:
            data_bits = []
    data_bits = np.array(data_bits, dtype=np.uint8)
    rem = data_bits.size % 8
    if rem != 0:
        data_bits = data_bits[:-rem]
    if data_bits.size == 0:
        return b""
    packed = np.packbits(data_bits)
    return packed.tobytes()

def encode_lsb(image_file, secret_data, output_filename, method="Standard"):
    """
    Dispatcher: method in ("Standard", "LSB-Matching", "Adaptive")
    Accepts secret_data as str or bytes. Returns PNG bytes or raises.
    """
    if isinstance(secret_data, str):
        secret_bytes = secret_data.encode('utf-8')
    elif isinstance(secret_data, bytes):
        secret_bytes = secret_data
    else:
        secret_bytes = str(secret_data).encode('utf-8')

    method = str(method).lower()
    if method in ("standard", "lsb", "simple"):
        return encode_lsb_simple(image_file, secret_bytes)
    elif method in ("lsb-matching", "matching", "lsb_matching"):
        return encode_lsb_lsb_matching(image_file, secret_bytes)
    elif method in ("adaptive", "adaptive-lsb"):
        return encode_lsb_adaptive(image_file, secret_bytes)
    else:
        raise ValueError(f"Unknown embedding method: {method}")

def decode_lsb(image_file, method="Standard"):
    method = str(method).lower()
    if method in ("standard", "lsb", "simple", "lsb-matching", "matching", "lsb_matching"):
        return decode_lsb_simple_or_matching(image_file)
    elif method in ("adaptive", "adaptive-lsb"):
        return decode_lsb_adaptive(image_file)
    else:
        raise ValueError(f"Unknown decoding method: {method}")

# -------------------------
# Audio / Video LSB functions (kept compatible with existing flow)
# -------------------------
def encode_lsb_audio(audio_file, secret_data, output_filename):
    st.warning("Ses Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır.")
    temp_input_path = f"temp_input_{datetime.datetime.now().timestamp()}_{getattr(audio_file, 'name', 'audio')}"
    temp_output_path_converted = f"temp_steg_converted_{datetime.datetime.now().timestamp()}.wav"
    temp_final_output_path = f"temp_final_output_{datetime.datetime.now().timestamp()}.wav"
    output_bytes = None

    try:
        cmd_convert = [
            "ffmpeg", "-i", "pipe:0",
            "-f", "wav", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd_convert, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wav_bytes, err = proc.communicate(input=audio_file.read())
        if proc.returncode != 0:
            st.error(f"Hata: Ses dönüştürme başarısız oldu: {err.decode()}")
            return None

        secret_str = str(secret_data)
        bits = "".join(format(ord(c), "08b") for c in secret_str) + "00000000" * 5
        total_bits, idx = len(bits), 0

        header_end = wav_bytes.find(b"data")
        if header_end == -1:
            st.error("WAV içinde 'data' chunk bulunamadı.")
            return None
        header_end += 8

        ba = bytearray(wav_bytes)
        for i in range(header_end, len(ba)):
            if idx >= total_bits:
                break
            ba[i] = (ba[i] & 0xFE) | int(bits[idx])
            idx += 1
        if idx < total_bits:
            st.warning(f"Uyarı: Sadece {idx}/{total_bits} bit gömülebildi.")

        cmd_out = ["ffmpeg", "-f", "wav", "-i", "pipe:0", "-f", "wav", "pipe:1"]
        proc2 = subprocess.Popen(cmd_out, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        final_bytes, err2 = proc2.communicate(input=bytes(ba))
        if proc2.returncode != 0:
            st.error(f"Hata: Son işlem başarısız oldu: {err2.decode()}")
            return None

        return final_bytes

    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosya veya komut (ffmpeg?) bulunamadı: {e}")
        return None
    except wave.Error as e:
        st.error(f"WAV dosyası işlenirken hata oluştu: {e}")
        return None
    except Exception as e:
        st.error(f"Ses kodlama sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    finally:
        for temp_file in [temp_input_path, temp_output_path_converted, temp_final_output_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

def decode_lsb_audio(audio_file):
    audio_byte_arr = io.BytesIO(audio_file.getvalue())
    try:
        with wave.open(audio_byte_arr, 'rb') as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
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
            if i % 10000 == 0:
                progress = min((i+1) / total_bytes, 1.0) if total_bytes > 0 else 1.0
                try:
                    progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_bytes} bayt)")
                except st.errors.StreamlitAPIException:
                    pass

        if 'progress_bar' in locals(): progress_bar.empty()

        if not found_terminator:
            st.warning("Uyarı: Terminator bulunamadı. Tüm dosya okundu, ancak gizli veri tamamlanmamış olabilir veya dosya LSB ile değiştirilmemiş olabilir.")

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
                    pass
        return decoded_data

    except wave.Error as e:
        st.error(f"WAV dosyası okunurken hata oluştu: {e}")
        return None
    except Exception as e:
        st.error(f"Ses çözme sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# -------------------------
# Video encode/decode (kept)
# -------------------------
def encode_lsb_video(video_file, secret_data, output_filename):
    st.warning("Video Steganografi işlemi disk üzerinde geçici dosyalar oluşturacaktır. Bu işlem uzun sürebilir.")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    temp_input = f"temp_in_{timestamp}_{getattr(video_file, 'name', 'video')}"
    try:
        with open(temp_input, "wb") as f:
            f.write(video_file.getvalue())

        temp_audio = f"temp_audio_{timestamp}.aac"
        cmd_audio = [
            "ffmpeg", "-i", temp_input,
            "-vn", "-acodec", "copy",
            "-y", temp_audio
        ]
        proc_a = subprocess.run(cmd_audio, capture_output=True, text=True)
        if proc_a.returncode != 0 or not os.path.exists(temp_audio):
            temp_audio = None

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            st.error("Video açılamadı.")
            return None, None
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

        temp_vid = f"temp_vid_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        out = cv2.VideoWriter(temp_vid, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            st.error("Geçici video dosyası oluşturulamadı.")
            return None, None

        bits = "".join(format(ord(c), "08b") for c in str(secret_data)) + "00000000"*5
        idx, total = 0, len(bits)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flat = frame.reshape(-1)
            for i in range(flat.size):
                if idx >= total:
                    break
                flat[i] = (flat[i] & 0xFE) | int(bits[idx])
                idx += 1
            frame = flat.reshape((height, width, 3))
            out.write(frame)
        cap.release()
        out.release()

        if idx < total:
            st.warning(f"Sadece {idx}/{total} bit gömülebildi.")

        if temp_audio:
            cmd_mux = [
                "ffmpeg",
                "-i", temp_vid,
                "-i", temp_audio,
                "-c:v", "copy",
                "-c:a", "copy",
                "-y",
                output_filename
            ]
        else:
            cmd_mux = ["ffmpeg", "-i", temp_vid, "-c:v", "copy", "-y", output_filename]

        proc_m = subprocess.run(cmd_mux, capture_output=True, text=True)
        if proc_m.returncode != 0 or not os.path.exists(output_filename):
            st.error("Muxing başarısız oldu.")
            return None, None

        with open(output_filename, "rb") as f:
            data = f.read()

        for p in (temp_input, temp_vid, temp_audio or ""):
            try:
                os.remove(p)
            except:
                pass

        return data, output_filename
    except Exception as e:
        st.error(f"Video kodlama sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals() and out.isOpened(): out.release()

def extract_lsb_from_frame(frame, binary_data_list, terminator_bits):
    height, width, _ = frame.shape
    for y in range(height):
        for x in range(width):
            for c in range(3):
                binary_data_list.append(str(frame[y, x, c] & 1))
                if len(binary_data_list) >= len(terminator_bits):
                    current_suffix = "".join(binary_data_list[-len(terminator_bits):])
                    if current_suffix == terminator_bits:
                        del binary_data_list[-len(terminator_bits):]
                        return True
    return False

def decode_lsb_video(video_file):
    st.warning("Video Steganografi çözümleme işlemi disk üzerinde geçici dosyalar oluşturabilir.")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    temp_input_path = f"temp_input_decode_{timestamp}_{getattr(video_file, 'name', 'video')}"
    try:
        with open(temp_input_path, "wb") as f:
            f.write(video_file.getvalue())

        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            st.error(f"Hata: Giriş video dosyası açılamadı.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        binary_data_list = []
        terminator_bits = '00000000' * 5
        found_terminator = False
        progress_text = "Video kareleri çözümleniyor..."
        progress_bar = st.progress(0.0, text=progress_text)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                try:
                    progress_bar.progress(progress, text=f"{progress_text} (Kare {frame_count}/{total_frames})")
                except st.errors.StreamlitAPIException:
                    pass
            found_terminator_in_frame = extract_lsb_from_frame(frame, binary_data_list, terminator_bits)
            if found_terminator_in_frame:
                found_terminator = True
                break

        if 'progress_bar' in locals(): progress_bar.empty()
        cap.release()

        if not found_terminator:
            st.warning("Uyarı: Terminator bulunamadı. Tüm video okundu, ancak gizli veri tamamlanmamış olabilir veya dosya LSB ile değiştirilmemiş olabilir.")

        binary_data = "".join(binary_data_list)
        remainder = len(binary_data) % 8
        if remainder != 0:
            st.warning(f"Uyarı: Çıkarılan bit sayısı ({len(binary_data)}) 8'in katı değil. Son {remainder} bit atlanıyor.")
            binary_data = binary_data[:-remainder]

        all_bytes_str = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
        decoded_json_str = ""
        try:
            byte_list = [int(byte_s, 2) for byte_s in all_bytes_str]
            decoded_json_str = bytearray(byte_list).decode('utf-8')
            if not (decoded_json_str.startswith('{') and decoded_json_str.endswith('}')):
                st.warning("Çıkarılan veri UTF-8 metin gibi görünüyor ancak geçerli JSON yapısı beklenmiyor.")
            return decoded_json_str
        except Exception:
            st.error("Çıkarılan veriler UTF-8 olarak çözülemedi veya bozuk.")
            return None
    except Exception as e:
        st.error(f"Video çözme sırasında beklenmedik bir hata oluştu: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    finally:
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except:
                pass

# -------------------------
# Encryption: AES-CBC (existing)
# -------------------------
def encrypt_data(data_bytes, key_string, original_filename=None):
    if not isinstance(data_bytes, bytes):
        if isinstance(data_bytes, str):
            data_bytes = data_bytes.encode('utf-8')
        else:
            st.error("Şifreleme hatası: Girdi 'bytes' türünde olmalı.")
            return None
    try:
        key = hashlib.sha256(key_string.encode('utf-8')).digest()
        cipher = AES.new(key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data_bytes, AES.block_size))
        iv = base64.b64encode(cipher.iv).decode('utf-8')
        ct = base64.b64encode(ct_bytes).decode('utf-8')
        result = {'iv': iv, 'ciphertext': ct}
        if original_filename:
            result['filename'] = os.path.basename(original_filename)
        return json.dumps(result)
    except Exception as e:
        st.error(f"Şifreleme sırasında hata: {e}")
        return None

def decrypt_data(json_input_str, key_string):
    try:
        key = hashlib.sha256(key_string.encode('utf-8')).digest()
        b64 = json.loads(json_input_str)
        iv = base64.b64decode(b64['iv'])
        ct = base64.b64decode(b64['ciphertext'])
        retrieved_filename = b64.get('filename')
        cipher = AES.new(key, AES.MODE_CBC, iv)
        pt_bytes = unpad(cipher.decrypt(ct), AES.block_size)
        return pt_bytes, retrieved_filename
    except (ValueError, KeyError) as e:
        st.error(f"Şifre çözme hatası: Veri bozuk veya şifre yanlış olabilir. Hata: {e}")
        return None, None
    except json.JSONDecodeError as e:
        st.error(f"Şifre çözme hatası: Girdi geçerli bir JSON değil. Hata: {e}")
        return None, None
    except Exception as e:
        st.error(f"Beklenmedik bir şifre çözme hatası oluştu: {e}")
        return None, None

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔒 Steganografi Uygulaması")
st.markdown("Verilerinizi resim, ses veya video dosyaları içine gizleyin ve şifreleyin.")
st.markdown("---")

operation = st.sidebar.radio("Yapmak istediğiniz işlemi seçin:", ("Gizle (Encode)", "Çöz (Decode)"))
media_type = st.sidebar.selectbox("Medya türünü seçin:", ("Resim (Image)", "Ses (Audio)", "Video (Video)"))
password = st.sidebar.text_input("Şifreyi girin (Gizleme ve Çözme için gerekli):", type="password")

# --- Encode ---
if operation == "Gizle (Encode)":
    st.header(f" Veri Gizleme ({media_type})")
    MAX_CARRIER_SIZE_MB = 50
    MAX_SECRET_SIZE_MB = 20
    MAX_CARRIER_SIZE_BYTES = MAX_CARRIER_SIZE_MB * 1024 * 1024
    MAX_SECRET_SIZE_BYTES = MAX_SECRET_SIZE_MB * 1024 * 1024

    st.subheader("1. Gizlenecek Veri")
    secret_choice = st.radio("Ne gizlemek istiyorsunuz?", ("Metin", "Dosya"), key="secret_choice")
    secret_data_to_embed_bytes = None
    original_secret_filename = None

    if secret_choice == "Metin":
        secret_data_input = st.text_area("Gizlenecek metni girin:", key="secret_text", max_chars=99999)
        if secret_data_input:
            try:
                secret_data_to_embed_bytes = secret_data_input.encode('utf-8')
                original_secret_filename = "gizli_metin.txt"
                if len(secret_data_to_embed_bytes) > MAX_SECRET_SIZE_BYTES:
                    st.error(f"Metin verisi çok büyük ({len(secret_data_to_embed_bytes)/(1024*1024):.2f} MB).")
                    secret_data_to_embed_bytes = None
            except Exception as e:
                st.error(f"Metin UTF-8'e dönüştürülürken hata: {e}")
                secret_data_to_embed_bytes = None
        else:
            st.info("Gizlemek için bir metin girin.")
    else:
        secret_file = st.file_uploader(f"Gizlenecek dosyayı yükleyin (Maksimum {MAX_SECRET_SIZE_MB} MB):", type=None, key="secret_file")
        if secret_file is not None:
            if secret_file.size > MAX_SECRET_SIZE_BYTES:
                st.error(f"Gizlenecek dosya '{secret_file.name}' boyutu limiti aşıyor.")
            else:
                original_secret_filename = secret_file.name
                secret_data_to_embed_bytes = secret_file.getvalue()

    st.subheader("2. Taşıyıcı Medya")
    uploaded_media_file = None
    media_source = None

    if "Resim" in media_type:
        media_source = st.radio("Görsel kaynağı:", ("Dosya yükle", "AI ile oluştur"), key="image_source")
        if media_source == "AI ile oluştur":
            st.markdown("#### AI ile Görsel Oluşturma")
            ai_prompt = st.text_input("Görsel için açıklama (prompt):", value="Dijital arkaplan", key="ai_prompt")
            resolution_options = ["128x128", "256x256", "384x384", "512x512"]
            default_resolution_str = "128x128"
            selected_resolution_str = st.select_slider("Görsel çözünürlüğü:", options=resolution_options, value=default_resolution_str, key="ai_res_str")
            try:
                width_str, height_str = selected_resolution_str.split('x')
                ai_resolution_tuple = (int(width_str), int(height_str))
            except Exception:
                ai_resolution_tuple = (256, 256)

            if 'ai_generated_image' not in st.session_state:
                st.session_state.ai_generated_image = None
            if 'last_ai_prompt' not in st.session_state:
                st.session_state.last_ai_prompt = ""
            if 'last_ai_res_str' not in st.session_state:
                st.session_state.last_ai_res_str = ""

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Önizleme Oluştur/Yenile", key="ai_preview"):
                    if ai_prompt:
                        with st.spinner("AI görsel oluşturuluyor..."):
                            st.session_state.ai_generated_image = generate_ai_image(ai_prompt, ai_resolution_tuple[0], ai_resolution_tuple[1])
                            st.session_state.last_ai_prompt = ai_prompt
                            st.session_state.last_ai_res_str = selected_resolution_str
                            st.success("AI görsel hazır.")
                    else:
                        st.warning("Lütfen görsel için bir açıklama girin.")
            if st.session_state.ai_generated_image:
                with col2:
                    caption_res = st.session_state.get('last_ai_res_str', default_resolution_str)
                    st.image(st.session_state.ai_generated_image, caption=f"Oluşturulan: '{st.session_state.last_ai_prompt}' ({caption_res})", width="stretch")
                    st.session_state.ai_generated_image.seek(0)
                    uploaded_media_file = st.session_state.ai_generated_image
        else:
            uploaded_media_file = st.file_uploader(
                f"Taşıyıcı görsel dosyasını yükleyin (PNG, BMP önerilir) (Maksimum {MAX_CARRIER_SIZE_MB} MB):",
                type=["png", "bmp", "tiff", "jpg", "jpeg"],
                key="carrier_image_upload")
    elif "Ses" in media_type:
        uploaded_media_file = st.file_uploader(
            f"Taşıyıcı ses dosyasını yükleyin (WAV, FLAC vb.) (Maksimum {MAX_CARRIER_SIZE_MB} MB):",
            type=["wav", "mp3", "flac", "aac", "ogg", "aiff"],
            key="carrier_audio_upload")
    elif "Video" in media_type:
        uploaded_media_file = st.file_uploader(
            f"Taşıyıcı video dosyasını yükleyin (Maksimum {MAX_CARRIER_SIZE_MB} MB):",
            type=["mp4", "avi", "mkv", "mov", "mpeg", "wmv"],
            key="carrier_video_upload")

    if uploaded_media_file and media_source != "AI ile oluştur":
        if hasattr(uploaded_media_file, 'size') and uploaded_media_file.size > MAX_CARRIER_SIZE_BYTES:
            st.error(f"Taşıyıcı medya dosyası boyutu limiti aşıyor.")
            uploaded_media_file = None
        elif hasattr(uploaded_media_file, 'getvalue') and len(uploaded_media_file.getvalue()) > MAX_CARRIER_SIZE_BYTES:
            st.error(f"Oluşturulan AI görselin boyutu beklenmedik şekilde limiti aşıyor.")
            uploaded_media_file = None

    st.subheader("Gömme yöntemi")
    embed_method = st.selectbox("Yöntem seç:", ("Standard LSB", "LSB-Matching", "Adaptive LSB"), index=0)

    st.subheader("3. Gizleme İşlemi")
    st.markdown("---")

    if st.button("Veriyi Gizle ve Şifrele", key="encode_button"):
        valid_input = True
        if secret_data_to_embed_bytes is None:
            st.error("Lütfen gizlenecek bir metin girin veya geçerli bir dosya yükleyin.")
            valid_input = False

        if media_source == "AI ile oluştur":
            if st.session_state.ai_generated_image is None:
                st.error("Lütfen önce bir AI görseli oluşturun veya 'Dosya yükle' seçeneğini kullanın.")
                valid_input = False
            else:
                uploaded_media_file = st.session_state.ai_generated_image
                uploaded_media_file.seek(0)
                carrier_filename_for_output = "ai_generated_image"
        elif uploaded_media_file is None:
            st.error(f"Lütfen bir taşıyıcı {media_type.split(' ')[0].lower()} dosyası yükleyin.")
            valid_input = False
        else:
            carrier_filename_for_output = os.path.splitext(uploaded_media_file.name)[0]

        if valid_input:
            with st.spinner(f"{media_type} içine veri gizleniyor ve şifreleniyor... Lütfen bekleyin..."):
                try:
                    encrypted_json_data = encrypt_data(secret_data_to_embed_bytes, password, original_secret_filename)
                    if encrypted_json_data is None:
                        raise ValueError("Şifreleme başarısız oldu.")
                    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename_base = f"{now_str}_steg_{carrier_filename_for_output}"
                    output_bytes = None
                    final_output_filename_from_func = None

                    if "Resim" in media_type:
                        output_filename = output_filename_base + ".png"
                        output_bytes = encode_lsb(uploaded_media_file, encrypted_json_data, output_filename, method=embed_method)
                        final_output_filename_from_func = output_filename
                    elif "Ses" in media_type:
                        output_filename = output_filename_base + ".wav"
                        output_bytes = encode_lsb_audio(uploaded_media_file, encrypted_json_data, output_filename)
                        final_output_filename_from_func = output_filename
                    elif "Video" in media_type:
                        output_filename_suggestion = output_filename_base + ".mkv"
                        output_bytes, final_output_filename_from_func = encode_lsb_video(uploaded_media_file, encrypted_json_data, output_filename_suggestion)

                    if output_bytes and final_output_filename_from_func:
                        st.success("Veri başarıyla gizlendi ve şifrelendi!")
                        st.info(f"Oluşturulan Dosya: {os.path.basename(final_output_filename_from_func)}")

                        mime_type = "application/octet-stream"
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
                        st.error("Veri gizleme işlemi başarısız oldu. Yukarıdaki hata mesajlarını kontrol edin.")
                except ValueError as ve:
                    st.error(f"İşlem Hatası: {ve}")
                except Exception as e:
                    st.error(f"Gizleme işlemi sırasında beklenmedik bir hata oluştu: {e}")
                    import traceback
                    print(traceback.format_exc())
                    st.info("Girdi dosyalarınızın formatını ve boyutunu kontrol edin. Gerekli programlar (ffmpeg, ffprobe) kurulu mu?")

# --- Decode ---
elif operation == "Çöz (Decode)":
    st.header(f" secretive Veri Çözme ({media_type})")
    decode_file_types = []
    if "Resim" in media_type:
        decode_file_types = ["png", "bmp", "tiff"]
        st.info("Yalnızca PNG, BMP, TIFF gibi kayıpsız formatlarda gizlenmiş veriler güvenilir şekilde çözülebilir.")
    elif "Ses" in media_type:
        decode_file_types = ["wav"]
        st.info("Ses çözme işlemi yalnızca '.wav' formatındaki dosyaları destekler.")
    elif "Video" in media_type:
        decode_file_types = ["avi", "mkv"]
        st.info("Video çözme işlemi genellikle '.avi' veya '.mkv' dosyalarını destekler.")

    steg_media_file = st.file_uploader(
        f"İçinde gizli veri olan {media_type.split(' ')[0].lower()} dosyasını yükleyin:",
        type=decode_file_types,
        key="steg_file_upload"
    )

    st.subheader("Çözme yöntemi")
    decode_method = st.selectbox("Hangi yöntemle gömüldüğünü seçin:", ("Standard LSB", "LSB-Matching", "Adaptive LSB"), index=0)

    st.markdown("---")

    if st.button("Veriyi Çöz", key="decode_button"):
        valid_input = True
        if steg_media_file is None:
            st.error(f"Lütfen çözülecek bir {media_type.split(' ')[0].lower()} dosyası yükleyin.")
            valid_input = False

        if valid_input:
            with st.spinner(f"{media_type} içinden veri çıkarılıyor ve şifre çözülüyor..."):
                try:
                    extracted_json_str = None
                    if "Resim" in media_type:
                        extracted_bytes = decode_lsb(steg_media_file, method=decode_method)
                        if extracted_bytes is None:
                            extracted_json_str = None
                        else:
                            # decode_lsb returns bytes for adaptive/simple. try to decode to str
                            try:
                                extracted_json_str = extracted_bytes.decode('utf-8')
                            except Exception:
                                extracted_json_str = None
                    elif "Ses" in media_type:
                        extracted_json_str = decode_lsb_audio(steg_media_file)
                    elif "Video" in media_type:
                        extracted_json_str = decode_lsb_video(steg_media_file)

                    if extracted_json_str:
                        decrypted_bytes, retrieved_filename = decrypt_data(extracted_json_str, password)
                        if decrypted_bytes is not None:
                            st.success("Veri başarıyla çıkarıldı ve şifresi çözüldü!")
                            try:
                                decoded_text = decrypted_bytes.decode('utf-8')
                                st.subheader("Çözülen Metin:")
                                st.text_area("Metin:", decoded_text, height=150, key="decoded_text_area")
                            except UnicodeDecodeError:
                                st.subheader("Çözülen Dosya:")
                                now = datetime.datetime.now()
                                timestamp = now.strftime("%Y%m%d_%H%M%S")
                                if retrieved_filename:
                                    if '.' in retrieved_filename:
                                        file_name_to_download = f"{timestamp}_decrypted_{retrieved_filename}"
                                    else:
                                        file_name_to_download = f"{timestamp}_decrypted_{retrieved_filename}.bin"
                                else:
                                    file_name_to_download = f"{timestamp}_decrypted_file.bin"
                                mime_type = "application/octet-stream"
                                if retrieved_filename:
                                    mime_guess = mimetypes.guess_type(retrieved_filename)[0]
                                    if mime_guess:
                                        mime_type = mime_guess
                                st.download_button(
                                    label=f"Çözülen Dosyayı İndir ({os.path.basename(file_name_to_download)})",
                                    data=decrypted_bytes,
                                    file_name=os.path.basename(file_name_to_download),
                                    mime=mime_type
                                )
                        else:
                            pass
                    else:
                        st.error("Dosyadan gizli veri çıkarılamadı. Dosya formatı doğru mu? Bu dosya içine veri gömülmüş mü?")
                except Exception as e:
                    st.error(f"Çözme işlemi sırasında beklenmedik bir hata oluştu: {e}")
                    import traceback
                    print(traceback.format_exc())
                    st.info("İpucu: Dosya türü ve şifrenizi kontrol edin. Dosyanın içinde veri olduğundan emin olun.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Bu uygulama LSB (Least Significant Bit) steganografi tekniklerini ve AES şifrelemesini kullanır.")
st.sidebar.warning("Büyük dosyalarla çalışmak zaman alabilir ve yüksek bellek kullanımı gerektirebilir.")
st.sidebar.markdown("Geliştirici: Ali11git")
