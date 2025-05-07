İstediğiniz tam olarak mümkün! FFmpeg'i **doğrudan bellek üzerinde çalıştırmak** için Python'da `subprocess` modülüyle **pipe'ları** kullanabilirsiniz. Streamlit ile entegre çözüm için adımlar:

---

### 1. **Streamlit'te Dosyayı Belleğe Alma**
Streamlit'in `st.file_uploader`'ı dosyayı zaten `BytesIO` benzeri bir bellek nesnesi olarak verir. Disk yazma/silme yapmanıza gerek yok:
```python
uploaded_file = st.file_uploader("Dosya Yükle", type=["mp4", "avi"])
if uploaded_file:
    input_data = uploaded_file.read()  # Dosyayı direkt belleğe al
```

---

### 2. **FFmpeg'i Pipe ile Çalıştırma**
Geçici dosya kullanmadan veriyi **stdin** üzerinden FFmpeg'e gönderip **stdout**'tan sonucu alın:

```python
import subprocess
from io import BytesIO

def convert_with_ffmpeg(input_bytes, output_format="mp4"):
    # FFmpeg komutunu hazırla (stdin/stdout kullanarak)
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite izni
        "-i", "pipe:0",      # Girdiyi stdin'den al
        "-f", output_format, # Çıktı formatını belirt (ÖNEMLİ!)
        "pipe:1"            # Çıktıyı stdout'a yaz
    ]
    
    # Prosesi başlat
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Veriyi gönder ve çıktıyı al
    stdout_data, stderr_data = process.communicate(input=input_bytes)
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg Error: {stderr_data.decode()}")
    
    return BytesIO(stdout_data)
```

---

### 3. **Streamlit'te Kullanım**
Dönüştürülen veriyi doğrudan indirme butonuna bağla:
```python
if uploaded_file:
    # FFmpeg ile dönüştür
    output_buffer = convert_with_ffmpeg(input_data, "mp4")
    
    # İndirme butonu
    st.download_button(
        label="Dönüştürülen Dosyayı İndir",
        data=output_buffer,
        file_name=f"converted.{output_format}"
    )
```

---

### 4. **Önemli Notlar**
- **Format Belirtme (`-f`):** FFmpeg'in pipe üzerinde çalışırken çıktı formatını (`-f mp4`) açıkça belirtin. Örneğin MP4 için `-f mp4` kullanın.
- **Bellek Yönetimi:** Büyük dosyalarda `stdout_data`'yı parçalar halinde (`chunks`) okuyup işleyebilirsiniz.
- **Hata Kontrolü:** `stderr_data`'yı kontrol ederek FFmpeg hatalarını yakalayın.

---

### 5. **Tam Örnek Kod**
```python
import streamlit as st
import subprocess
from io import BytesIO

def convert_with_ffmpeg(input_bytes, output_format):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", "pipe:0",
        "-f", output_format,
        "pipe:1"
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=input_bytes)
    if process.returncode != 0:
        st.error(f"Hata: {stderr.decode()}")
        return None
    return BytesIO(stdout)

# Streamlit UI
st.title("Steganografi Dönüştürücü")
uploaded_file = st.file_uploader("Dosya Seç", type=["mp4", "avi", "mov"])
output_format = st.selectbox("Çıktı Formatı", ["mp4", "avi", "mov"])

if uploaded_file and output_format:
    input_data = uploaded_file.read()
    output_buffer = convert_with_ffmpeg(input_data, output_format)
    if output_buffer:
        st.download_button(
            "İndir",
            data=output_buffer,
            file_name=f"converted.{output_format}"
        )
```

---

### 6. **Optimizasyonlar**
- **Büyük Dosyalar için:** `communicate()` yerine stdin/stdout'u parçalar halinde okuyup yazabilirsiniz.
- **Progress Bar:** FFmpeg'in ilerlemesini Streamlit'te göstermek için özel çözümler kullanabilirsiniz.

Bu yöntemle **geçici dosya yazma/silme** işlemlerini tamamen ortadan kaldırabilirsiniz! 🚀
