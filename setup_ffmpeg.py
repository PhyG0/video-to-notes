
import os
import zipfile
import urllib.request
import shutil
import glob

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
DOWNLOAD_path = "ffmpeg.zip"
EXTRACT_path = "ffmpeg_extracted"
BIN_path = "bin"

def setup_ffmpeg():
    print(f"Downloading FFmpeg from {FFMPEG_URL}...")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, DOWNLOAD_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print("Extracting archive...")
    with zipfile.ZipFile(DOWNLOAD_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_path)

    # Find the bin directory inside the extracted folder
    # Usually it's ffmpeg-*-essentials_build/bin
    bin_dirs = glob.glob(os.path.join(EXTRACT_path, "**", "bin"), recursive=True)
    
    if not bin_dirs:
        print("Could not find bin directory in extracted archive.")
        return

    ffmpeg_bin = bin_dirs[0]
    
    if not os.path.exists(BIN_path):
        os.makedirs(BIN_path)

    for file in os.listdir(ffmpeg_bin):
        if file.endswith(".exe"):
            shutil.copy(os.path.join(ffmpeg_bin, file), os.path.join(BIN_path, file))
            print(f"Copied {file} to {BIN_path}")

    print("Cleaning up...")
    os.remove(DOWNLOAD_path)
    shutil.rmtree(EXTRACT_path)
    print("FFmpeg setup complete.")

if __name__ == "__main__":
    setup_ffmpeg()
