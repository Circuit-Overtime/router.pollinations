import os
from huggingface_hub import snapshot_download

def download_qwen_0_5b():
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    CACHE_DIR = "models"
    MODEL_PATH = os.path.join(CACHE_DIR, "qwen2.5-0.5b-instruct")
    
    print(f"🚀 Starting Qwen2.5-0.5B-Instruct model download...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Save path: {MODEL_PATH}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        print("\n📥 Downloading model files...")
        local_dir = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"\n✅ Download complete!")
        print(f"Model saved to: {local_dir}")

        print("\n🔍 Listing downloaded files:")
        for f in os.listdir(local_dir):
            print("  •", f)

        print("\n✨ Model is ready!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False


if __name__ == "__main__":
    download_qwen_0_5b()
