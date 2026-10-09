import argparse
import os
import sys

def download_model(size="small"):
    repo_id = f"microsoft/trocr-{size}-handwritten"
    local_dir = f"models/trocr-{size}-handwritten"
    
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)
        
    try:
        snapshot_download(repo_id, local_dir=local_dir, local_files_only=False)
        print("Download complete.")
        
        # Calculate folder size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(local_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
                    
        print(f"Folder size: {total_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Download failed: {e}")
        print("If you have network issues, run:")
        print(f"python -c \"from huggingface_hub import snapshot_download as d; d('{repo_id}', local_dir='{local_dir}')\"")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TrOCR models")
    parser.add_argument("--size", choices=["small", "base", "both"], default="small", help="Model size to download")
    args = parser.parse_args()
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    if args.size in ["small", "both"]:
        download_model("small")
    if args.size in ["base", "both"]:
        download_model("base")
