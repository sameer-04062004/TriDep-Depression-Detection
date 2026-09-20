"""
TriDep — Deploy to Hugging Face Spaces
Uploads the contents of hf_space/ directly to your Hugging Face Space.
"""
import os
import sys
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import HfApi, create_repo

REPO_ID = "sameer-04062004/TriDep-Depression-Detection"
LOCAL_DIR = Path(__file__).resolve().parent / "hf_space"

print("=" * 60)
print(f"  Deploying TriDep to Hugging Face Spaces: {REPO_ID}")
print("=" * 60)

# Check for token
token = os.environ.get("HF_TOKEN")
if not token:
    print("\nPlease enter your Hugging Face Write Token:")
    print("👉 Get it here: https://huggingface.co/settings/tokens (Role: Write)\n")
    token = input("HF Token: ").strip()

if not token:
    print("❌ Error: Token is required to deploy to Hugging Face.")
    sys.exit(1)

api = HfApi(token=token)

try:
    user_info = api.whoami()
    username = user_info.get("name")
    print(f"\n✅ Authenticated as: {username}")
    
    # Adjust repo ID if token belongs to a different username
    target_repo = f"{username}/TriDep-Depression-Detection"
    
    print(f"Checking Space repository: {target_repo} ...")
    api.create_repo(
        repo_id=target_repo,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True
    )
    print("✅ Space repository ready!")

    print(f"Uploading files from {LOCAL_DIR} ...")
    api.upload_folder(
        folder_path=str(LOCAL_DIR),
        repo_id=target_repo,
        repo_type="space",
        commit_message="Deploy TriDep Multimodal Depression Detection app"
    )

    space_url = f"https://huggingface.co/spaces/{target_repo}"
    print("\n" + "=" * 60)
    print(f"🎉 Deployment Complete!")
    print(f"👉 Live App URL: {space_url}")
    print("=" * 60 + "\n")

except Exception as e:
    print(f"\n❌ Deployment failed: {e}")
    print("\nAlternative manual upload:")
    print(f"1. Go to https://huggingface.co/new-space")
    print(f"2. Name: TriDep-Depression-Detection, SDK: Gradio")
    print(f"3. Upload the files inside the 'hf_space' folder directly via the 'Files' tab!")
