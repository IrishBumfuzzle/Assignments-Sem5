#!/usr/bin/env python3
"""Upload trained checkpoints, tokenizers, configs and results for C1-C5 to HuggingFace Hub."""

import argparse
import os
import sys
from pathlib import Path

def upload_config_dir(api, output_dir, repo_id, token=None):
    print(f"\n==========================================")
    print(f"Uploading {output_dir} -> https://huggingface.co/{repo_id}")
    print(f"==========================================")
    
    if not output_dir.exists():
        print(f"[ERROR] Directory does not exist: {output_dir}")
        return False
        
    try:
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
        print(f"[hf] Repo {repo_id} verified / created.")
    except Exception as e:
        print(f"[hf ERROR] Could not create/access repo {repo_id}: {e}")
        return False

    # Check if there is a nested run directory (e.g. outputs/C1/C1)
    nested_dir = output_dir / output_dir.name
    search_dirs = [nested_dir, output_dir] if nested_dir.is_dir() else [output_dir]

    # Collect files to upload
    files_to_upload = {}
    for d in search_dirs:
        for fname in (
            "model_best.pt",
            "model_last.pt",
            "config.json",
            "results.json",
            "test_samples.txt",
            "training_curves.png",
            "val_samples_epoch1.txt",
        ):
            fp = d / fname
            if fp.exists() and fname not in files_to_upload:
                files_to_upload[fname] = fp

        tok_dir = d / "tokenizers"
        if tok_dir.is_dir():
            for tf in tok_dir.iterdir():
                if tf.is_file() and tf.name not in files_to_upload:
                    files_to_upload[tf.name] = tf

    if not files_to_upload:
        print(f"[WARNING] No files found to upload in {output_dir}")
        return False

    for name, fpath in files_to_upload.items():
        print(f"  Uploading {name} ({fpath.stat().st_size / 1e6:.2f} MB)...")
        try:
            url = api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=name,
                repo_id=repo_id,
                token=token,
            )
            print(f"    -> Done: {url}")
        except Exception as e:
            print(f"    -> [ERROR] Failed to upload {name}: {e}")
            return False

    print(f"[SUCCESS] All files for {repo_id} uploaded successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload trained C1-C5 models to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (hf_...)")
    parser.add_argument("--user", type=str, default="IrishBumfuzzle", help="Hugging Face username / org")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Outputs base directory")
    parser.add_argument("--configs", nargs="+", default=["C1", "C2", "C3", "C4", "C5"], help="Configs to upload")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[ERROR] Hugging Face token not provided.")
        print("Please pass --token <your_hf_token> or export HF_TOKEN=<your_hf_token>")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = api.whoami(token=token)
        print(f"Authenticated with Hugging Face as: {user_info['name']}")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate with provided Hugging Face token: {e}")
        sys.exit(1)

    base_dir = Path(args.outputs_dir)
    success = True
    for cfg in args.configs:
        cfg_dir = base_dir / cfg
        repo_id = f"{args.user}/anlp-a1-{cfg}"
        res = upload_config_dir(api, cfg_dir, repo_id, token)
        if not res:
            success = False

    if success:
        print("\nAll configurations successfully uploaded to Hugging Face!")
    else:
        print("\nSome uploads failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
