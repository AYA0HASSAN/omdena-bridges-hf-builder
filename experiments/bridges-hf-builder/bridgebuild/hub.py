"""Where the dataset lives: a Hugging Face dataset repo, or a local folder (dry runs and tests).

Both expose the same small interface: commit(files), get(path), exists(path), list(prefix).
One commit per shard keeps the commit count low (the Hub recommends < ~100 files per commit and a few
thousand commits per repo at most).
"""
import os
import shutil
from pathlib import Path


class LocalHub:
    def __init__(self, folder):
        self.root = Path(folder)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_id = str(self.root)

    def commit(self, files, message=""):
        """files: {path_in_repo: local_path}"""
        for dst, src in files.items():
            p = self.root / dst
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, p)

    def get(self, path, cache_dir=None):
        p = self.root / path
        return str(p) if p.exists() else None

    def exists(self, path):
        return (self.root / path).exists()

    def list(self, prefix=""):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()
                      and str(p.relative_to(self.root)).startswith(prefix))

    def url(self):
        return str(self.root.resolve())


class HFHub:
    def __init__(self, repo_id, private=True, token=None, revision="main"):
        from huggingface_hub import HfApi
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        self.repo_id, self.revision = repo_id, revision
        self.api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    def commit(self, files, message=""):
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for dst, src in files.items()]
        for attempt in range(4):
            try:
                self.api.create_commit(self.repo_id, ops, commit_message=message or "update", repo_type="dataset",
                                       revision=self.revision)
                return
            except Exception as e:
                if attempt == 3:
                    raise
                import time
                print(f"  [hub] commit failed ({str(e)[:80]}), retrying")
                time.sleep(10 * (attempt + 1))

    def get(self, path, cache_dir=None):
        from huggingface_hub import hf_hub_download
        if not self.exists(path):
            return None
        return hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision,
                               local_dir=cache_dir) if cache_dir else \
            hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision)

    def exists(self, path):
        return self.api.file_exists(self.repo_id, path, repo_type="dataset", revision=self.revision)

    def list(self, prefix=""):
        return sorted(f for f in self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision)
                      if f.startswith(prefix))

    def url(self):
        return f"https://huggingface.co/datasets/{self.repo_id}"


def open_hub(cfg):
    h = cfg.get("hub", {})
    if h.get("local_dir"):
        return LocalHub(h["local_dir"])
    return HFHub(h["repo_id"], private=h.get("private", True), revision=h.get("revision", "main"))
