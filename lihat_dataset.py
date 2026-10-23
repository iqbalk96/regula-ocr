from huggingface_hub import HfApi

api = HfApi()

repo_id = "Azzindani/ID_REG"

files = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)

for file in files:
    if hasattr(file, "size"):
        size_mb = file.size / (1024 * 1024)
        print(f"{size_mb:10.2f} MB  {file.path}")
