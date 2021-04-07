import os

def get_current_branch():
    refs = os.popen("git branch --format='%(refname)' -a --contains $(git rev-list --max-count 1 --all)").read().strip().split("\n")
    for ref in refs:
        if ref.startswith("refs/heads/"):
            return ref.replace("refs/heads/", "")
    return None
