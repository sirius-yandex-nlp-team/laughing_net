import os
import requests

token = os.environ['TOKEN']
repo = os.environ['REPO']
owner = os.environ['OWNER']
issue_number = os.environ['ISSUE_NUMBER']

headers = {
    'Accept': 'application/vnd.github.starfox-preview+json',
    'Authorization': f'token {token}',
}

labels = [
    {
        'full': 'hypothesis',
        'short': 'HYP',
    },
    {
        'full': 'task',
        'short': 'TASK',
    }
]

def get_issue():
    response = requests.get(
        f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}', 
        headers=headers, 
    )
    return response.json()

def get_labels(issue):
    return [i['name'] for i in issue['labels']]

def remove_title_prefix(title, prefix):
    title = title.strip()
    if title.startswith(f'[{prefix}') and ']' in title:
        title = title[title.find(']')+1:].strip()
    return title

def set_title_prefix(title, prefix):
    title = f'[{prefix}-{issue_number}] {title}'
    return title

def process_title(issue):
    title = issue['title']
    issue_labels = get_labels(issue)
    for label in labels:
        title = remove_title_prefix(title, label['short'])
    for label in labels:
        if label['full'] in issue_labels:
            return set_title_prefix(title, label['short'])
    return title

def get_changes(issue):
    new_title = process_title(issue)
    changes = f'{{"title":"{new_title}"}}'
    return changes

def apply_changes(changes):
    response = requests.patch(
        f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}', 
        headers=headers, 
        data=changes
    )

def main():
    issue = get_issue()
    changes = get_changes(issue)
    apply_changes(changes)

if __name__ == "__main__":
    main()
