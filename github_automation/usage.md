# Safe GitHub Automation GUI

I fixed/replaced the broken script with a safer version:

```text
git_workflow_gui.py
```

## Why the old script was risky

The old flow did this near commit time:

```bash
git checkout dev
git pull origin dev
git checkout -b features/something
```

That can be dangerous if you already edited files, because switching branches/pulling while files are modified can cause conflicts or mix work into the wrong branch.

The new script uses a safer workflow:

1. Safely update `dev`
2. Create/switch to a feature branch before editing
3. Open VS Code
4. Edit files
5. Select changed files
6. Commit selected files only
7. Push branch
8. Open GitHub compare/PR link

## New safety features

### 1. Refuses risky `dev` updates

The app will not pull `dev` if you have uncommitted files.

It shows a message telling you to commit, stash, or discard first.

### 2. Uses fast-forward only

For updating `dev`, it uses:

```bash
git pull --ff-only origin dev
```

This avoids surprise merge commits.

### 3. Detects ahead/behind/diverged `dev`

If local `dev` is:

- behind `origin/dev`: it safely fast-forwards
- ahead of `origin/dev`: it stops and asks you to handle it manually
- ahead and behind: it stops because branches diverged

### 4. Encourages feature branches before editing

New button:

```text
2. Create/switch feature branch
```

Use this before opening VS Code and editing.

### 5. Safer commit behavior

When committing:

- If you are already on the feature branch, it commits there.
- If you are on `dev` with uncommitted files and the target feature branch does not exist, it creates the feature branch while preserving your work.
- If switching branches would be unsafe, it refuses.

### 6. Selected files only

It only runs:

```bash
git add -- selected-file
```

for the files you chose in the UI.

### 7. Large files use Git LFS

Files larger than the threshold are tracked with Git LFS:

```bash
git lfs track path/to/file
```

and `.gitattributes` is included automatically.

## Recommended usage

1. Run the app.
2. Choose repo folder, for example:

```text
D:\EEG
```

3. Click:

```text
Validate
```

4. Click:

```text
1. Safely update dev
```

5. Enter branch name, for example:

```text
features/eeg-loader
```

6. Click:

```text
2. Create/switch feature branch
```

7. Click:

```text
3. Open VS Code
```

8. Edit your files.
9. Go back to the app and click:

```text
4. Refresh changed files
```

10. Select files to commit.
11. Enter commit message.
12. Click:

```text
5. Commit + push + review
```

## Build to EXE

Put `git_workflow_gui.py` inside your repo or another folder, then run:

```powershell
cd /d D:\EEG
pyinstaller --onefile --windowed --name GitWorkflowGUI git_workflow_gui.py
```

If you put it in a subfolder:

```powershell
cd /d D:\EEG
pyinstaller --onefile --windowed --name GitWorkflowGUI github-automation\git_workflow_gui.py
```

The output will be:

```text
D:\EEG\dist\GitWorkflowGUI.exe
```

## Important

Do not copy the Python script from rendered Markdown/chat text if it contains things like:

```text
from **future** import annotations
[p.is](http://p.is)_file()
git_workflow_[gui.py](http://gui.py)
```

That means the code was corrupted by Markdown formatting.

Use the actual file:

```text
git_workflow_gui.py
```
