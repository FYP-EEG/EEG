# Git Workflow Automation GUI

This workspace contains `git_workflow_gui.py`, a Windows-friendly Python/Tkinter app you can turn into an `.exe`.

## What it does

The app provides a UI for the workflow you described:

1. **Get up-to-date files**
   - `git fetch origin`
   - `git checkout dev`
   - `git pull origin dev`
2. **Open VS Code** for the repository folder.
3. **Choose changed files** from a checkbox-style list.
4. **Automatically use Git LFS pointers for large selected files** above the configured MB threshold.
5. Ask for a branch name such as `features/something`.
6. Ask for a commit message.
7. Commit and push to that branch.
8. Open the GitHub compare page for code review:
   - `https://github.com/<owner>/<repo>/compare/dev...features/something?expand=1`

## Requirements

Install these on the machine where you run the app:

- Python 3.10+
- Git
- VS Code, with the `code` command available in PATH
- Git LFS if you will push large files: <https://git-lfs.com/>
- PyInstaller, only needed to build the `.exe`

## Build the EXE

Open Command Prompt or PowerShell in the folder containing `git_workflow_gui.py`, then run:

```powershell
pip install pyinstaller
pyinstaller --onefile --windowed --name GitWorkflowGUI git_workflow_gui.py
```

Your `.exe` will be created here:

```text
dist/GitWorkflowGUI.exe
```

You can copy that `.exe` anywhere, but the target machine still needs Git installed and authenticated for GitHub.

## Recommended first-time setup

Before using the app, confirm these commands work manually in your repository:

```powershell
git status
git checkout dev
git pull origin dev
git lfs version
code .
```

If `git lfs version` fails, install Git LFS and run:

```powershell
git lfs install
```

If `code .` fails, open VS Code, press `Ctrl+Shift+P`, and search for:

```text
Shell Command: Install 'code' command in PATH
```

On Windows, you may instead need to reinstall VS Code and enable **Add to PATH**.

## How to use

1. Run `GitWorkflowGUI.exe` or run the script directly:

   ```powershell
   python git_workflow_gui.py
   ```

2. Browse to your repository folder. For your example, it sounds like the repo may be at `D:\` or a folder under `D:\`.
3. Click **Validate**.
4. Click **1. Get up-to-date dev**.
5. Click **2. Open VS Code**.
6. Make your edits.
7. Return to the app and click **3. Refresh changed files**.
8. Click rows to select/deselect files.
9. Enter a feature branch name, for example:

   ```text
   features/eeg-signal-loader
   ```

10. Enter a commit message.
11. Click **4. Commit + push + open review link**.

## Notes for your branch structure

Your `dev` branch can contain the full working structure:

```text
D:.
|   eeg script
|   learningmaterial.md
|   pointer for trained model
|   readme.md
|   roadmap.md
|
+---devlog-anson
+---devlog-brian
+---EEG_signals
\---pygame
```

You said `main` should only have:

```text
EEG_signals/
pygame/library scripts
eeg script
readme.md
roadmap.md
```

This app uses `dev` as the base branch and opens review links comparing your feature branch back to `dev`. It does **not** automatically clean or restrict the `main` branch. If you want, I can also add a separate **Prepare main release** button that copies/merges only those allowed paths into `main`.

## Large files and pointers

The app uses Git LFS for selected files larger than the threshold shown in the UI. It runs commands similar to:

```powershell
git lfs install
git lfs track path/to/large-file
git add .gitattributes
git add path/to/large-file
```

Git LFS stores a small pointer in Git and uploads the actual large file to LFS storage.

## Safety behavior

- The app only stages files you selected.
- It refuses to commit if no files are staged.
- It prevents using `main`, `master`, or `dev` as the feature branch name.
- If the feature branch already exists locally, it checks it out and merges the latest `dev` into it.
