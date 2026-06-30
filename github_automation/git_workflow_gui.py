"""
Git Workflow Automation GUI

A Windows-friendly Tkinter app that automates a common GitHub workflow:
- checkout dev
- pull origin dev
- open VS Code
- choose changed files with checkboxes
- automatically track large selected files with Git LFS pointers
- ask for a feature branch name and commit message
- commit and push to that branch
- open the GitHub compare / pull-request page

Build to .exe with:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name GitWorkflowGUI git_workflow_gui.py

Requirements on the target machine:
- Git installed and available in PATH
- VS Code command `code` available in PATH, or use the app's fallback opener
- Git LFS installed if you want large-file pointer support: https://git-lfs.com/

Notes:
- Default base branch is `dev` because your workflow is mainly for the dev branch.
- Large files are tracked through Git LFS when selected and larger than LARGE_FILE_MB.
- The app opens a GitHub compare URL after pushing: /compare/dev...your-branch?expand=1
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, Text, END, DISABLED, NORMAL, messagebox, filedialog
from tkinter import ttk


# ------------------------------ Configuration ------------------------------ #

BASE_BRANCH = "dev"
REMOTE_NAME = "origin"
LARGE_FILE_MB = 50
DEFAULT_REPO_PATH = str(Path.cwd())

# Optional helper documentation for your branch structure.
MAIN_BRANCH_EXPECTED_PATHS = [
    "EEG_signals/",
    "pygame/",  # especially pygame/library scripts and pygame/assets if needed
    "eeg script",
    "readme.md",
    "roadmap.md",
]


# ------------------------------ Git Utilities ------------------------------ #

class GitError(RuntimeError):
    pass


def run_cmd(args: list[str], cwd: Path, check: bool = True) -> str:
    """Run a command and return combined stdout/stderr text."""
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    proc = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        startupinfo=startupinfo,
    )
    output = proc.stdout.strip()
    if check and proc.returncode != 0:
        raise GitError(f"Command failed: {' '.join(shlex.quote(a) for a in args)}\n\n{output}")
    return output


def git(args: list[str], cwd: Path, check: bool = True) -> str:
    return run_cmd(["git", *args], cwd, check=check)


def is_git_repo(path: Path) -> bool:
    try:
        out = git(["rev-parse", "--is-inside-work-tree"], path)
        return out.strip().lower() == "true"
    except Exception:
        return False


def repo_root(path: Path) -> Path:
    return Path(git(["rev-parse", "--show-toplevel"], path)).resolve()


def normalize_remote_url(remote: str) -> str | None:
    """Convert common Git remote formats into https://github.com/owner/repo."""
    remote = remote.strip()
    if not remote:
        return None

    # git@github.com:owner/repo.git
    m = re.match(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?$", remote)
    if m:
        return f"https://github.com/{m.group('owner')}/{m.group('repo')}"

    # ssh://git@github.com/owner/repo.git
    m = re.match(r"ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?$", remote)
    if m:
        return f"https://github.com/{m.group('owner')}/{m.group('repo')}"

    # https://github.com/owner/repo.git
    m = re.match(r"(?P<url>https://github\.com/[^/]+/.+?)(?:\.git)?$", remote)
    if m:
        return m.group("url")

    return None


def compare_url(cwd: Path, branch: str, base: str = BASE_BRANCH) -> str | None:
    try:
        remote = git(["remote", "get-url", REMOTE_NAME], cwd)
    except Exception:
        return None
    base_url = normalize_remote_url(remote)
    if not base_url:
        return None
    safe_branch = branch.replace("#", "%23")
    return f"{base_url}/compare/{base}...{safe_branch}?expand=1"


def lfs_available(cwd: Path) -> bool:
    try:
        git(["lfs", "version"], cwd)
        return True
    except Exception:
        return False


def parse_status_porcelain_z(raw: bytes) -> list[tuple[str, str]]:
    """
    Parse `git status --porcelain=v1 -z`.
    Returns list of (status_code, path). Handles rename records enough for selection.
    """
    entries = raw.split(b"\0")
    result: list[tuple[str, str]] = []
    i = 0
    while i < len(entries):
        item = entries[i]
        if not item:
            i += 1
            continue
        text = item.decode("utf-8", errors="replace")
        status = text[:2]
        path = text[3:]
        if status.startswith("R") or status.startswith("C"):
            # Next entry is the old path. We add only the new path.
            i += 1
        result.append((status, path))
        i += 1
    return result


def get_changed_files(cwd: Path) -> list[tuple[str, str]]:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise GitError(proc.stderr.decode("utf-8", errors="replace"))
    return parse_status_porcelain_z(proc.stdout)


def file_size_mb(cwd: Path, rel_path: str) -> float:
    p = cwd / rel_path
    try:
        if p.is_file():
            return p.stat().st_size / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def lfs_pattern_for(path: str) -> str:
    """
    Track the exact selected file by default.
    This avoids accidentally tracking unrelated files with the same extension.
    """
    return path.replace("\\", "/")


def validate_branch_name(name: str) -> str | None:
    name = name.strip()
    if not name:
        return "Branch name is required."
    if " " in name:
        return "Branch name cannot contain spaces. Use features/my-feature."
    if name.startswith("-") or name.endswith("/") or ".." in name or "//" in name:
        return "Branch name is not valid for Git."
    if name in {"main", "master", BASE_BRANCH}:
        return f"Use a feature branch, not {name!r}. Example: features/eeg-loader."
    return None


# ------------------------------ Background Jobs ----------------------------- #

@dataclass
class JobMessage:
    kind: str
    text: str


class Worker:
    def __init__(self, app: "GitWorkflowApp") -> None:
        self.app = app

    def start(self, label: str, func, *args, **kwargs) -> None:
        if self.app.busy:
            messagebox.showinfo("Busy", "A task is already running. Please wait.")
            return
        self.app.set_busy(True)
        self.app.log(f"\n--- {label} ---")

        def target() -> None:
            try:
                result = func(*args, **kwargs)
                self.app.queue.put(JobMessage("done", str(result or "Done.")))
            except Exception as exc:
                self.app.queue.put(JobMessage("error", str(exc)))
            finally:
                self.app.queue.put(JobMessage("idle", ""))

        threading.Thread(target=target, daemon=True).start()


# ----------------------------------- GUI ------------------------------------ #

class GitWorkflowApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Git Workflow Automation")
        self.root.geometry("1040x760")
        self.root.minsize(900, 650)

        self.queue: queue.Queue[JobMessage] = queue.Queue()
        self.worker = Worker(self)
        self.busy = False
        self.file_vars: dict[str, BooleanVar] = {}
        self.file_rows: list[tuple[str, str, BooleanVar]] = []

        self.repo_var = StringVar(value=DEFAULT_REPO_PATH)
        self.branch_var = StringVar(value="features/")
        self.commit_var = StringVar(value="")
        self.large_mb_var = StringVar(value=str(LARGE_FILE_MB))

        self._build_ui()
        self.root.after(100, self._poll_queue)

    @property
    def cwd(self) -> Path:
        return Path(self.repo_var.get()).expanduser().resolve()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        repo_frame = ttk.LabelFrame(outer, text="Repository")
        repo_frame.pack(fill="x", pady=(0, 10))
        ttk.Entry(repo_frame, textvariable=self.repo_var).pack(side="left", fill="x", expand=True, padx=8, pady=8)
        ttk.Button(repo_frame, text="Browse...", command=self.browse_repo).pack(side="left", padx=(0, 8))
        ttk.Button(repo_frame, text="Validate", command=self.validate_repo).pack(side="left", padx=(0, 8))

        actions = ttk.LabelFrame(outer, text="Workflow Actions")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="1. Get up-to-date dev", command=self.update_dev).grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        ttk.Button(actions, text="2. Open VS Code", command=self.open_vscode).grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        ttk.Button(actions, text="3. Refresh changed files", command=self.refresh_files).grid(row=0, column=2, padx=8, pady=8, sticky="ew")
        ttk.Button(actions, text="Select all", command=lambda: self.set_all_files(True)).grid(row=0, column=3, padx=8, pady=8, sticky="ew")
        ttk.Button(actions, text="Select none", command=lambda: self.set_all_files(False)).grid(row=0, column=4, padx=8, pady=8, sticky="ew")
        for i in range(5):
            actions.columnconfigure(i, weight=1)

        commit_frame = ttk.LabelFrame(outer, text="Commit and Push")
        commit_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(commit_frame, text="Branch name:").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(commit_frame, textvariable=self.branch_var).grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        ttk.Label(commit_frame, text="Commit message:").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(commit_frame, textvariable=self.commit_var).grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        ttk.Label(commit_frame, text="LFS threshold MB:").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        ttk.Entry(commit_frame, textvariable=self.large_mb_var, width=8).grid(row=0, column=3, padx=8, pady=8, sticky="w")
        ttk.Button(commit_frame, text="4. Commit + push + open review link", command=self.commit_push_review).grid(
            row=1, column=2, columnspan=2, padx=8, pady=8, sticky="ew"
        )
        commit_frame.columnconfigure(1, weight=1)

        main = ttk.PanedWindow(outer, orient="vertical")
        main.pack(fill="both", expand=True)

        files_container = ttk.LabelFrame(main, text="Changed Files - choose what to add")
        main.add(files_container, weight=3)

        self.files_canvas = None
        self.files_frame = ttk.Frame(files_container)
        self.files_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.tree = ttk.Treeview(self.files_frame, columns=("status", "size", "lfs"), show="tree headings", selectmode="none")
        self.tree.heading("#0", text="File")
        self.tree.heading("status", text="Git status")
        self.tree.heading("size", text="Size MB")
        self.tree.heading("lfs", text="LFS?")
        self.tree.column("#0", width=560)
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("size", width=90, anchor="e")
        self.tree.column("lfs", width=80, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Button-1>", self._toggle_tree_checkbox)

        scroll = ttk.Scrollbar(self.files_frame, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)

        log_frame = ttk.LabelFrame(main, text="Log")
        main.add(log_frame, weight=2)
        self.log_text = Text(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text.configure(state=DISABLED)

        hint = (
            "Tip: Run step 1 before starting work. Then edit in VS Code, refresh files, select files, "
            "enter features/something and a commit message, then push. Large selected files are tracked with Git LFS pointers."
        )
        ttk.Label(outer, text=hint, foreground="#555").pack(fill="x", pady=(8, 0))

    def browse_repo(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.repo_var.get() or DEFAULT_REPO_PATH)
        if chosen:
            self.repo_var.set(chosen)
            self.validate_repo()

    def validate_repo(self) -> None:
        try:
            cwd = self.cwd
            if not cwd.exists():
                raise GitError(f"Path does not exist: {cwd}")
            if not is_git_repo(cwd):
                raise GitError(f"Not a Git repository: {cwd}")
            root = repo_root(cwd)
            self.repo_var.set(str(root))
            current = git(["branch", "--show-current"], root, check=False).strip() or "detached HEAD"
            remote = git(["remote", "get-url", REMOTE_NAME], root, check=False).strip()
            self.log(f"Repository OK: {root}")
            self.log(f"Current branch: {current}")
            self.log(f"Remote {REMOTE_NAME}: {remote or '(not set)'}")
        except Exception as exc:
            messagebox.showerror("Repository error", str(exc))
            self.log(f"ERROR: {exc}")

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.root.config(cursor="watch" if busy else "")

    def log(self, text: str) -> None:
        self.log_text.configure(state=NORMAL)
        self.log_text.insert(END, text + "\n")
        self.log_text.see(END)
        self.log_text.configure(state=DISABLED)

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg.kind == "done":
                    self.log(msg.text)
                elif msg.kind == "error":
                    self.log("ERROR: " + msg.text)
                    messagebox.showerror("Error", msg.text)
                elif msg.kind == "idle":
                    self.set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def update_dev(self) -> None:
        self.worker.start("Get up-to-date files", self._update_dev)

    def _update_dev(self) -> str:
        cwd = self.cwd
        if not is_git_repo(cwd):
            raise GitError(f"Not a Git repository: {cwd}")
        root = repo_root(cwd)
        self.repo_var.set(str(root))
        out = []
        out.append(git(["fetch", REMOTE_NAME], root, check=False))
        out.append(git(["checkout", BASE_BRANCH], root))
        out.append(git(["pull", REMOTE_NAME, BASE_BRANCH], root))
        return "\n".join(x for x in out if x) or f"Checked out and pulled {BASE_BRANCH}."

    def open_vscode(self) -> None:
        cwd = self.cwd
        try:
            # Preferred: VS Code CLI. It can fail if `code` is not in PATH.
            subprocess.Popen(["code", str(cwd)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log(f"Opened VS Code: {cwd}")
        except Exception:
            try:
                os.startfile(str(cwd))  # type: ignore[attr-defined]
                self.log(f"Could not find `code`; opened folder instead: {cwd}")
            except Exception as exc:
                messagebox.showerror("Open VS Code failed", str(exc))
                self.log(f"ERROR: {exc}")

    def refresh_files(self) -> None:
        self.worker.start("Refresh changed files", self._refresh_files)

    def _refresh_files(self) -> str:
        cwd = self.cwd
        if not is_git_repo(cwd):
            raise GitError(f"Not a Git repository: {cwd}")
        root = repo_root(cwd)
        self.repo_var.set(str(root))
        changes = get_changed_files(root)
        self.queue.put(JobMessage("files", ""))
        # GUI update must be on main thread, so schedule it.
        self.root.after(0, lambda: self.populate_files(changes))
        if not changes:
            return "No changed files found."
        return f"Found {len(changes)} changed file(s). Click file rows to select/deselect."

    def populate_files(self, changes: list[tuple[str, str]]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.file_vars.clear()
        self.file_rows.clear()

        try:
            threshold = float(self.large_mb_var.get())
        except ValueError:
            threshold = LARGE_FILE_MB

        root = self.cwd
        for status, path in changes:
            size = file_size_mb(root, path)
            lfs = "yes" if size >= threshold and (root / path).is_file() else ""
            var = BooleanVar(value=True)
            self.file_vars[path] = var
            self.file_rows.append((status, path, var))
            prefix = "☑ "
            self.tree.insert("", "end", iid=path, text=prefix + path, values=(status, f"{size:.2f}", lfs))

    def _toggle_tree_checkbox(self, event) -> None:
        row = self.tree.identify_row(event.y)
        if not row or row not in self.file_vars:
            return
        var = self.file_vars[row]
        var.set(not var.get())
        current_text = self.tree.item(row, "text")
        clean = current_text[2:] if current_text.startswith(("☑ ", "☐ ")) else current_text
        self.tree.item(row, text=("☑ " if var.get() else "☐ ") + clean)

    def set_all_files(self, selected: bool) -> None:
        for path, var in self.file_vars.items():
            var.set(selected)
            text = self.tree.item(path, "text")
            clean = text[2:] if text.startswith(("☑ ", "☐ ")) else text
            self.tree.item(path, text=("☑ " if selected else "☐ ") + clean)

    def selected_files(self) -> list[str]:
        return [path for path, var in self.file_vars.items() if var.get()]

    def commit_push_review(self) -> None:
        branch = self.branch_var.get().strip()
        commit_message = self.commit_var.get().strip()
        branch_error = validate_branch_name(branch)
        if branch_error:
            messagebox.showerror("Invalid branch name", branch_error)
            return
        if not commit_message:
            messagebox.showerror("Missing commit message", "Please enter a commit message.")
            return
        files = self.selected_files()
        if not files:
            messagebox.showerror("No files selected", "Select at least one changed file to add.")
            return
        self.worker.start("Commit, push, and open review link", self._commit_push_review, branch, commit_message, files)

    def _commit_push_review(self, branch: str, commit_message: str, files: list[str]) -> str:
        root = repo_root(self.cwd)
        try:
            threshold = float(self.large_mb_var.get())
        except ValueError:
            threshold = LARGE_FILE_MB

        out: list[str] = []

        current = git(["branch", "--show-current"], root, check=False).strip()
        if current != BASE_BRANCH:
            out.append(git(["checkout", BASE_BRANCH], root))
        out.append(git(["pull", REMOTE_NAME, BASE_BRANCH], root))

        # Create/reset local feature branch from current dev if it does not exist.
        existing = git(["branch", "--list", branch], root, check=False).strip()
        if existing:
            out.append(git(["checkout", branch], root))
            out.append(git(["merge", BASE_BRANCH], root))
        else:
            out.append(git(["checkout", "-b", branch], root))

        large_files = [f for f in files if file_size_mb(root, f) >= threshold and (root / f).is_file()]
        if large_files:
            if not lfs_available(root):
                raise GitError(
                    "Some selected files are large, but Git LFS is not installed or not available.\n"
                    "Install it from https://git-lfs.com/ then run `git lfs install`, or lower/remove those files."
                )
            out.append(git(["lfs", "install"], root, check=False))
            for f in large_files:
                pattern = lfs_pattern_for(f)
                out.append(git(["lfs", "track", pattern], root))
            if (root / ".gitattributes").exists():
                files = sorted(set(files + [".gitattributes"]))

        # Add selected files only.
        for f in files:
            out.append(git(["add", "--", f], root))

        staged = git(["diff", "--cached", "--name-only"], root, check=False).strip()
        if not staged:
            raise GitError("Nothing is staged after adding selected files. Nothing to commit.")

        out.append("Staged files:\n" + staged)
        out.append(git(["commit", "-m", commit_message], root))
        out.append(git(["push", "-u", REMOTE_NAME, branch], root))

        url = compare_url(root, branch, BASE_BRANCH)
        if url:
            webbrowser.open(url)
            out.append(f"Opened code review link: {url}")
        else:
            out.append("Could not build GitHub compare URL. Check your origin remote.")

        return "\n".join(x for x in out if x)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = GitWorkflowApp()
    app.run()
