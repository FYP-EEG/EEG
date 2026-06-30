"""
Git Workflow Automation GUI - SAFE VERSION

This Windows-friendly Tkinter app automates a safer GitHub workflow:

Recommended flow:
1. Validate repo
2. Safely update dev
   - refuses to pull if you have uncommitted work
   - uses fast-forward only, so it will not create surprise merge commits
3. Create/switch to a feature branch from dev BEFORE editing
4. Open VS Code
5. Edit files
6. Refresh changed files and choose files with checkbox-style rows
7. Commit selected files only
8. Auto-track large selected files with Git LFS pointers
9. Push the feature branch
10. Open GitHub compare/PR link against dev

Build to .exe:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name GitWorkflowGUI git_workflow_gui.py

Requirements:
- Git in PATH
- VS Code `code` command in PATH, optional
- Git LFS, optional but required for large-file pointer support
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, Text, END, DISABLED, NORMAL, messagebox, filedialog
from tkinter import ttk


# ------------------------------ Configuration ------------------------------ #

BASE_BRANCH = "dev"
REMOTE_NAME = "origin"
LARGE_FILE_MB = 50
DEFAULT_REPO_PATH = r"D:\EEG" if os.name == "nt" and Path(r"D:\EEG").exists() else str(Path.cwd())

MAIN_BRANCH_EXPECTED_PATHS = [
    "EEG_signals/",
    "pygame/",  # adjust manually if main should only contain specific pygame files
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
        return git(["rev-parse", "--is-inside-work-tree"], path).strip().lower() == "true"
    except Exception:
        return False


def repo_root(path: Path) -> Path:
    return Path(git(["rev-parse", "--show-toplevel"], path)).resolve()


def current_branch(cwd: Path) -> str:
    return git(["branch", "--show-current"], cwd, check=False).strip()


def has_uncommitted_changes(cwd: Path) -> bool:
    return bool(git(["status", "--porcelain", "--untracked-files=all"], cwd, check=False).strip())


def working_tree_summary(cwd: Path) -> str:
    text = git(["status", "--short", "--branch", "--untracked-files=all"], cwd, check=False).strip()
    return text or "Working tree clean."


def local_branch_exists(cwd: Path, branch: str) -> bool:
    return bool(git(["branch", "--list", branch], cwd, check=False).strip())


def remote_branch_exists(cwd: Path, branch: str) -> bool:
    return bool(git(["ls-remote", "--heads", REMOTE_NAME, branch], cwd, check=False).strip())


def ensure_remote_base_exists(cwd: Path) -> None:
    if not remote_branch_exists(cwd, BASE_BRANCH):
        raise GitError(f"Remote branch {REMOTE_NAME}/{BASE_BRANCH} was not found.")


def ahead_behind(cwd: Path, left_ref: str, right_ref: str) -> tuple[int, int]:
    """
    Return commits unique to left_ref and commits unique to right_ref.
    Example: ahead_behind(root, 'dev', 'origin/dev') -> local ahead, local behind.
    """
    out = git(["rev-list", "--left-right", "--count", f"{left_ref}...{right_ref}"], cwd, check=False).strip()
    if not out:
        return 0, 0
    parts = out.split()
    if len(parts) != 2:
        return 0, 0
    return int(parts[0]), int(parts[1])


def normalize_remote_url(remote: str) -> str | None:
    """Convert common GitHub remote formats into https://github.com/owner/repo."""
    remote = remote.strip()
    if not remote:
        return None

    m = re.match(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?$", remote)
    if m:
        return f"https://github.com/{m.group('owner')}/{m.group('repo')}"

    m = re.match(r"ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?$", remote)
    if m:
        return f"https://github.com/{m.group('owner')}/{m.group('repo')}"

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
    safe_base = urllib.parse.quote(base, safe="/")
    safe_branch = urllib.parse.quote(branch, safe="/")
    return f"{base_url}/compare/{safe_base}...{safe_branch}?expand=1"


def lfs_available(cwd: Path) -> bool:
    try:
        git(["lfs", "version"], cwd)
        return True
    except Exception:
        return False


def parse_status_porcelain_z(raw: bytes) -> list[tuple[str, str]]:
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
            i += 1  # skip old path entry
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
    """Track the exact selected file by default."""
    return path.replace("\\", "/")


def validate_branch_name(name: str) -> str | None:
    name = name.strip()
    if not name:
        return "Branch name is required."
    if " " in name:
        return "Branch name cannot contain spaces. Use features/my-feature."
    if name.startswith("-") or name.endswith("/") or ".." in name or "//" in name:
        return "Branch name is not valid for Git."
    if name.endswith(".lock") or "@{" in name or "\\" in name:
        return "Branch name contains invalid Git characters."
    if name in {"main", "master", BASE_BRANCH}:
        return f"Use a feature branch, not {name!r}. Example: features/eeg-loader."
    if not name.startswith(("features/", "feature/", "fix/", "bugfix/", "hotfix/", "docs/", "chore/")):
        return "Use a descriptive branch prefix, for example: features/eeg-loader."
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
        self.root.title("Safe GitHub Workflow Automation")
        self.root.geometry("1080x780")
        self.root.minsize(920, 660)

        self.queue: queue.Queue[JobMessage] = queue.Queue()
        self.worker = Worker(self)
        self.busy = False
        self.file_vars: dict[str, BooleanVar] = {}

        self.repo_var = StringVar(value=DEFAULT_REPO_PATH)
        self.branch_var = StringVar(value="features/")
        self.commit_var = StringVar(value="")
        self.large_mb_var = StringVar(value=str(LARGE_FILE_MB))
        self.status_var = StringVar(value="Choose your repo and click Validate.")

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
        ttk.Label(outer, textvariable=self.status_var, foreground="#174ea6").pack(fill="x", pady=(0, 8))

        branch_frame = ttk.LabelFrame(outer, text="Safe Workflow")
        branch_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(branch_frame, text="Feature branch:").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(branch_frame, textvariable=self.branch_var).grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        ttk.Label(branch_frame, text="Commit message:").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(branch_frame, textvariable=self.commit_var).grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        ttk.Label(branch_frame, text="LFS threshold MB:").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        ttk.Entry(branch_frame, textvariable=self.large_mb_var, width=8).grid(row=0, column=3, padx=8, pady=8, sticky="w")
        branch_frame.columnconfigure(1, weight=1)

        actions = ttk.LabelFrame(outer, text="Actions")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="1. Safely update dev", command=self.update_dev).grid(row=0, column=0, padx=6, pady=8, sticky="ew")
        ttk.Button(actions, text="2. Create/switch feature branch", command=self.prepare_feature_branch).grid(row=0, column=1, padx=6, pady=8, sticky="ew")
        ttk.Button(actions, text="3. Open VS Code", command=self.open_vscode).grid(row=0, column=2, padx=6, pady=8, sticky="ew")
        ttk.Button(actions, text="4. Refresh changed files", command=self.refresh_files).grid(row=0, column=3, padx=6, pady=8, sticky="ew")
        ttk.Button(actions, text="5. Commit + push + review", command=self.commit_push_review).grid(row=0, column=4, padx=6, pady=8, sticky="ew")
        ttk.Button(actions, text="Select all", command=lambda: self.set_all_files(True)).grid(row=1, column=3, padx=6, pady=(0, 8), sticky="ew")
        ttk.Button(actions, text="Select none", command=lambda: self.set_all_files(False)).grid(row=1, column=4, padx=6, pady=(0, 8), sticky="ew")
        for i in range(5):
            actions.columnconfigure(i, weight=1)

        main = ttk.PanedWindow(outer, orient="vertical")
        main.pack(fill="both", expand=True)

        files_container = ttk.LabelFrame(main, text="Changed Files - click rows to select/deselect")
        main.add(files_container, weight=3)

        self.tree = ttk.Treeview(files_container, columns=("status", "size", "lfs"), show="tree headings", selectmode="none")
        self.tree.heading("#0", text="File")
        self.tree.heading("status", text="Git status")
        self.tree.heading("size", text="Size MB")
        self.tree.heading("lfs", text="LFS?")
        self.tree.column("#0", width=620)
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("size", width=90, anchor="e")
        self.tree.column("lfs", width=80, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self.tree.bind("<Button-1>", self._toggle_tree_checkbox)

        scroll = ttk.Scrollbar(files_container, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.tree.configure(yscrollcommand=scroll.set)

        log_frame = ttk.LabelFrame(main, text="Log")
        main.add(log_frame, weight=2)
        self.log_text = Text(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text.configure(state=DISABLED)

        hint = (
            "Safe rule: update dev first, create/switch to a feature branch before editing, then commit and push. "
            "The app refuses risky pulls when uncommitted work exists."
        )
        ttk.Label(outer, text=hint, foreground="#555").pack(fill="x", pady=(8, 0))

    # ------------------------------ UI Helpers ------------------------------ #

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def log(self, text: str) -> None:
        self.log_text.configure(state=NORMAL)
        self.log_text.insert(END, text + "\n")
        self.log_text.see(END)
        self.log_text.configure(state=DISABLED)

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.root.config(cursor="watch" if busy else "")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg.kind == "done":
                    self.log(msg.text)
                    self.validate_repo(silent=True)
                elif msg.kind == "error":
                    self.log("ERROR: " + msg.text)
                    messagebox.showerror("Error", msg.text)
                    self.validate_repo(silent=True)
                elif msg.kind == "idle":
                    self.set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def browse_repo(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.repo_var.get() or DEFAULT_REPO_PATH)
        if chosen:
            self.repo_var.set(chosen)
            self.validate_repo()

    def validate_repo(self, silent: bool = False) -> None:
        try:
            cwd = self.cwd
            if not cwd.exists():
                raise GitError(f"Path does not exist: {cwd}")
            if not is_git_repo(cwd):
                raise GitError(f"Not a Git repository: {cwd}")
            root = repo_root(cwd)
            self.repo_var.set(str(root))
            branch = current_branch(root) or "detached HEAD"
            remote = git(["remote", "get-url", REMOTE_NAME], root, check=False).strip()
            short = git(["status", "-sb", "--untracked-files=all"], root, check=False).strip()
            self.set_status(f"Repo: {root} | Branch: {branch} | {short.splitlines()[0] if short else 'clean'}")
            if not silent:
                self.log(f"Repository OK: {root}")
                self.log(f"Current branch: {branch}")
                self.log(f"Remote {REMOTE_NAME}: {remote or '(not set)'}")
                self.log(working_tree_summary(root))
        except Exception as exc:
            self.set_status("Repository problem. See log.")
            if not silent:
                messagebox.showerror("Repository error", str(exc))
            self.log(f"ERROR: {exc}")

    # ------------------------------ Main Actions ---------------------------- #

    def update_dev(self) -> None:
        self.worker.start("Safely update dev", self._update_dev)

    def _update_dev(self) -> str:
        root = self._validated_root()
        out: list[str] = []

        if has_uncommitted_changes(root):
            raise GitError(
                "Cannot update dev safely because you have uncommitted files.\n\n"
                "Commit them on a feature branch, stash them, or discard them first.\n\n"
                f"Current status:\n{working_tree_summary(root)}"
            )

        out.append(git(["fetch", REMOTE_NAME], root, check=False))
        ensure_remote_base_exists(root)

        # If local dev does not exist, create it tracking origin/dev.
        if not local_branch_exists(root, BASE_BRANCH):
            out.append(git(["checkout", "-b", BASE_BRANCH, f"{REMOTE_NAME}/{BASE_BRANCH}"], root))
            return "\n".join(x for x in out if x) or f"Created local {BASE_BRANCH} from {REMOTE_NAME}/{BASE_BRANCH}."

        out.append(git(["checkout", BASE_BRANCH], root))
        ahead, behind = ahead_behind(root, BASE_BRANCH, f"{REMOTE_NAME}/{BASE_BRANCH}")

        if ahead and behind:
            raise GitError(
                f"Local {BASE_BRANCH} and {REMOTE_NAME}/{BASE_BRANCH} have diverged.\n"
                f"Local is ahead by {ahead} and behind by {behind}.\n\n"
                "This requires manual review. Do not auto-pull."
            )
        if ahead and not behind:
            raise GitError(
                f"Local {BASE_BRANCH} is ahead of {REMOTE_NAME}/{BASE_BRANCH} by {ahead} commit(s).\n\n"
                "Safer options:\n"
                "1. Push those commits if they belong on dev, or\n"
                "2. Move them to a feature branch, then reset dev to origin/dev."
            )
        if behind:
            out.append(git(["pull", "--ff-only", REMOTE_NAME, BASE_BRANCH], root))
        else:
            out.append(f"{BASE_BRANCH} is already up to date with {REMOTE_NAME}/{BASE_BRANCH}.")

        return "\n".join(x for x in out if x)

    def prepare_feature_branch(self) -> None:
        branch = self.branch_var.get().strip()
        branch_error = validate_branch_name(branch)
        if branch_error:
            messagebox.showerror("Invalid branch name", branch_error)
            return
        self.worker.start("Create/switch feature branch", self._prepare_feature_branch, branch)

    def _prepare_feature_branch(self, branch: str) -> str:
        root = self._validated_root()
        out: list[str] = []

        if has_uncommitted_changes(root):
            raise GitError(
                "Create/switch feature branch is intentionally blocked because you have uncommitted files.\n\n"
                "If this is new work on dev, you may use the Commit button to create the branch while preserving changes, "
                "or manually run: git checkout -b features/your-branch\n\n"
                f"Current status:\n{working_tree_summary(root)}"
            )

        out.append(git(["fetch", REMOTE_NAME], root, check=False))
        ensure_remote_base_exists(root)

        # Update dev safely first.
        if local_branch_exists(root, BASE_BRANCH):
            out.append(git(["checkout", BASE_BRANCH], root))
            ahead, behind = ahead_behind(root, BASE_BRANCH, f"{REMOTE_NAME}/{BASE_BRANCH}")
            if ahead or behind:
                if ahead:
                    raise GitError(f"Local {BASE_BRANCH} is ahead of origin. Resolve that before creating a new branch.")
                out.append(git(["pull", "--ff-only", REMOTE_NAME, BASE_BRANCH], root))
        else:
            out.append(git(["checkout", "-b", BASE_BRANCH, f"{REMOTE_NAME}/{BASE_BRANCH}"], root))

        if local_branch_exists(root, branch):
            out.append(git(["checkout", branch], root))
            # Safe because tree is clean.
            out.append(git(["merge", "--no-edit", f"{REMOTE_NAME}/{BASE_BRANCH}"], root))
        elif remote_branch_exists(root, branch):
            out.append(git(["checkout", "-b", branch, f"{REMOTE_NAME}/{branch}"], root))
            out.append(git(["merge", "--no-edit", f"{REMOTE_NAME}/{BASE_BRANCH}"], root))
        else:
            out.append(git(["checkout", "-b", branch, f"{REMOTE_NAME}/{BASE_BRANCH}"], root))

        return "\n".join(x for x in out if x) + f"\nReady to edit on branch: {branch}"

    def open_vscode(self) -> None:
        cwd = self.cwd
        try:
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
        root = self._validated_root()
        changes = get_changed_files(root)
        self.root.after(0, lambda: self.populate_files(changes))
        if not changes:
            return "No changed files found."
        return f"Found {len(changes)} changed file(s). Click file rows to select/deselect."

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
            messagebox.showerror("No files selected", "Select at least one changed file to add. Click Refresh changed files first.")
            return
        self.worker.start("Commit, push, and open review link", self._commit_push_review, branch, commit_message, files)

    def _commit_push_review(self, branch: str, commit_message: str, files: list[str]) -> str:
        root = self._validated_root()
        out: list[str] = []
        try:
            threshold = float(self.large_mb_var.get())
        except ValueError:
            threshold = LARGE_FILE_MB

        out.append(git(["fetch", REMOTE_NAME], root, check=False))
        ensure_remote_base_exists(root)

        current = current_branch(root)
        if not current:
            raise GitError("Detached HEAD detected. Please checkout a branch first.")

        # Safe branch handling:
        # - If already on target feature branch: stay there.
        # - If on dev with uncommitted work and target branch does not exist: create branch, preserving work.
        # - If on any other branch with uncommitted work: refuse to switch.
        if current != branch:
            dirty = has_uncommitted_changes(root)
            if dirty:
                if current == BASE_BRANCH and not local_branch_exists(root, branch) and not remote_branch_exists(root, branch):
                    out.append(git(["checkout", "-b", branch], root))
                else:
                    raise GitError(
                        f"You are on branch {current!r} with uncommitted files.\n"
                        f"The app will not switch to {branch!r} because that could mix work.\n\n"
                        "Commit/stash/discard first, or switch/create the branch manually.\n\n"
                        f"Current status:\n{working_tree_summary(root)}"
                    )
            else:
                if local_branch_exists(root, branch):
                    out.append(git(["checkout", branch], root))
                elif remote_branch_exists(root, branch):
                    out.append(git(["checkout", "-b", branch, f"{REMOTE_NAME}/{branch}"], root))
                else:
                    # Create feature branch from latest origin/dev, not stale local dev.
                    out.append(git(["checkout", "-b", branch, f"{REMOTE_NAME}/{BASE_BRANCH}"], root))

        # Warn if feature branch is behind dev. Do not auto-merge when uncommitted changes exist.
        ahead, behind = ahead_behind(root, "HEAD", f"{REMOTE_NAME}/{BASE_BRANCH}")
        if behind:
            out.append(
                f"WARNING: This branch is behind {REMOTE_NAME}/{BASE_BRANCH} by {behind} commit(s). "
                "Commit will continue, but you may need to merge dev later if GitHub reports conflicts."
            )

        # Git LFS for large selected files.
        large_files = [f for f in files if file_size_mb(root, f) >= threshold and (root / f).is_file()]
        if large_files:
            if not lfs_available(root):
                raise GitError(
                    "Some selected files are large, but Git LFS is not installed or not available.\n"
                    "Install it from https://git-lfs.com/ then run `git lfs install`, or deselect those files."
                )
            out.append(git(["lfs", "install"], root, check=False))
            for f in large_files:
                out.append(git(["lfs", "track", lfs_pattern_for(f)], root))
            if (root / ".gitattributes").exists():
                files = sorted(set(files + [".gitattributes"]))

        # Stage only selected files.
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

    # ------------------------------ File List ------------------------------- #

    def populate_files(self, changes: list[tuple[str, str]]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.file_vars.clear()

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
            self.tree.insert("", "end", iid=path, text="☑ " + path, values=(status, f"{size:.2f}", lfs))

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

    def _validated_root(self) -> Path:
        cwd = self.cwd
        if not cwd.exists():
            raise GitError(f"Path does not exist: {cwd}")
        if not is_git_repo(cwd):
            raise GitError(f"Not a Git repository: {cwd}")
        root = repo_root(cwd)
        self.root.after(0, lambda: self.repo_var.set(str(root)))
        return root

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    GitWorkflowApp().run()
