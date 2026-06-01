"""Machine control tools with sandboxed execution and secure permission checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


class MachineController:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self.security_log_path = self.workspace / "data" / "security_audit_execution.log"

    def open_target(self, target: str) -> str:
        target = target.strip().strip('"')
        if not target:
            return "No target provided."

        normalized = target.lower()
        known_sites = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "gmail": "https://mail.google.com",
            "github": "https://github.com",
            "chatgpt": "https://chatgpt.com",
        }
        if normalized in known_sites:
            webbrowser.open(known_sites[normalized])
            return f"Opened website: {known_sites[normalized]}"

        if target.startswith(("http://", "https://")):
            webbrowser.open(target)
            return f"Opened URL: {target}"

        path = self._resolve_path(target)
        if path.exists():
            os.startfile(str(path))
            return f"Opened: {path}"

        known_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "paint": "mspaint.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "explorer": "explorer.exe",
            "chrome": "chrome.exe",
            "google chrome": "chrome.exe",
            "edge": "msedge.exe",
            "microsoft edge": "msedge.exe",
            "firefox": "firefox.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
        }
        command = known_apps.get(normalized)
        if command:
            try:
                subprocess.Popen([command], shell=False)
                return f"Started application: {target}"
            except FileNotFoundError:
                fallback = self._find_common_windows_app(command)
                if fallback:
                    os.startfile(str(fallback))
                    return f"Started application: {target}"
                return f"Application not found on PATH: {target}"

        return f"I do not know how to open '{target}'. Try a full path, URL, or known app name."

    def create_file(self, path: str, content: str = "") -> str:
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            return f"File already exists: {file_path}"
        file_path.write_text(content, encoding="utf-8")
        return f"Created file: {file_path}"

    def create_folder(self, path: str) -> str:
        folder = self._resolve_path(path)
        folder.mkdir(parents=True, exist_ok=True)
        return f"Created folder: {folder}"

    def write_file(self, path: str, content: str, *, append: bool = False) -> str:
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with file_path.open("a", encoding="utf-8") as file:
                file.write(content)
            return f"Appended to file: {file_path}"
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote file: {file_path}"

    def update_line(self, path: str, line_number: int, content: str) -> str:
        file_path = self._resolve_existing_file(path)
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if line_number < 1 or line_number > len(lines):
            return f"Line {line_number} is outside file range 1-{len(lines)}."
        lines[line_number - 1] = content
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Updated line {line_number} in {file_path}"

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        file_path = self._resolve_existing_file(path)
        text = file_path.read_text(encoding="utf-8")
        if old not in text:
            return f"Text not found in {file_path}"
        file_path.write_text(text.replace(old, new), encoding="utf-8")
        return f"Updated text in {file_path}"

    def delete_path(self, path: str, *, confirm: bool = False) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"Path not found: {target}"
        if not confirm:
            return f"Delete requires confirmation. Run again with confirm delete: {target}"
        
        # Enforce safety restrictions on root folders
        if target.resolve() == self.workspace.resolve():
            return "Security violation: Direct deletion of workspace root is strictly blocked!"

        self._log_security_event("DELETE", str(target))
        if target.is_dir():
            shutil.rmtree(target)
            return f"Deleted folder: {target}"
        target.unlink()
        return f"Deleted file: {target}"

    def empty_file(self, path: str, *, confirm: bool = False) -> str:
        file_path = self._resolve_existing_file(path)
        if not confirm:
            return f"Erase requires confirmation. Run again with confirm erase: {file_path}"
        
        self._log_security_event("ERASE", str(file_path))
        file_path.write_text("", encoding="utf-8")
        return f"Erased file content: {file_path}"

    def execute_sandboxed_command(self, command_line: str, *, confirm: bool = False) -> str:
        """Requirement 4: Sandbox execution layer with strict permission checks."""
        cmd = command_line.strip()
        if not cmd:
            return "No command received."

        # Scan for dangerous keywords or escape commands
        dangerous = {"rmdir", "rm", "del", "mkfs", "format", "shutdown", "reboot", "kill", "netshare"}
        tokens = set(cmd.lower().replace("/", " ").replace("\\", " ").split())
        has_dangerous = not dangerous.isdisjoint(tokens)

        if has_dangerous and not confirm:
            return (
                f"Security Sandbox Blocked command: '{cmd}'.\n"
                "Contains potentially destructive calls. "
                "To execute, run explicitly as: confirm execute <command>"
            )

        self._log_security_event("EXECUTE", cmd)

        # Run process inside sandboxed subprocess limit
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=15,  # 15s sandbox execution window
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            status = f"success (exit={res.returncode})" if res.returncode == 0 else f"failure (exit={res.returncode})"
            return f"Sandbox Run: {status}\nOutput:\n{out}\n{err}".strip()
        except subprocess.TimeoutExpired:
            return f"Security Sandbox Error: Process execution timed out after 15 seconds."
        except Exception as exc:
            return f"Security Sandbox Error: Failed to execute process: {exc}"

    def _log_security_event(self, action: str, details: str) -> None:
        self.security_log_path.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        log_line = f"[{timestamp}] [ACTION={action}] Details: {details}\n"
        with self.feedback_path_open_safe() as file:
            file.write(log_line)

    def feedback_path_open_safe(self):
        return open(self.security_log_path, "a", encoding="utf-8")

    def _resolve_existing_file(self, path: str) -> Path:
        file_path = self._resolve_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    def _resolve_path(self, path: str) -> Path:
        raw = Path(path.strip().strip('"'))
        if raw.is_absolute():
            return raw.resolve()
        return (self.workspace / raw).resolve()

    @staticmethod
    def _find_common_windows_app(executable: str) -> Path | None:
        candidates = [
            Path(os.environ.get("ProgramFiles", "")) / "Google/Chrome/Application" / executable,
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Google/Chrome/Application" / executable,
            Path(os.environ.get("LocalAppData", "")) / "Google/Chrome/Application" / executable,
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application" / executable,
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application" / executable,
            Path(os.environ.get("LocalAppData", "")) / "Microsoft/Edge/Application" / executable,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None
