from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Vulnerability:
    vuln_type: str
    description: str
    severity: str
    line_number: int
    matched_text: str
    suggested_fix: str


class SecurityAuditor:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()

        # Define high-fidelity static analysis rules
        self.rules = [
            {
                "id": "SEC-SQLI",
                "type": "SQL Injection Risk",
                "pattern": re.compile(
                    r"execute\s*\(\s*['\"].*?%s.*?['\"]\s*%.*?\)|execute\s*\(\s*f['\"].*?\{.*?\}.*?['\"]\s*\)",
                    re.IGNORECASE,
                ),
                "severity": "CRITICAL",
                "description": "Raw user input concatenated or interpolated into a SQL statement can lead to SQL Injection.",
                "suggested_fix": "Use parameterized queries or prepare statements instead of raw string interpolation.",
            },
            {
                "id": "SEC-CMDI",
                "type": "Command Injection Risk",
                "pattern": re.compile(
                    r"subprocess\.(?:Popen|run|call|check_output)\s*\([^)]*shell\s*=\s*True[^)]*\)|os\.system\s*\(",
                    re.IGNORECASE,
                ),
                "severity": "HIGH",
                "description": "Executing system commands with 'shell=True' or using 'os.system' with raw string inputs allows execution of arbitrary commands.",
                "suggested_fix": "Pass command arguments as a list of strings, set shell=False, or use specialized APIs.",
            },
            {
                "id": "SEC-SECRET",
                "type": "Cleartext Secret Exposure",
                "pattern": re.compile(
                    r"(?:api_key|password|secret|token|private_key)\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]",
                    re.IGNORECASE,
                ),
                "severity": "HIGH",
                "description": "Hardcoded secrets or private keys exposed directly in source files are vulnerable to leaks.",
                "suggested_fix": "Load secrets dynamically from secure environment variables or a local configuration file.",
            },
            {
                "id": "SEC-XSS",
                "type": "Cross-Site Scripting (XSS)",
                "pattern": re.compile(
                    r"dangerouslySetInnerHTML\s*=|innerHTML\s*=|document\.write\s*\(",
                    re.IGNORECASE,
                ),
                "severity": "MEDIUM",
                "description": "Direct insertion of unescaped HTML content creates standard cross-site scripting vulnerabilities.",
                "suggested_fix": "Sanitize and escape all input, or use native React rendering/textContent to prevent script injection.",
            },
            {
                "id": "SEC-DESER",
                "type": "Insecure Deserialization",
                "pattern": re.compile(
                    r"pickle\.loads\s*\(|yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader",
                    re.IGNORECASE,
                ),
                "severity": "HIGH",
                "description": "Deserializing untrusted data with pickle or raw yaml loading allows executing arbitrary Python payloads.",
                "suggested_fix": "Use safe JSON deserialization or yaml.safe_load() instead.",
            },
        ]

    def audit_code(self, code: str) -> list[Vulnerability]:
        findings: list[Vulnerability] = []
        lines = code.splitlines()

        for rule in self.rules:
            for match in rule["pattern"].finditer(code):
                # Calculate corresponding line number from index
                match_start = match.start()
                line_no = code[:match_start].count("\n") + 1
                matched_line = lines[line_no - 1].strip() if line_no <= len(lines) else ""

                findings.append(
                    Vulnerability(
                        vuln_type=rule["type"],
                        description=rule["description"],
                        severity=rule["severity"],
                        line_number=line_no,
                        matched_text=matched_line,
                        suggested_fix=rule["suggested_fix"],
                    )
                )

        return findings

    def audit_file(self, filepath: Path | str) -> list[Vulnerability]:
        path = Path(filepath).resolve()
        if not path.is_file():
            return []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return self.audit_code(content)
        except Exception:
            return []

    def audit_project(self, target_path: Path | str | None = None) -> dict[str, list[Vulnerability]]:
        scan_dir = Path(target_path or self.workspace).resolve()
        results: dict[str, list[Vulnerability]] = {}
        allowed_extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".rs", ".go"}

        for file_path in scan_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in allowed_extensions:
                continue
            if any(part in file_path.parts for part in {".venv", "__pycache__", ".git", "data"}):
                continue

            findings = self.audit_file(file_path)
            if findings:
                rel = file_path.relative_to(scan_dir)
                results[str(rel)] = findings

        return results

    def defensive_mock_test(self, vuln_type_key: str) -> str:
        """Educational sandboxed defense demonstration."""
        demo_guides = {
            "sqli": """
[SQL Injection Mock Defense Guide]
Legal Environment: Local SQLite Sandbox.

VULNERABLE PATTERN:
  cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

SECURE MITIGATION:
  cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
            """,
            "cmdi": """
[Command Injection Mock Defense Guide]
Legal Environment: Controlled Subprocess Sandbox.

VULNERABLE PATTERN:
  os.system(f"ping {user_host}")  # if user_host = "127.0.0.1; rm -rf /"

SECURE MITIGATION:
  import subprocess
  subprocess.run(["ping", "-c", "4", user_host], shell=False)
            """,
            "secret": """
[Secrets Protection Mock Defense Guide]
Legal Environment: Environment Decoupling.

VULNERABLE PATTERN:
  API_KEY = "sk-live-1234abcd5678"

SECURE MITIGATION:
  import os
  API_KEY = os.getenv("JARVIS_API_KEY")
            """,
        }
        return demo_guides.get(
            vuln_type_key.lower().strip(),
            "Unknown or unsupported vulnerability key. Available: sqli, cmdi, secret.",
        )
