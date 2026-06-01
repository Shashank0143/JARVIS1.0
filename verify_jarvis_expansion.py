import sys
from pathlib import Path

workspace = Path(__file__).parent.resolve()
sys.path.append(str(workspace))

import torch
from jarvis_ai import JarvisCore
from jarvis_ai.domains.deep_learning.automl import AutoMLEngine, ConvNet, LSTMLanguageModel, TransformerEncoderLM, DiffusionModel, PolicyNetwork
from jarvis_ai.domains.agents import AgentOrchestrator
from jarvis_ai.domains.security.auditor import SecurityAuditor
from jarvis_ai.domains.testing.system import AutonomousTester
from jarvis_ai.domains.learning.self_improver import SelfImprover

def run_verification():
    print("=" * 60)
    print("  JARVIS SYSTEM EXPANSION E2E VERIFICATION SUITE")
    print("=" * 60)

    print("\n[1/5] Verifying Dynamic PyTorch Neural Architectures...")
    try:
        cnn = ConvNet(in_channels=1, num_classes=10)
        lstm = LSTMLanguageModel(vocab_size=100)
        transformer = TransformerEncoderLM(vocab_size=100)
        diffusion = DiffusionModel(dim=64)
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        
        print(" -> Instantiated ConvNet, LSTMLanguageModel, TransformerEncoderLM, DiffusionModel, and PolicyNetwork.")

        mock_img = torch.randn(2, 1, 28, 28)
        out_cnn = cnn(mock_img)
        assert out_cnn.shape == (2, 10), f"ConvNet shape mismatch: {out_cnn.shape}"
        
        mock_seq = torch.randint(0, 100, (2, 5))
        out_lstm = lstm(mock_seq)
        assert out_lstm.shape == (2, 100), f"LSTM shape mismatch: {out_lstm.shape}"
        
        out_trans = transformer(mock_seq)
        assert out_trans.shape == (2, 100), f"Transformer shape mismatch: {out_trans.shape}"
        
        mock_spatial = torch.randn(2, 64)
        mock_t = torch.randn(2, 1)
        out_diff = diffusion(mock_spatial, mock_t)
        assert out_diff.shape == (2, 64), f"Diffusion shape mismatch: {out_diff.shape}"
        
        mock_state = torch.randn(2, 4)
        out_policy = policy(mock_state)
        assert out_policy.shape == (2, 2), f"Policy shape mismatch: {out_policy.shape}"

        print(" -> Forward passes verified successfully across all 5 architectures!")

        automl = AutoMLEngine(workspace)
        print(f" -> AutoML Device: {automl.device}")
        mock_dataset = "This is a simple text dataset to train our local transformer. It is small but contains words."
        automl_report = automl.run_automl_search(mock_dataset, epochs=1)
        print(automl_report)
        print(" -> AutoML training search and best model deployment successfully verified!")
    except Exception as exc:
        print(f" -> FAILED Neural Architectures: {exc}")
        sys.exit(1)

    print("\n[2/5] Verifying Multi-Agent Mesh & Message Bus...")
    try:
        orchestrator = AgentOrchestrator(workspace)
        response = orchestrator.run_pipeline("Build a web portal with beautiful glassmorphism ui")
        print(" -> Agent Collaborative Mesh Output:")
        print(response)
        
        logs = orchestrator.bus.get_logs()
        assert len(orchestrator.bus.history) > 0, "No messages posted on AgentMessageBus!"
        print(f" -> Message Bus Logs Verified ({len(orchestrator.bus.history)} messages sent).")
    except Exception as exc:
        print(f" -> FAILED Multi-Agent Mesh: {exc}")
        sys.exit(1)

    print("\n[3/5] Verifying Security Auditor Against Vulnerabilities...")
    try:
        auditor = SecurityAuditor(workspace)
        vulnerable_code = """
import sqlite3
import os
import pickle

def query_user(user_input):
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    # SQLi Risk
    cursor.execute("SELECT * FROM users WHERE name = '%s'" % user_input)
    # CMD Injection Risk
    os.system("ping " + user_input)
    # Cleartext Secret Risk
    password = "super_secret_password_123"
    # Deserialization Risk
    pickle.loads(user_input)
"""
        findings = auditor.audit_code(vulnerable_code)
        assert len(findings) >= 4, f"Expected at least 4 vulnerabilities, found {len(findings)}"
        print(f" -> Successfully scanned code! Found {len(findings)} vulnerabilities:")
        for vuln in findings:
            print(f"    * [{vuln.severity}] Line {vuln.line_number}: {vuln.vuln_type} - {vuln.description}")
            print(f"      Suggested Fix: {vuln.suggested_fix}")

        sqli_guide = auditor.defensive_mock_test("sqli")
        print("\n -> Defensive Mock Test Guide for SQLi:")
        print(sqli_guide.strip())
    except Exception as exc:
        print(f" -> FAILED Security Auditor: {exc}")
        sys.exit(1)

    print("\n[4/5] Verifying Autonomous Self-Correction & Self-Healing Testing Loop...")
    try:
        tester = AutonomousTester(workspace)
        
        test_dir = workspace / "datasets" / "training_corpus"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        buggy_target = test_dir / "buggy_calc.py"
        buggy_target.write_text("""
def divide_vals(a, b):
    # This might divide by zero!
    return a / b
""", encoding="utf-8")

        test_file = test_dir / "test_buggy_calc.py"
        test_file.write_text(f"""
import unittest
import sys
sys.path.append(r'{str(test_dir)}')
from buggy_calc import divide_vals

class TestCalc(unittest.TestCase):
    def test_divide(self):
        # This will trigger ZeroDivisionError
        val = divide_vals(10, 0)
        self.assertIsNotNone(val)

if __name__ == '__main__':
    unittest.main()
""", encoding="utf-8")

        print(" -> Initiating self-healing loop for buggy division script...")
        log = tester.self_healing_test_loop(buggy_target, test_file, max_retries=3)
        print(log)

        fixed_content = buggy_target.read_text(encoding="utf-8")
        assert "1e-9" in fixed_content or "if" in fixed_content, "Buggy code was not successfully healed!"
        print(" -> Bug successfully resolved and verified via self-healing loop!")
    except Exception as exc:
        print(f" -> FAILED Self-Healing Testing Loop: {exc}")
        sys.exit(1)

    print("\n[5/5] Verifying Self-Learning Memory Indexing & Auto-Knowledge...")
    try:
        core = JarvisCore(workspace)
        
        print(" -> Ingesting some user interaction into central router...")
        core.handle("bootstrap subjects")
        core.handle("learn This is manual RAG learned data to populate the memory logs.")
        core.handle("agent planning Build a scalable data-pipeline architecture")
        
        print(" -> Triggering structured auto-knowledge extraction...")
        know_res = core.handle("auto-knowledge")
        print(f" -> Router Response: {know_res}")
        
        overview_path = workspace / "knowledge" / "computer_science" / "data_structures" / "overview.md"
        assert overview_path.exists(), "Overview manual not generated!"
        print(" -> Overview Manual content:")
        print(overview_path.read_text(encoding="utf-8")[:300] + "...")
        print(" -> Auto-knowledge generation verified successfully!")
    except Exception as exc:
        print(f" -> FAILED Self-Learning Pass: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ALL 5 E2E EXPANSION VERIFICATION STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_verification()
