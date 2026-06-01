from __future__ import annotations

import re
from pathlib import Path

from .assistant import LocalCodingAssistant
from .archive import AnswerArchive
from .domains.conversation import ConversationAgent
from .domains.deep_learning import DeepLearningBrain
from .domains.deep_learning.automl import AutoMLEngine
from .domains.coder.templates import UniversalCoder
from .domains.education import SubjectManager
from .learning import InternetLearningEngine
from .machine import MachineController
from .memory import MemoryStore
from .router import IntentRouter
from .domains.agents import AgentOrchestrator
from .domains.learning.self_improver import SelfImprover


class JarvisCore:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self.memory = MemoryStore(self.workspace / "data")
        self.coding = LocalCodingAssistant(self.workspace / "data")
        self.archive = AnswerArchive(self.workspace / "data")
        self.learning = InternetLearningEngine(self.coding, self.workspace / "data")
        self.machine = MachineController(self.workspace)
        self.router = IntentRouter()
        self.conversation = ConversationAgent(self.memory, self.coding.rag, self.coding.model)
        self.subjects = SubjectManager(self.workspace / "knowledge", self.coding)
        self.deep_learning = DeepLearningBrain(self.coding)
        self.universal_coder = UniversalCoder(self.workspace)

        self.agents = AgentOrchestrator(self.workspace)
        self.self_improver = SelfImprover(self.workspace)
        self.automl = AutoMLEngine(self.workspace)
        
        self.vision = None
        self.voice = None

    def enable_voice(self) -> str:
        from .voice import VoiceInterface

        self.voice = VoiceInterface()
        return "Voice recognition enabled."

    def enable_vision(self, camera_index: int = 0) -> str:
        from .vision import VisionProcessor

        self.vision = VisionProcessor(camera_index)
        return "Camera processing enabled."

    def handle(self, command: str) -> str:
        command = command.strip()
        if not command:
            return "No command received."
        self.memory.remember("command", command)

        math_response = self._try_evaluate_math(command)
        if math_response is not None:
            evaluation = self.self_improver.evaluate_response(command, math_response)
            reward = 1.0 if evaluation["score"] >= 8.0 else -0.5 if evaluation["score"] < 5.0 else 0.2
            self.self_improver.record_feedback(command, "math", reward)
            self.memory.remember("response", math_response)
            self.archive.save_answer(command, math_response, source="jarvis-core")
            return math_response

        lowered = command.lower()
        intent = self.router.route(command)

        try:
            if intent.name == "agent":
                response = self._handle_agent(command)
            elif intent.name == "automl":
                response = self._handle_automl(command)
            elif intent.name == "security":
                response = self._handle_security(command)
            elif intent.name == "testing":
                response = self._handle_testing(command)
            elif intent.name == "self_improve":
                response = self._handle_self_improve(command)
            elif intent.name == "machine":
                response = self._handle_machine(command, lowered)
            elif intent.name == "vision":
                response = self.observe_camera()
            elif intent.name == "learn":
                response = self._handle_learn(command[len("learn ") :])
            elif intent.name == "education":
                response = self._handle_education(command)
            elif intent.name == "deep_learning":
                response = self._handle_deep_learning(command)
            elif intent.name == "coding":
                response = self._handle_coding(command)
            elif intent.name == "conversation":
                response = self.conversation.answer(command).answer
            else:
                response = self._answer_local_first(command)
        except Exception as exc:
            response = f"Command failed: {exc}"

        evaluation = self.self_improver.evaluate_response(command, response)
        reward = 1.0 if evaluation["score"] >= 8.0 else -0.5 if evaluation["score"] < 5.0 else 0.2
        self.self_improver.record_feedback(command, intent.name, reward)

        self.memory.remember("response", response)
        self.archive.save_answer(command, response, source="jarvis-core")
        return response

    def _handle_agent(self, command: str) -> str:
        prefix = "run agent " if command.lower().startswith("run agent ") else "agent "
        body = command[len(prefix) :].strip()
        
        words = body.split()
        if len(words) > 1 and words[0].lower() in self.agents.agents:
            agent_key = words[0].lower()
            agent_instruction = " ".join(words[1:])
            return self.agents.agents[agent_key].execute(agent_instruction, {})
            
        return self.agents.run_pipeline(body)

    def _handle_automl(self, command: str) -> str:
        body = command[len("automl ") :].strip()
        path = Path(body or "datasets/training_corpus")
        try:
            content = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else "Simulated dataset sequence."
        except Exception:
            content = "Simulated dataset sequence."
        return self.automl.run_automl_search(content)

    def _handle_security(self, command: str) -> str:
        prefix = "security audit " if command.lower().startswith("security audit ") else "security-audit "
        if not command.lower().startswith(prefix):
            prefix = "audit "
        body = command[len(prefix) :].strip()
        return self.agents.agents["security"].execute(body or ".", {})

    def _handle_testing(self, command: str) -> str:
        if command.lower().startswith("self-heal "):
            return self.agents.agents["testing"].execute(command, {})
        
        body = command[len("run tests ") :].strip()
        parts = body.split()
        if len(parts) >= 2:
            return self.agents.agents["testing"].execute(f"self-heal {parts[0]} {parts[1]}", {})
            
        if body:
            test_content = self.agents.agents["testing"].tester.generate_unit_tests(body)
            test_path = Path(body).parent / f"test_{Path(body).name}"
            test_path.write_text(test_content, encoding="utf-8")
            return f"Generated unittest suite for '{body}' at: {test_path}"
            
        return "Autonomous Testing expects: run tests <target_file> <test_file>"

    def _handle_self_improve(self, command: str) -> str:
        lowered = command.lower().strip()
        if "auto-knowledge" in lowered or "knowledge" in lowered:
            return self.self_improver.auto_generate_knowledge()
        recent = self.memory.recent(limit=2)
        if len(recent) >= 2:
            q = recent[0].get("content", "")
            r = recent[1].get("content", "")
            eval_metrics = self.self_improver.evaluate_response(q, r)
            return f"Self-Evaluation Score of last interaction: {eval_metrics['score']}/10.0 | Metrics: {eval_metrics['metrics']}"
        return "Self-Improvement loop online. Run 'auto-knowledge' to extract memory files."

    def _answer_local_first(self, command: str) -> str:
        subject_answer = self.subjects.answer_from_subject(command)
        if subject_answer:
            return subject_answer
        # Synthesize dynamically via LLM using our upgraded RAG & memory synthesis assistant
        return self.coding.answer(command, use_web=False)

    def _handle_coding(self, command: str) -> str:
        response = self.universal_coder.answer(command)
        if response:
            return response
        return self.coding.answer(command, use_web=False)

    def observe_camera(self) -> str:
        if self.vision is None:
            self.enable_vision()
        observation = self.vision.observe_once()
        explanation = observation.explain()
        self.learn(explanation)
        return explanation

    def learn(self, text: str) -> str:
        self.memory.remember("learned", text)
        chunks, tokens = self.coding.ingest_text(text, source="jarvis-memory", title="Jarvis memory")
        return f"Learned {tokens} tokens into {chunks} RAG chunks."

    def learn_from_internet(self, topic: str, *, pages: int = 5) -> str:
        stats = self.learning.learn_topic(topic, max_pages=pages)
        return f"Internet learning complete for '{topic}': {stats.summary()}"

    def learn_curriculum(self, *, pages_per_topic: int = 3) -> str:
        stats = self.learning.learn_default_curriculum(pages_per_topic=pages_per_topic)
        return f"Curriculum learning complete: {stats.summary()}"

    def learn_pytorch_tutorials(self, *, pages: int = 8) -> str:
        stats = self.learning.learn_pytorch_tutorials(max_pages=pages)
        return f"PyTorch tutorial learning complete: {stats.summary()}"

    def _handle_learn(self, payload: str) -> str:
        lowered = payload.lower().strip()
        if lowered in {"curriculum", "default curriculum", "from internet", "internet"}:
            return self.learn_curriculum()
        if lowered.startswith("internet "):
            return self.learn_from_internet(payload[len("internet ") :].strip())
        if lowered.startswith("from internet "):
            return self.learn_from_internet(payload[len("from internet ") :].strip())
        if lowered in {"pytorch", "pytorch tutorials", "official pytorch tutorials"}:
            return self.learn_pytorch_tutorials()
        return self.learn(payload)

    def _handle_machine(self, command: str, lowered: str) -> str:
        if lowered.startswith("execute ") or lowered.startswith("confirm execute "):
            confirm = lowered.startswith("confirm execute ")
            body = command[len("confirm execute ") :] if confirm else command[len("execute ") :]
            return self.machine.execute_sandboxed_command(body, confirm=confirm)
        if lowered.startswith("open "):
            return self.machine.open_target(command[5:])
        if lowered.startswith("create folder "):
            return self.machine.create_folder(command[len("create folder ") :])
        if lowered.startswith("create file "):
            return self.machine.create_file(command[len("create file ") :])
        if lowered.startswith("write "):
            return self._handle_write(command)
        if lowered.startswith("append "):
            return self._handle_write(command, append=True)
        if lowered.startswith("update line "):
            return self._handle_update_line(command)
        if lowered.startswith("replace "):
            return self._handle_replace(command)
        if lowered.startswith("delete ") or lowered.startswith("confirm delete "):
            confirm = lowered.startswith("confirm delete ")
            path = command[len("confirm delete ") :] if confirm else command[len("delete ") :]
            return self.machine.delete_path(path, confirm=confirm)
        if lowered.startswith("erase ") or lowered.startswith("confirm erase "):
            confirm = lowered.startswith("confirm erase ")
            path = command[len("confirm erase ") :] if confirm else command[len("erase ") :]
            return self.machine.empty_file(path, confirm=confirm)
        return "Machine command not understood."

    def _handle_education(self, command: str) -> str:
        lowered = command.lower().strip()
        if lowered == "bootstrap subjects":
            return self.subjects.bootstrap()
        match = re.match(r"subject\s+(.+)$", command, flags=re.IGNORECASE)
        if match:
            return self.subjects.create_subject(match.group(1))
        match = re.match(r"chapter\s+(.+?)\s+in\s+(.+)$", command, flags=re.IGNORECASE)
        if match:
            return self.subjects.create_chapter(match.group(2), match.group(1))
        match = re.match(
            r"note\s+(.+?)\s+in\s+(.+?)\s*/\s*(.+?)\s+with\s+(.+)",
            command,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return self.subjects.save_note(
                match.group(2),
                match.group(3),
                match.group(1),
                match.group(4),
            )
        answer = self.subjects.answer_from_subject(command, subject="mathematics")
        if answer:
            return answer
        return (
            "Education commands: `bootstrap subjects`, `subject <name>`, "
            "`chapter <chapter> in <subject>`, `note <title> in <subject>/<chapter> with <content>`."
        )

    def _handle_deep_learning(self, command: str) -> str:
        lowered = command.lower().strip()
        if lowered in {"neural status", "deep learning", "tensorflow", "pytorch"}:
            return self.deep_learning.status()
        if lowered.startswith("train tensorflow"):
            return self.deep_learning.train_tensorflow_demo()
        if lowered.startswith("train neural"):
            return self.deep_learning.train(
                self.workspace / "datasets" / "training_corpus",
                epochs=1,
                steps_per_epoch=20,
            )
        return self.deep_learning.status()

    def _handle_write(self, command: str, *, append: bool = False) -> str:
        prefix = "append " if append else "write "
        body = command[len(prefix) :]
        match = re.match(r"(.+?)\s+(?:with|as|content)\s+(.+)", body, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return f"Use: {prefix}<file> with <content>"
        return self.machine.write_file(match.group(1), match.group(2), append=append)

    def _handle_update_line(self, command: str) -> str:
        body = command[len("update line ") :]
        match = re.match(r"(\d+)\s+in\s+(.+?)\s+(?:with|to)\s+(.+)", body, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return "Use: update line <number> in <file> with <content>"
        return self.machine.update_line(match.group(2), int(match.group(1)), match.group(3))

    def _handle_replace(self, command: str) -> str:
        body = command[len("replace ") :]
        match = re.match(r"(.+?)\s+in\s+(.+?)\s+with\s+(.+)", body, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return "Use: replace <old text> in <file> with <new text>"
        return self.machine.replace_in_file(match.group(2), match.group(1), match.group(3))

    def _try_evaluate_math(self, command: str) -> str | None:
        text = command.lower().strip()
        text = re.sub(r"^(what is|calculate|please solve|solve|whats|what's)\s+", "", text)
        text = text.rstrip("?").strip()
        
        m = re.match(r"^add\s+([\d\.]+)\s+(?:and|to)\s+([\d\.]+)$", text)
        if m:
            try:
                n1, n2 = float(m.group(1)), float(m.group(2))
                val = n1 + n2
                return f"{n1:g} + {n2:g} = {val:g}"
            except ValueError:
                pass
                
        m = re.match(r"^subtract\s+([\d\.]+)\s+from\s+([\d\.]+)$", text)
        if m:
            try:
                n1, n2 = float(m.group(1)), float(m.group(2))
                val = n2 - n1
                return f"{n2:g} - {n1:g} = {val:g}"
            except ValueError:
                pass
                
        m = re.match(r"^multiply\s+([\d\.]+)\s+(?:and|by|times)\s+([\d\.]+)$", text)
        if m:
            try:
                n1, n2 = float(m.group(1)), float(m.group(2))
                val = n1 * n2
                return f"{n1:g} * {n2:g} = {val:g}"
            except ValueError:
                pass
                
        m = re.match(r"^divide\s+([\d\.]+)\s+by\s+([\d\.]+)$", text)
        if m:
            try:
                n1, n2 = float(m.group(1)), float(m.group(2))
                if n2 == 0:
                    return "Division by zero is undefined."
                val = n1 / n2
                return f"{n1:g} / {n2:g} = {val:g}"
            except ValueError:
                pass

        expr = text.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("divided by", "/")
        expr = re.sub(r"\s*([\+\-\*\/])\s*", r"\1", expr)
        if re.match(r"^[\d\.\+\-\*\/\(\)]+$", expr):
            try:
                val = eval(expr, {"__builtins__": None}, {})
                return f"{command.strip()} = {val:g}"
            except Exception:
                pass
        return None
