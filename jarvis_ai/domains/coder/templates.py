from __future__ import annotations

import re
from pathlib import Path


class UniversalCoder:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()

    def list_supported_languages(self) -> list[str]:
        return [
            "python",
            "javascript",
            "typescript",
            "java",
            "cpp",
            "rust",
            "go",
            "swift",
            "kotlin",
            "php",
            "csharp",
        ]

    def list_supported_templates(self) -> list[str]:
        return [
            "backend-fastapi",
            "backend-node",
            "frontend-nextjs",
            "frontend-react",
            "mobile-react-native",
            "mobile-flutter",
            "desktop-electron",
            "devops-pipeline",
            "database-schema",
            "game-engine-mini",
            "machine-learning-pipeline",
        ]

    def answer(self, prompt: str) -> str:
        lowered = prompt.lower().strip()
        if self._is_capability_question(lowered):
            return self._capabilities_answer()

        full_stack = self._full_stack_answer(lowered)
        if full_stack:
            return full_stack

        scaffold = self._detect_scaffold_request(lowered)
        if scaffold:
            template_name, language, target_dir_name = scaffold
            created = self.generate_project(template_name, language, target_dir_name)
            return (
                f"{created}\n\n"
                f"Next steps:\n"
                f"1. Open `{target_dir_name}`.\n"
                f"2. Install the dependencies listed in the generated files.\n"
                f"3. Run the project with the command shown in its README or package file."
            )

        snippet = self._generate_snippet(lowered)
        if snippet:
            title, code, notes = snippet
            return f"{title}\n\n```{notes['language']}\n{code.strip()}\n```\n\n{notes['explanation']}"

        return ""

    def _full_stack_answer(self, lowered: str) -> str:
        wants_frontend = any(word in lowered for word in {"frontend", "front end", "react", "html", "website"})
        wants_backend = any(word in lowered for word in {"backend", "back end", "api", "server", "fastapi"})
        wants_database = any(word in lowered for word in {"database", "sql", "schema", "db"})
        wants_full_stack = "full stack" in lowered or "fullstack" in lowered
        if not (wants_full_stack or (wants_frontend and wants_backend and wants_database)):
            return ""

        return """
Basic full-stack starter: frontend, backend, and database.

Frontend `index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Users App</title>
  </head>
  <body>
    <h1>Users</h1>
    <button onclick="loadUsers()">Load users</button>
    <ul id="users"></ul>

    <script>
      async function loadUsers() {
        const response = await fetch("http://127.0.0.1:8000/users");
        const users = await response.json();
        document.getElementById("users").innerHTML = users
          .map((user) => `<li>${user.name} - ${user.email}</li>`)
          .join("");
      }
    </script>
  </body>
</html>
```

Backend `main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

users = [
    {"id": 1, "name": "Roopesh", "email": "roopesh@example.com"},
]

@app.get("/users")
def list_users():
    return users
```

Database `schema.sql`:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);
```

Run backend: `pip install fastapi uvicorn` then `uvicorn main:app --reload`.
Open the frontend HTML file in a browser and click `Load users`.
""".strip()

    def _is_capability_question(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in {
                "how can you help",
                "what can you do",
                "what can you create",
                "help me code",
            }
        )

    def _capabilities_answer(self) -> str:
        languages = ", ".join(self.list_supported_languages())
        templates = ", ".join(self.list_supported_templates())
        return (
            "I can help with programming by generating starter code, explaining bugs, "
            "creating project boilerplate, and building basic frontend, backend, database, "
            "DevOps, and machine-learning examples.\n\n"
            f"Supported languages: {languages}.\n\n"
            f"Project templates: {templates}.\n\n"
            "Examples:\n"
            "- `write basic python code`\n"
            "- `create backend fastapi project in python called my_api`\n"
            "- `create frontend react app in javascript called my_site`\n"
            "- `write database sql schema for users`"
        )

    def _detect_scaffold_request(self, lowered: str) -> tuple[str, str, str] | None:
        if not any(word in lowered for word in {"create", "generate", "scaffold", "make", "build"}):
            return None
        if not any(word in lowered for word in {"project", "app", "api", "frontend", "backend", "database"}):
            return None

        language = self._detect_language(lowered)
        template_name = self._detect_template(lowered, language)
        if not template_name:
            return None

        name_match = re.search(r"\b(?:called|named|as|in folder)\s+([A-Za-z0-9_-]+)", lowered)
        target = name_match.group(1) if name_match else template_name.replace("-", "_")
        return template_name, language, target

    def _detect_language(self, lowered: str) -> str:
        language_aliases = {
            "python": "python",
            "py": "python",
            "javascript": "javascript",
            "js": "javascript",
            "typescript": "typescript",
            "ts": "typescript",
            "java": "java",
            "cpp": "cpp",
            "c++": "cpp",
            "rust": "rust",
            "go": "go",
            "golang": "go",
            "php": "php",
            "csharp": "csharp",
            "c#": "csharp",
            "swift": "swift",
            "kotlin": "kotlin",
        }
        for alias, language in language_aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return language
        if "react" in lowered or "frontend" in lowered:
            return "javascript"
        if "database" in lowered or "sql" in lowered:
            return "sql"
        return "python"

    def _detect_template(self, lowered: str, language: str) -> str | None:
        if "fastapi" in lowered or ("backend" in lowered and language == "python") or "api" in lowered:
            return "backend-fastapi" if language == "python" else "backend-node"
        if "node" in lowered or "express" in lowered:
            return "backend-node"
        if "next" in lowered:
            return "frontend-nextjs"
        if "react" in lowered or "frontend" in lowered or "website" in lowered:
            return "frontend-react"
        if "database" in lowered or "schema" in lowered or "sql" in lowered:
            return "database-schema"
        if "devops" in lowered or "docker" in lowered:
            return "devops-pipeline"
        if "machine learning" in lowered or "ml " in lowered:
            return "machine-learning-pipeline"
        if language in self.list_supported_languages():
            return language
        return None

    def _generate_snippet(self, lowered: str) -> tuple[str, str, dict[str, str]] | None:
        if "database" in lowered or "sql" in lowered or "schema" in lowered:
            return (
                "Basic SQL database schema",
                """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
""",
                {
                    "language": "sql",
                    "explanation": "This creates two related tables: users and posts. It is a clean starting point for SQLite or can be adapted for PostgreSQL/MySQL.",
                },
            )

        if "frontend" in lowered or "react" in lowered:
            return (
                "Basic React frontend component",
                """
import { useState } from "react";

export default function App() {
  const [name, setName] = useState("");

  return (
    <main className="app">
      <h1>Basic React App</h1>
      <input
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="Enter your name"
      />
      <p>Hello {name || "developer"}.</p>
    </main>
  );
}
""",
                {
                    "language": "jsx",
                    "explanation": "This is a small stateful frontend example. It reads input from the user and updates the page immediately.",
                },
            )

        if "backend" in lowered or "api" in lowered or "fastapi" in lowered:
            return (
                "Basic Python FastAPI backend",
                """
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Todo(BaseModel):
    title: str
    done: bool = False

todos: list[Todo] = []

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/todos")
def create_todo(todo: Todo):
    todos.append(todo)
    return todo

@app.get("/todos")
def list_todos():
    return todos
""",
                {
                    "language": "python",
                    "explanation": "Install `fastapi` and `uvicorn`, then run `uvicorn main:app --reload`. This gives you a working JSON API.",
                },
            )

        if "html" in lowered or "website" in lowered:
            return (
                "Basic HTML page",
                """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Basic Website</title>
    <style>
      body {
        font-family: Georgia, serif;
        margin: 0;
        padding: 40px;
        background: #f4f1ea;
        color: #1d1d1d;
      }
      button {
        padding: 10px 16px;
        border: 1px solid #1d1d1d;
        background: #ffffff;
        cursor: pointer;
      }
    </style>
  </head>
  <body>
    <h1>My Basic Website</h1>
    <p>This is a simple frontend page.</p>
    <button onclick="alert('Hello from JavaScript')">Click me</button>
  </body>
</html>
""",
                {
                    "language": "html",
                    "explanation": "This single file includes HTML, CSS, and JavaScript. Open it in a browser to run it.",
                },
            )

        if "javascript" in lowered or " js " in f" {lowered} ":
            return (
                "Basic JavaScript code",
                """
function add(a, b) {
  return a + b;
}

const result = add(1, 2);
console.log(`Result: ${result}`);
""",
                {
                    "language": "javascript",
                    "explanation": "This defines a function, calls it, and prints the result. Run it with Node.js or in a browser console.",
                },
            )

        if "python" in lowered or "code" in lowered or "program" in lowered:
            return (
                "Basic Python code",
                """
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    print(greet("Jarvis"))
    print("1 + 2 =", add(1, 2))
""",
                {
                    "language": "python",
                    "explanation": "This shows a function, type hints, a simple calculation, and the standard `if __name__ == \"__main__\"` entry point.",
                },
            )

        return None

    def generate_project(self, template_name: str, language: str, target_dir_name: str) -> str:
        lang = language.lower().strip()
        tmpl = template_name.lower().strip()
        target_dir = self.workspace / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)

        files = self.get_template_files(tmpl, lang)
        if not files:
            return f"Template '{tmpl}' in language '{lang}' is not supported or not found."
        if "README.md" not in files:
            files["README.md"] = self._project_readme(tmpl, lang)

        for filename, content in files.items():
            filepath = target_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content.strip(), encoding="utf-8")

        return f"Successfully generated '{tmpl}' boilerplate in '{lang}' inside {target_dir}."

    def _project_readme(self, template_name: str, language: str) -> str:
        commands = {
            "backend-fastapi": "pip install -r requirements.txt\nuvicorn main:app --reload",
            "backend-node": "npm install\nnpm start",
            "frontend-nextjs": "npm install\nnpm run dev",
            "frontend-react": "npm install\nnpm run dev",
            "database-schema": "Load `schema.sql` in your database client.",
            "devops-pipeline": "docker compose up --build",
            "machine-learning-pipeline": "python ml_pipeline.py",
        }
        command = commands.get(template_name, f"Run the generated {language} entry file.")
        return f"""
# {template_name}

Generated by Mini Jarvis.

## Run

```bash
{command}
```
"""

    def get_template_files(self, template_name: str, language: str) -> dict[str, str]:
        files: dict[str, str] = {}

        if template_name == "backend-fastapi" and language == "python":
            files["main.py"] = """
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="JARVIS Production API", version="1.0.0", description="Secure production API gateway.")

class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float

items_db: List[Item] = []

@app.get("/api/v1/items", response_model=List[Item])
async def get_items():
    return items_db

@app.post("/api/v1/items", status_code=status.HTTP_201_CREATED, response_model=Item)
async def create_item(item: Item):
    for existing in items_db:
        if existing.id == item.id:
            raise HTTPException(status_code=400, detail="Item already exists")
    items_db.append(item)
    return item
"""
            files["Dockerfile"] = """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            files["requirements.txt"] = "fastapi>=0.110.0\nuvicorn>=0.28.0\npydantic>=2.6.0"

        elif template_name == "backend-node" and language in ("javascript", "typescript"):
            ext = "ts" if language == "typescript" else "js"
            files[f"server.{ext}"] = """
const express = require('express');
const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json());

let users = [
    { id: 1, name: 'Jarvis Creator' }
];

app.get('/api/users', (req, res) => {
    res.json(users);
});

app.post('/api/users', (req, res) => {
    const user = req.body;
    if (!user.name) {
        return res.status(400).json({ error: 'Name is required' });
    }
    users.push(user);
    res.status(201).json(user);
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
"""
            files["package.json"] = """
{
  "name": "jarvis-node-backend",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.19.2"
  }
}
"""

        elif template_name == "frontend-nextjs" and language in ("typescript", "javascript"):
            ext = "tsx" if language == "typescript" else "jsx"
            files[f"app/page.{ext}"] = """
'use client';
import { useState } from 'react';

export default function Home() {
  const [active, setActive] = useState(false);

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'radial-gradient(circle, #1e3c72 0%, #2a5298 100%)',
      color: '#fff',
      fontFamily: 'system-ui, sans-serif'
    }}>
      <h1 style={{ fontSize: '3rem', margin: '0 0 1rem 0' }}>JARVIS AI Dashboard</h1>
      <p style={{ fontSize: '1.2rem', marginBottom: '2rem' }}>Offline High-Performance Interface</p>
      <button 
        onClick={() => setActive(!active)}
        style={{
          padding: '12px 24px',
          fontSize: '1rem',
          fontWeight: 'bold',
          border: 'none',
          borderRadius: '8px',
          background: active ? '#4caf50' : '#ff9800',
          color: '#fff',
          cursor: 'pointer',
          transition: 'all 0.3s ease'
        }}>
        {active ? 'SYSTEM ONLINE' : 'ACTIVATE JARVIS'}
      </button>
    </div>
  );
}
"""
            files["package.json"] = """
{
  "name": "jarvis-nextjs-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^14.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
"""

        elif template_name == "frontend-react" and language in ("typescript", "javascript"):
            ext = "tsx" if language == "typescript" else "jsx"
            main_ext = "tsx" if language == "typescript" else "jsx"
            files[f"src/App.{ext}"] = """
import { useState } from "react";
import "./styles.css";

export default function App() {
  const [task, setTask] = useState("");
  const [tasks, setTasks] = useState(["Plan the app", "Build the first screen"]);

  function addTask() {
    const nextTask = task.trim();
    if (!nextTask) {
      return;
    }
    setTasks([...tasks, nextTask]);
    setTask("");
  }

  return (
    <main className="shell">
      <section className="panel">
        <h1>Project Tasks</h1>
        <div className="entry">
          <input
            value={task}
            onChange={(event) => setTask(event.target.value)}
            placeholder="Add a task"
          />
          <button onClick={addTask}>Add</button>
        </div>
        <ul>
          {tasks.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </main>
  );
}
"""
            files["src/styles.css"] = """
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  background: linear-gradient(135deg, #eef2e3, #b8d8d8);
  color: #1f2a2a;
}

.shell {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 24px;
}

.panel {
  width: min(680px, 100%);
  background: #ffffff;
  border: 1px solid #263737;
  padding: 28px;
}

.entry {
  display: flex;
  gap: 8px;
}

input {
  flex: 1;
  padding: 10px;
}

button {
  padding: 10px 16px;
  border: 0;
  background: #263737;
  color: #ffffff;
  cursor: pointer;
}

li {
  margin: 8px 0;
}
"""
            files[f"src/main.{main_ext}"] = """
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""
            files["index.html"] = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Jarvis React App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.{main_ext}"></script>
  </body>
</html>
"""
            files["package.json"] = """
{
  "name": "jarvis-react-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^5.0.0",
    "vite": "^7.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {}
}
"""

        elif template_name == "mobile-flutter" and language == "kotlin":
            files["android/app/src/main/kotlin/com/jarvis/MainActivity.kt"] = """
package com.jarvis

import io.flutter.embedding.android.FlutterActivity

class MainActivity: FlutterActivity() {
}
"""
            files["lib/main.dart"] = """
import 'package:flutter/material.dart';

void main() => runApp(JarvisApp());

class JarvisApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: Colors.black87,
        appBar: AppBar(
          title: Text("JARVIS Mobile Interface"),
          backgroundColor: Colors.deepPurple,
        ),
        body: Center(
          child: Text(
            "Welcome to the local system.",
            style: TextStyle(color: Colors.white, fontSize: 20),
          ),
        ),
      ),
    );
  }
}
"""

        elif template_name == "rust" or language == "rust":
            files["Cargo.toml"] = """
[package]
name = "jarvis-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.37.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
"""
            files["src/main.rs"] = """
#[tokio::main]
async fn main() {
    println!("Initializing JARVIS Core System in Rust...");
    println!("Environment checks completed successfully.");
}
"""

        elif language == "go":
            files["go.mod"] = "module jarvis-system\n\ngo 1.21"
            files["main.go"] = """
package main

import (
	"fmt"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "JARVIS Microservice running in Go!")
}

func main() {
	http.HandleFunc("/", handler)
	fmt.Println("Server listening on :8080...")
	http.ListenAndServe(":8080", nil)
}
"""

        elif language == "cpp":
            files["main.cpp"] = """
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "JARVIS High-Performance Core starting up in C++..." << std::endl;
    std::vector<std::string> components = {"Core", "AutoML", "Agents", "Testing"};
    for (const auto& comp : components) {
        std::cout << "Loaded component: " << comp << std::endl;
    }
    return 0;
}
"""

        elif language == "java":
            files["src/com/jarvis/CoreApp.java"] = """
package com.jarvis;

public class CoreApp {
    public static void main(String[] args) {
        System.out.println("JARVIS Enterprise Platform starting up in Java...");
        System.out.println("Ready for scalable execution.");
    }
}
"""

        elif language == "swift":
            files["main.swift"] = """
import Foundation

print("JARVIS Apple Core System initialized in Swift.")
let components = ["Vision", "Voice", "RAG"]
for component in components {
    print("Mounted modular component: \\(component)")
}
"""

        elif language == "csharp":
            files["Program.cs"] = """
using System;

namespace JarvisCore
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("JARVIS .NET Systems Online.");
        }
    }
}
"""

        elif language == "php":
            files["index.php"] = """
<?php
echo "JARVIS Web Portal initialized in PHP.";
?>
"""

        elif template_name == "devops-pipeline":
            files["docker-compose.yml"] = """
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: jarvis_db
      POSTGRES_PASSWORD: secret_password
    ports:
      - "5432:5432"
"""
            files["kubernetes.yaml"] = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jarvis-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jarvis-app
  template:
    metadata:
      labels:
        app: jarvis-app
    spec:
      containers:
      - name: api
        image: jarvis-api:latest
        ports:
        - containerPort: 8000
"""

        elif template_name == "database-schema":
            files["schema.sql"] = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

        elif template_name == "game-engine-mini":
            files["mini_game.py"] = """
import pygame
import sys

def run_game():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("JARVIS Mini Game")
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((20, 24, 30))
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    print("Starting mini pygame template...")
    # run_game()
"""

        elif template_name == "machine-learning-pipeline":
            files["ml_pipeline.py"] = """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_ml_pipeline():
    print("Generating simulated dataset...")
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting local Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Pipeline complete. Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_ml_pipeline()
"""
        if not files:
            files["README.md"] = f"# JARVIS Boilerplate\nScaffold for {template_name} in {language}.\n"

        return files
