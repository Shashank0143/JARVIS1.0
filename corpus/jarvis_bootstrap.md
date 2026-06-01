# Jarvis Bootstrap Knowledge

This local assistant should behave like a small desktop Jarvis:

- Keep a central controller that routes commands to focused classes.
- Keep machine actions separate from coding, voice, vision, memory, and learning.
- Save every command, response, and useful answer in local data files.
- Use RAG for grounded answers instead of inventing facts.
- Learn from trusted internet sources by fetching pages, cleaning text, indexing chunks, and training the local model state.
- Avoid hosted AI APIs and hosted model inference.
- Ask for confirmation before destructive actions such as deleting files, deleting folders, or erasing file contents.
- Prefer local/offline components for speech recognition, camera analysis, and model inference.
- When model generation is weak, fall back to retrieved evidence and practical steps.
- Keep source URLs in memory so repeated learning jobs do not duplicate the same page.

Useful local command grammar:

```text
open chrome
open youtube
create file notes.txt
write notes.txt with hello
update line 1 in notes.txt with new text
replace old text in notes.txt with new text
delete notes.txt
confirm delete notes.txt
learn Python is the primary project language
learn internet python file handling
learn curriculum
what do you see
```
