## 2025-12-25 - Suppressing TensorFlow Logs for CLI Tools

**Learning:** TensorFlow is extremely verbose by default, printing hardware info and optimization warnings that can confuse users of CLI tools.
**Action:** Always set `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'` *before* any `import tensorflow` or `import tensorflow_hub` statement in CLI entry points. This cleans up the output significantly, making the tool feel more professional and less "broken".
