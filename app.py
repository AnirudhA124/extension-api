import re
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)  # 🔧 fixed typo: _name_ ➝ __name__
CORS(app)

genai.configure(api_key="AIzaSyCSrh2w4-Nt11jh1QQUc_wQG7kt_WI_xtM")
model = genai.GenerativeModel("models/gemini-1.5-flash")

def extract_imports(code: str):
    # 🔍 Match both 'import x' and 'from x import y'
    matches = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', code, re.MULTILINE)
    # Remove duplicates, strip built-ins like __future__ if needed
    cleaned = list(set(matches))
    return cleaned

@app.route("/receive-data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json()
        requirement = data.get("requirement", "")
        print("📥 Received Requirement:", requirement)

        gemini_prompt = f"""
You're an assistant that returns only code content (no markdown or comments).
The requirement is: {requirement}

Please return only valid Python code that should go inside a Jupyter Notebook (.ipynb).
"""

        response = model.generate_content(gemini_prompt)
        code = response.text.strip()
        print("🧠 Gemini Raw Code:\n", code)

        # ✂ Strip markdown fences if present
        cleaned_code = re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.MULTILINE).strip()

        # ✅ Extract imports
        dependencies = extract_imports(cleaned_code)
        print("📦 Detected dependencies:", dependencies)

        # ✅ Convert to notebook format
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in cleaned_code.splitlines()]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.5",
                    "mimetype": "text/x-python",
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "pygments_lexer": "ipython3",
                    "nbconvert_exporter": "python",
                    "file_extension": ".py"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        return jsonify({
            "files": [
                {
                    "path": "notebook.ipynb",
                    "content": json.dumps(notebook)
                }
            ],
            "dependencies": dependencies  # ✅ Sent back to extension
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": "Failed to generate notebook."}), 500

if __name__ == "__main__":  # 🔧 fixed typo: _main_ ➝ __main__
    app.run(debug=True)
