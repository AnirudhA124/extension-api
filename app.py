import re
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

genai.configure(api_key="AIzaSyCSrh2w4-Nt11jh1QQUc_wQG7kt_WI_xtM")
model = genai.GenerativeModel("models/gemini-1.5-flash")

def extract_imports(code: str):
    """Extract import statements from code"""
    matches = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', code, re.MULTILINE)
    cleaned = list(set(matches))
    return cleaned

def parse_code_structure(code: str):
    """Parse code into logical sections: imports, functions, classes, and main execution"""
    lines = code.split('\n')
    
    imports = []
    functions = []
    classes = []
    main_code = []
    current_block = []
    current_type = None
    indent_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines temporarily
        if not stripped:
            if current_block:
                current_block.append(line)
            continue
            
        # Check if it's an import
        if re.match(r'^\s*(?:import|from)\s+', line):
            if current_block and current_type:
                if current_type == 'function':
                    functions.append('\n'.join(current_block))
                elif current_type == 'class':
                    classes.append('\n'.join(current_block))
                else:
                    main_code.append('\n'.join(current_block))
                current_block = []
            imports.append(line)
            current_type = None
            continue
            
        # Check if it's a function definition
        if re.match(r'^\s*def\s+', line):
            if current_block and current_type:
                if current_type == 'function':
                    functions.append('\n'.join(current_block))
                elif current_type == 'class':
                    classes.append('\n'.join(current_block))
                else:
                    main_code.append('\n'.join(current_block))
            current_block = [line]
            current_type = 'function'
            indent_level = len(line) - len(line.lstrip())
            continue
            
        # Check if it's a class definition
        if re.match(r'^\s*class\s+', line):
            if current_block and current_type:
                if current_type == 'function':
                    functions.append('\n'.join(current_block))
                elif current_type == 'class':
                    classes.append('\n'.join(current_block))
                else:
                    main_code.append('\n'.join(current_block))
            current_block = [line]
            current_type = 'class'
            indent_level = len(line) - len(line.lstrip())
            continue
            
        # Check if we're still in a function or class
        if current_type in ['function', 'class']:
            line_indent = len(line) - len(line.lstrip())
            if line_indent > indent_level or (line_indent == indent_level and stripped.startswith(('def ', 'class ', '@'))):
                current_block.append(line)
                continue
            else:
                # End of function/class block
                if current_type == 'function':
                    functions.append('\n'.join(current_block))
                elif current_type == 'class':
                    classes.append('\n'.join(current_block))
                current_block = [line]
                current_type = 'main'
                continue
                
        # Regular code
        current_block.append(line)
        current_type = 'main'
    
    # Handle remaining block
    if current_block:
        if current_type == 'function':
            functions.append('\n'.join(current_block))
        elif current_type == 'class':
            classes.append('\n'.join(current_block))
        else:
            main_code.append('\n'.join(current_block))
    
    return {
        'imports': '\n'.join(imports) if imports else '',
        'functions': functions,
        'classes': classes,
        'main_code': [block for block in main_code if block.strip()]
    }

def create_notebook_cells(code_structure, file_structure):
    """Create notebook cells based on code structure and file organization"""
    cells = []
    
    # Add imports cell if exists
    if code_structure['imports'].strip():
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Import Dependencies\n"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_structure['imports'].splitlines()]
        })
    
    # Add classes cell if exists
    if code_structure['classes']:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Class Definitions\n"]
        })
        for class_code in code_structure['classes']:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in class_code.splitlines()]
            })
    
    # Add functions cell if exists
    if code_structure['functions']:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Function Definitions\n"]
        })
        for func_code in code_structure['functions']:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in func_code.splitlines()]
            })
    
    # Add main execution code
    if code_structure['main_code']:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Main Execution\n"]
        })
        for main_block in code_structure['main_code']:
            if main_block.strip():
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in main_block.splitlines()]
                })
    
    return cells

def create_file_structure(files_info, base_code):
    """Create multiple files based on LLM suggested structure"""
    files = []
    
    if isinstance(files_info, list):
        for file_info in files_info:
            if isinstance(file_info, dict) and 'path' in file_info:
                file_path = file_info['path']
                file_content = file_info.get('content', base_code)
                
                if file_path.endswith('.ipynb'):
                    # Create notebook file
                    code_structure = parse_code_structure(file_content)
                    cells = create_notebook_cells(code_structure, file_info)
                    
                    notebook = {
                        "cells": cells,
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
                    
                    files.append({
                        "path": file_path,
                        "content": json.dumps(notebook, indent=2)
                    })
                else:
                    # Regular Python file
                    files.append({
                        "path": file_path,
                        "content": file_content
                    })
    else:
        # Fallback: create single notebook
        code_structure = parse_code_structure(base_code)
        cells = create_notebook_cells(code_structure, {})
        
        notebook = {
            "cells": cells,
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
        
        files.append({
            "path": "main.ipynb",
            "content": json.dumps(notebook, indent=2)
        })
    
    return files

@app.route("/receive-data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json()
        requirement = data.get("requirement", "")
        print("📥 Received Requirement:", requirement)

        # Enhanced prompt for better structure and file organization
        gemini_prompt = f"""
You are a Python code generator that creates well-structured projects. 

Requirement: {requirement}

Please provide your response in the following JSON format:
{{
    "files": [
        {{
            "path": "main.ipynb",
            "content": "# Your main notebook code here"
        }},
        {{
            "path": "utils.py",
            "content": "# Utility functions here"
        }}
    ],
    "main_code": "# Complete implementation code here"
}}

Guidelines:
1. Organize code into logical files (main notebook, utility modules, etc.)
2. Separate imports, class definitions, function definitions, and main execution
3. Use clear function and class names
4. Include proper documentation
5. Structure the project for maintainability
6. The main_code should contain the complete implementation
7. If creating multiple files, organize them logically (utils.py for utilities, models.py for ML models, etc.)

Focus on creating clean, modular code that follows Python best practices.
"""

        response = model.generate_content(gemini_prompt)
        response_text = response.text.strip()
        print("🧠 Gemini Raw Response:\n", response_text)

        # Try to parse JSON response
        try:
            # Extract JSON from response if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group(1))
            else:
                # Try to parse the entire response as JSON
                response_json = json.loads(response_text)
            
            files_info = response_json.get('files', [])
            main_code = response_json.get('main_code', response_text)
            
        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat entire response as code
            print("⚠️ Could not parse JSON, treating as raw code")
            main_code = re.sub(r"^```(?:python)?|```$", "", response_text.strip(), flags=re.MULTILINE).strip()
            files_info = []

        # Extract dependencies from main code
        dependencies = extract_imports(main_code)
        print("📦 Detected dependencies:", dependencies)

        # Create file structure
        files = create_file_structure(files_info, main_code)
        
        print("📁 Generated files:")
        for file in files:
            print(f"  - {file['path']}")

        return jsonify({
            "files": files,
            "dependencies": dependencies
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": f"Failed to generate notebook: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Flask server is running"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)