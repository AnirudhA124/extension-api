import re
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,EmailStr
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
import google.auth.exceptions
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Firebase
if not firebase_admin._apps:
    firebase_config = os.getenv("FIREBASE_CONFIG")
    cred_dict = json.loads(firebase_config)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Gemini config
genai.configure(api_key="AIzaSyCSrh2w4-Nt11jh1QQUc_wQG7kt_WI_xtM")
model = genai.GenerativeModel("models/gemini-1.5-flash")


class RequirementRequest(BaseModel):
    requirement: str

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RequirementRequest(BaseModel):
    requirement: str

# Your existing Firebase setup assumed...

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

@app.post("/forgot-password")
async def forgot_password(data: ForgotPasswordRequest):
    email = data.email.strip()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    try:
        # ✅ Generate reset link
        reset_link = auth.generate_password_reset_link(email)
        print("🔗 Generated reset link:", reset_link)

        # ✅ Send email using SMTP (e.g., Gmail)
        sender_email = "agrawalanirudh18@gmail.com"
        sender_password = "uien meff nohq lwls"  # Use App Password (NOT your main Gmail password)
        subject = "Reset your password"

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email
        message["Subject"] = subject

        html_content = f"""
        <html>
          <body>
            <p>Hi,<br>
               Click the link below to reset your password:<br>
               <a href="{reset_link}">Reset Password</a>
            </p>
          </body>
        </html>
        """

        message.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())

        return {"success": True, "message": "Password reset email sent."}

    except auth.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        print("Password reset error:", str(e))
        raise HTTPException(status_code=500, detail="Error sending reset email.")

@app.post("/signup")
async def signup(data: SignupRequest):
    name, email, password = data.name.strip(), data.email.strip(), data.password.strip()

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="All fields are required.")

    try:
        user_record = auth.create_user(email=email, password=password, display_name=name)
        db.collection("users").document(user_record.uid).set({
            "name": name,
            "email": email,
            "uid": user_record.uid,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return {"success": True, "message": "Account created successfully. Please log in."}

    except auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=400, detail="Email already in use.")
    except Exception as e:
        print("Error during signup:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/login")
async def login(data: LoginRequest):
    email, password = data.email.strip(), data.password.strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required.")

    try:
        user = auth.get_user_by_email(email)
        return {
            "success": True,
            "message": "Login verified by email (password not validated).",
            "user": {
                "uid": user.uid,
                "email": user.email,
                "name": user.display_name,
            }
        }
    except auth.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        print("Login error:", str(e))
        raise HTTPException(status_code=500, detail="Login failed. Server error.")

def split_main_code_blocks(main_code_text):
    """Split main code into smaller logical blocks for better notebook structure"""
    if not main_code_text.strip():
        return []
    
    lines = main_code_text.split('\n')
    blocks = []
    current_block = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip().lower()
        
        # Skip empty lines at the beginning of a potential new block
        if not stripped and not current_block:
            i += 1
            continue
            
        # Check if this line should start a new block
        should_start_new_block = False
        
        # Start new block for comments that indicate sections
        if stripped.startswith('#') and any(keyword in stripped for keyword in 
                                          ['data generation', 'data processing', 'analysis', 
                                           'visualization', 'plotting', 'example', 'usage',
                                           'generate', 'process', 'analyze', 'plot', 'calculate']):
            should_start_new_block = True
        
        # Start new block for variable assignments that look like main tasks
        elif any(pattern in line for pattern in ['=', 'primes_less_than', 'num_to_check', 'N =', 'results =', 'data =']):
            # Check if this is a significant assignment (not just a loop variable)
            if not any(skip_pattern in stripped for skip_pattern in ['for ', 'while ', 'if ', 'elif ', 'else:']):
                should_start_new_block = True
        
        # Start new block for function calls that are standalone operations
        elif re.match(r'^\s*\w+\s*\(', line) and not any(skip in stripped for skip in ['print', 'return', 'if', 'for', 'while']):
            should_start_new_block = True
            
        # Start new block for print statements (often separate display tasks)
        elif 'print(' in line and current_block:
            should_start_new_block = True
            
        # Start new block for plotting commands
        elif any(plot_cmd in line for plot_cmd in ['plt.', 'sns.', 'fig', 'ax.', 'plot(']):
            should_start_new_block = True
            
        # Start new block for conditional statements that seem like separate examples
        elif stripped.startswith('if ') and any(keyword in stripped for keyword in ['is_prime', 'check', 'test']):
            should_start_new_block = True
        
        # If we should start a new block and we have content in current block
        if should_start_new_block and current_block:
            block_text = '\n'.join(current_block).strip()
            if block_text:
                blocks.append(block_text)
            current_block = []
        
        current_block.append(line)
        i += 1
    
    # Add the final block
    if current_block:
        block_text = '\n'.join(current_block).strip()
        if block_text:
            blocks.append(block_text)
    
    # Post-process blocks to ensure better separation
    return post_process_blocks(blocks)

def post_process_blocks(blocks):
    """Further refine blocks to ensure each task is truly separate"""
    refined_blocks = []
    
    for block in blocks:
        # Check if this block contains multiple distinct tasks
        lines = block.split('\n')
        sub_blocks = []
        current_sub_block = []
        
        for line in lines:
            stripped = line.strip()
            
            # Split on significant transitions
            if (stripped.startswith('#') and 
                any(keyword in stripped.lower() for keyword in ['example', 'usage', 'test', 'check']) and
                current_sub_block):
                
                # Save current sub-block
                sub_block_text = '\n'.join(current_sub_block).strip()
                if sub_block_text:
                    sub_blocks.append(sub_block_text)
                current_sub_block = [line]
                
            elif (stripped.startswith('if ') and 
                  'is_prime' in stripped and 
                  current_sub_block and
                  not any('if ' in cb_line for cb_line in current_sub_block)):
                
                # This looks like a separate example/test
                sub_block_text = '\n'.join(current_sub_block).strip()
                if sub_block_text:
                    sub_blocks.append(sub_block_text)
                current_sub_block = [line]
                
            else:
                current_sub_block.append(line)
        
        # Add the final sub-block
        if current_sub_block:
            sub_block_text = '\n'.join(current_sub_block).strip()
            if sub_block_text:
                sub_blocks.append(sub_block_text)
        
        # Add sub-blocks or original block if no splitting occurred
        if len(sub_blocks) > 1:
            refined_blocks.extend(sub_blocks)
        else:
            refined_blocks.append(block)
    
    return [b for b in refined_blocks if b.strip()]

def split_single_block_aggressively(code_block):
    """More aggressive splitting for cases where logical separation is needed"""
    lines = code_block.split('\n')
    blocks = []
    current_block = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Force split on these patterns
        force_split_patterns = [
            lambda l: l.startswith('# ---') and any(kw in l.lower() for kw in ['example', 'usage', 'test']),
            lambda l: l.startswith('if ') and 'is_prime' in l and len(current_block) > 0,
            lambda l: 'print(' in l and any('=' in cb for cb in current_block),
            lambda l: stripped.startswith('num_to_check') and len(current_block) > 0,
        ]
        
        should_force_split = any(pattern(stripped) for pattern in force_split_patterns)
        
        if should_force_split and current_block:
            block_text = '\n'.join(current_block).strip()
            if block_text:
                blocks.append(block_text)
            current_block = []
        
        current_block.append(line)
        
        # Also split after certain completion patterns
        if (stripped.startswith('print(') and 
            'prime numbers less than' in stripped.lower() and 
            i < len(lines) - 1):
            
            block_text = '\n'.join(current_block).strip()
            if block_text:
                blocks.append(block_text)
            current_block = []
    
    # Add final block
    if current_block:
        block_text = '\n'.join(current_block).strip()
        if block_text:
            blocks.append(block_text)
    
    return [b for b in blocks if b.strip()]

def parse_code_structure(code: str):
    if not code:
        return {'imports': '', 'functions': [], 'classes': [], 'main_code': []}

    lines = code.split('\n')
    imports, functions, classes, main_code_lines = [], [], [], []
    current_block, current_type, indent_level = [], None, 0

    for line in lines:
        stripped = line.strip()
        
        # Handle empty lines
        if not stripped:
            if current_block:
                current_block.append(line)
            continue

        # Handle imports
        if re.match(r'^\s*(?:import|from)\s+', line):
            if current_block and current_type:
                block = '\n'.join(current_block).strip()
                if block:
                    if current_type == 'function':
                        functions.append(block)
                    elif current_type == 'class':
                        classes.append(block)
                    else:
                        main_code_lines.extend(current_block)
                current_block = []
            imports.append(line)
            current_type = None
            continue

        # Handle function definitions
        if re.match(r'^\s*def\s+', line):
            if current_block and current_type:
                block = '\n'.join(current_block).strip()
                if block:
                    if current_type == 'function':
                        functions.append(block)
                    elif current_type == 'class':
                        classes.append(block)
                    else:
                        main_code_lines.extend(current_block)
            current_block = [line]
            current_type = 'function'
            indent_level = len(line) - len(line.lstrip())
            continue

        # Handle class definitions
        if re.match(r'^\s*class\s+', line):
            if current_block and current_type:
                block = '\n'.join(current_block).strip()
                if block:
                    if current_type == 'function':
                        functions.append(block)
                    elif current_type == 'class':
                        classes.append(block)
                    else:
                        main_code_lines.extend(current_block)
            current_block = [line]
            current_type = 'class'
            indent_level = len(line) - len(line.lstrip())
            continue

        # Handle continuation of functions/classes
        if current_type in ['function', 'class']:
            line_indent = len(line) - len(line.lstrip())
            if line_indent > indent_level or (line_indent == indent_level and not stripped.startswith(('def ', 'class '))):
                current_block.append(line)
                continue
            else:
                # End of function/class
                block = '\n'.join(current_block).strip()
                if block:
                    if current_type == 'function':
                        functions.append(block)
                    else:
                        classes.append(block)
                current_block = [line]
                current_type = 'main'
                continue

        # Default to main code
        if current_type != 'main':
            current_block = [line]
            current_type = 'main'
        else:
            current_block.append(line)

    # Handle final block
    if current_block:
        if current_type == 'function':
            block = '\n'.join(current_block).strip()
            if block:
                functions.append(block)
        elif current_type == 'class':
            block = '\n'.join(current_block).strip()
            if block:
                classes.append(block)
        else:
            main_code_lines.extend(current_block)

    # Process main code - split into logical blocks
    main_code_text = '\n'.join(main_code_lines).strip()
    main_code_blocks = split_main_code_blocks(main_code_text) if main_code_text else []

    return {
        'imports': '\n'.join(imports),
        'functions': functions,
        'classes': classes,
        'main_code': main_code_blocks
    }


def create_notebook_cells(code_structure, file_structure):
    cells = []

    def add_code_block(title, blocks):
        if blocks:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {title}\n"]
            })
            for block in blocks:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in block.splitlines()]
                })

    if code_structure['imports'].strip():
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Import Dependencies\n"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_structure['imports'].splitlines()]
        })

    add_code_block("Class Definitions", code_structure['classes'])
    add_code_block("Function Definitions", code_structure['functions'])
    add_code_block("Main Execution", code_structure['main_code'])

    return cells


def create_file_structure(files_info, base_code):
    files = []

    def make_notebook(content):
        structure = parse_code_structure(content)
        return {
            "cells": create_notebook_cells(structure, {}),
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
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "pygments_lexer": "ipython3",
                    "nbconvert_exporter": "python",
                    "file_extension": ".py"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

    if isinstance(files_info, list) and files_info:
        for file_info in files_info:
            path = file_info.get("path")
            content = file_info.get("content", base_code)
            if path.endswith('.ipynb'):
                notebook = make_notebook(content)
                files.append({"path": path, "content": json.dumps(notebook, indent=2)})
            else:
                files.append({"path": path, "content": content})
    else:
        notebook = make_notebook(base_code)
        files.append({"path": "main.ipynb", "content": json.dumps(notebook, indent=2)})

    return files


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "FastAPI server is running"}


@app.post("/receive-data")
async def receive_data(req: RequirementRequest):
    try:
        requirement = req.requirement
        print("📥 Received Requirement:", requirement)

        # Fixed: Use regular string instead of f-string to avoid formatting issues
        gemini_prompt = """
You are Octavian, an AI coding assistant that generates Python code for Jupyter notebooks.

User Requirement: """ + requirement + """

Generate well-structured Python code that will be converted into a Jupyter notebook. Follow these guidelines:

1. *Code Structure*: Write complete, executable Python code with clear separation:
   - Import statements at the top
   - Function definitions (each function separate)
   - Main execution code broken into logical steps
   - Each major task should be separable for debugging

2. *For Data Science/Analysis Tasks*: Structure the code in distinct logical blocks:
   - Data generation/simulation
   - Data processing/analysis  
   - Visualization/plotting
   - Each block should be completely separate with clear comments

3. *Separation Guidelines*: 
   - Put function definitions separate from their usage
   - Separate data generation from analysis
   - Separate analysis from visualization  
   - Use clear comments to indicate what each block does
   - Add blank lines between different logical sections

4. *Output Format*: Respond with clean Python code only.
   Make sure the code is:
   - Executable and complete
   - Well-commented with section headers
   - Properly separated into logical blocks
   - Uses appropriate libraries

Generate the Python code now with clear logical separation:
"""

        response = model.generate_content(gemini_prompt)
        response_text = response.text.strip()
        print("🧠 Gemini Raw Response:\n", response_text)

        # Try parsing JSON
        try:
            match = re.search(r'(?:json)?\s*(\{.*?\})\s*', response_text, re.DOTALL)
            if match:
                data_json = json.loads(match.group(1))
            else:
                cleaned = re.sub(r"^(?:json)?|$", "", response_text.strip(), flags=re.MULTILINE).strip()
                data_json = json.loads(cleaned)

            files_info = data_json.get("files", [])
            main_code = data_json.get("main_code", "")
            dependencies = data_json.get("dependencies", [])

        except (json.JSONDecodeError, AttributeError) as e:
            print("⚠ JSON parse failed:", e)
            # fallback to treating the entire response as raw code
            main_code = re.sub(r"^(?:python)?|$", "", response_text.strip(), flags=re.MULTILINE).strip()
            files_info = []
            dependencies = []

        # Final fallback if Gemini returned just plain code
        if not main_code and not files_info:
            main_code = response_text

        files = create_file_structure(files_info, main_code)
        print("📁 Generated files:", [f["path"] for f in files])

        # Also return the raw code for easier access
        return {
            "files": files, 
            "dependencies": dependencies,
            "raw_code": main_code  # Add this for direct code access
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": f"Failed to generate notebook: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
