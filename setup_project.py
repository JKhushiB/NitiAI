#!/usr/bin/env python3
"""
Setup script for NitiAI project
Run this script to organize your project structure and prepare for GitHub upload
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the proper directory structure"""
    
    # Define the new structure
    dirs_to_create = [
        "utils",
        ".streamlit",
        "data", 
        "notebooks",
        "tests"
    ]
    
    print("📁 Creating directory structure...")
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ Created: {dir_name}/")
    
    return True

def move_existing_files():
    """Move existing files to proper locations"""
    
    file_moves = [
        ("policy_data.csv", "data/policy_data.csv"),
        ("updated_dataset.csv", "data/updated_dataset.csv"), 
        ("data_preprocessing.ipynb", "notebooks/data_preprocessing.ipynb")
    ]
    
    print("\n📦 Moving existing files...")
    for src, dst in file_moves:
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                print(f"   ✅ Moved: {src} → {dst}")
            except Exception as e:
                print(f"   ⚠️  Could not move {src}: {e}")
        else:
            print(f"   ℹ️  File not found: {src}")

def create_utils_init():
    """Create __init__.py file for utils package"""
    
    init_content = '''# utils/__init__.py

from .user_profile import UserProfile
from .tools import *
from .agent_setup import setup_agent

__all__ = [
    'UserProfile',
    'SchemeSearchTool',
    'EligibilityCheckTool', 
    'DocumentRequirementTool',
    'BenefitsSearchTool',
    'SchemeSummaryTool',
    'setup_agent'
]'''
    
    with open("utils/__init__.py", "w") as f:
        f.write(init_content)
    
    print("   ✅ Created: utils/__init__.py")

def create_env_template():
    """Create environment template file"""
    
    env_template = """# .env.template
# Copy this file to .env and fill in your actual values

GROQ_API_KEY=your_groq_api_key_here

# Optional: Other API keys
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("   ✅ Created: .env.template")

def update_final_code():
    """Update final_code.py to work with new structure"""
    
    if not os.path.exists("final_code.py"):
        print("   ⚠️  final_code.py not found, skipping update")
        return
    
    # Read the current file
    with open("final_code.py", "r") as f:
        content = f.read()
    
    # Update import paths for data files
    content = content.replace('"updated_dataset.csv"', '"data/updated_dataset.csv"')
    content = content.replace('"policy_data.csv"', '"data/policy_data.csv"')
    
    # Save as legacy file
    with open("legacy_final_code.py", "w") as f:
        f.write(content)
    
    print("   ✅ Created: legacy_final_code.py (updated with new paths)")

def create_run_script():
    """Create a simple run script"""
    
    run_script = """#!/bin/bash
# run.sh - Simple script to run the application

echo "🏛️  Starting NitiAI..."
echo "📊 Loading AI models..."

streamlit run app.py
"""
    
    with open("run.sh", "w") as f:
        f.write(run_script)
    
    # Make it executable on Unix systems
    try:
        os.chmod("run.sh", 0o755)
    except:
        pass
    
    print("   ✅ Created: run.sh")

def print_next_steps():
    """Print instructions for next steps"""
    
    next_steps = """
🎉 Project setup complete! 

📋 Next Steps:

1. 🔑 Set up your API key:
   - Copy .env.template to .env
   - Add your Groq API key to .env
   - Or create .streamlit/secrets.toml with your API key

2. 📝 Create the modular files:
   - Copy the user_profile.py content to utils/user_profile.py
   - Copy the tools.py content to utils/tools.py  
   - Copy the agent_setup.py content to utils/agent_setup.py
   - Copy the app.py content to app.py

3. 🧪 Test the application:
   - Run: pip install -r requirements.txt
   - Run: streamlit run app.py

4. 📤 Upload to GitHub:
   - git init
   - git add .
   - git commit -m "Initial commit: NitiAI Assistant"
   - git branch -M main
   - git remote add origin https://github.com/yourusername/nitiai.git
   - git push -u origin main

5. 🚀 Deploy:
   - Use Streamlit Cloud for easy deployment
   - Or deploy to Heroku, Railway, or other platforms

📁 Your project structure is now:
```
NITIAI/
├── app.py                  # Main Streamlit app
├── utils/                  # Modular components
│   ├── __init__.py
│   ├── user_profile.py
│   ├── tools.py
│   └── agent_setup.py
├── data/                   # Data files
│   ├── policy_data.csv
│   └── updated_dataset.csv
├── notebooks/              # Jupyter notebooks
│   └── data_preprocessing.ipynb
├── chroma_store/          # Vector database
├── requirements.txt       # Dependencies
├── .streamlit/           # Streamlit config
├── README.md            # Documentation
├── .gitignore          # Git ignore rules
└── .env.template      # Environment template
```

💡 Tips:
- Keep your API keys secure and never commit them
- Test locally before deploying
- Check the README.md for detailed instructions
- Use GitHub Issues to track bugs and features

Happy coding! 🚀
"""
    
    print(next_steps)

def main():
    """Main setup function"""
    
    print("🏛️  NitiAI Project Setup")
    print("=" * 50)
    
    try:
        # Create directory structure
        create_directory_structure()
        
        # Move existing files
        move_existing_files()
        
        # Create utility files
        print("\n🔧 Creating utility files...")
        create_utils_init()
        create_env_template()
        create_run_script()
        
        # Update existing code
        print("\n📝 Updating existing files...")
        update_final_code()
        
        # Print next steps
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()