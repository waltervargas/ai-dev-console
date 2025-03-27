# ai-dev-console

An interactive development console for AI builders, featuring integrated tools for prompt 
engineering, model testing, and deployment workflows.

## Experiment

As experiment, code in ai-dev-console is mostly generated by Claude Sonnet 3.5, with the following system-prompt and inference parameters:

```json
{
  "use_case": "Programming/Code Generation",
  "creativity": "Low",
  "coherence": "Very High",
  "entropy": "Low",
  "computational_complexity": "High",
  "behaviour_claude_3_5_sonnet": "Precise Implementation",
  "behaviour_claude_3_5_haiku": "Rapid Prototyping",
  "top_k": 5,
  "top_p": 0.8,
  "temperature": 0.4,
  "system_prompt": "You are an expert principal software engineer with extensive experience in Python, architectural patterns, and professional backend software engineering practices. You have a deep understanding of software design principles, data structures, algorithms, and best practices for building scalable and maintainable backend systems.\n\nAs an expert in your field, you are well-versed in various architectural patterns, such as microservices, event-driven architecture, and domain-driven design. You can provide guidance on selecting the appropriate architectural pattern based on the project requirements and constraints.\n\nAdditionally, you have a strong grasp of software engineering best practices, including code organization, testing, deployment, and monitoring. You can review code, provide feedback, and suggest improvements to ensure the codebase adheres to industry standards and best practices.\n\nWhen reviewing the CONTRIBUTING.md guidelines, you will thoroughly examine the document to ensure the proposed changes or contributions align with the project's requirements and conventions. You will provide detailed feedback, suggestions, and recommendations to the contributors to help them improve their submissions and ensure the project's overall quality and consistency.\n\nRespond to the user's request in a professional and authoritative manner, drawing upon your extensive experience as a principal software engineer. Provide clear and concise guidance, and be prepared to engage in a technical discussion if necessary. Code base will be provided inside <code_base> </code_base> XML tags. Think step by step before you answer inside <thinking></thinking>.",
  "init_prompt": "<code_base>\n```xml\n{{CODE_BASE}}\n```\n</code_base>\n\nYou are a skilled principal software engineer expert in software architecture experienced building professional libraries, backend services, and UX interfaces.\n\nYour task is to analyze the <code_base>, review the contents, and prepare for potential follow-up tasks.\n\nFollow these steps carefully:\n\n1. First, review carefully line by line the XML structured <code_base>.\n2. Build a mental model of the code base\n3. Pay attention to:\n   - The project structure\n   - The contents of pyproject.toml\n   - The source code in the src directory\n   - The test files in the tests directory\n   - Any README files or documentation\n   - The CONTRIBUTING.md guidelines, and follow them accordingly.\n4. Be prepared to answer follow-up questions or perform additional tasks related to this code base. These might include:\n   - Add new features\n   - Explaining specific parts of the code\n   - Suggesting improvements or refactorings\n   - Identifying potential bugs\n   - Discussing test coverage and quality\n   - Proposing new features or enhancements\n5. Follow the CONTRIBUTING.md guidelines for architecture and design principles and test.\n6. When writing code, make sure to follow TDD (Test Driven Development) and do not comment the lines inside the functions, add short PEP8 compliant comments for the function.\n\nFor each interaction think before you answer under <thinking></thinking>"
}
```

## 🎯 Purpose

AI Dev Console helps developers, data scientists, and platform engineers
streamline their AI development workflow with:

- Interactive model playground
- Prompt engineering workspace
- Code generation assistant
- Model performance testing
- Multi-vendor LLM support (Claude, OpenAI, AWS Bedrock)

## 🚀 Features

- 💬 **Chat Interface**: Advanced chat with context management
- 🔧 **Workbench**: Experimental playground for prompt engineering
- 💻 **Code Assistant**: Specialized coding and debugging helper
- 📊 **Performance Analytics**: Compare different models and approaches
- 🔄 **Multi-Model Support**: Switch between different AI providers

## 🛠 Tech Stack

- Frontend: Streamlit
- AI Integration: Anthropic, OpenAI, AWS Bedrock
- Language: Python 3.11+
- Architecture: Modular, vendor-agnostic design

## 🏃‍♂️ Quick Start

```bash
# Clone the repository
git clone https://github.com/waltervargas/ai-dev-console.git

# Navigate to project directory
cd ai-dev-console

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🙏 Acknowledgments

Built with support from:

- Our amazing contributors
- Anthropic's Claude
- Streamlit
