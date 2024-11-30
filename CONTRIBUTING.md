# Contribution Guidelines

## Overview

This document outlines the design concepts, and contribution guidelines for the
AI Dev Console project. We follow Python's standards (PEPs) while maintaining a
clear, modular structure.

## Project Structure

```
ai-dev-console/
├── src/                         # Source code root (PyPA recommendation)
│   ├── ai_dev_console/          # Main package (reusable library)
│   │   ├── models/              # Domain models
│   │   ├── services/            # Business logic
│   │   └── config/              # Configuration
│   └── ai_dev_console_apps/     # Command entrypoints
│       ├── cli/                 # Command-line interface entry points (prompt, debug, etc)
│       └── gui/                 # Gui apps (streamlit, other)
├── tests/                       # Test suite
├── pyproject.toml               # Project metadata and dependencies
├── CONTRIBUTING.md              # Contribution Guidelines
└── README.md                    # Project documentation
```

## Practices

1. **Package Design**

   - Keep core packages independent
   - Use dependency injection
   - Follow SOLID principles

2. **Testing**

   - Use descriptive names for test functions and methods.
   - Ensure each test is independent; avoid dependencies between tests.
   - Utilize setup and teardown methods to maintain a consistent test
     environment.
   - Design tests to verify a single aspect of the code's behavior.
   - Follow PEP 8 guidelines for code style within tests.
   - Aim for comprehensive test coverage, including edge cases.
   - Use specific assertion methods to validate expected outcomes.
   - Include docstrings in test functions to describe their purpose.
   - Tests are written using pytest

3. **Documentation**

   - Document public APIs
   - Include usage examples
   - Keep design decisions documented

## Design Concepts and Rationale

| Design Concept                    | Rationale                                                                            | Reference                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Source in `src/` directory        | Prevents import conflicts and ensures installed package is used during development   | [PyPA Packaging Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)                     |
| Separate package from entrypoints | Enables reuse of core functionality while providing multiple ways to use the package | [entry-points-spec](https://packaging.python.org/en/latest/specifications/entry-points/)                         |
| Use of `pyproject.toml`           | Modern, standardized project metadata and build system configuration                 | [pyproject-toml-spec](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-toml-spec) |
| Type hints throughout             | Enables static type checking and better IDE support                                  | [PEP 484](https://peps.python.org/pep-0484/), [typing-spec](https://typing.readthedocs.io/en/latest/spec/)       |
| Domain-driven design in models    | Clear separation of concerns and business logic                                      | [DDD Refrence](https://www.domainlanguage.com/wp-content/uploads/2016/05/DDD_Reference_2015-03.pdf)              |
| Optional dependencies             | Allows users to install only what they need                                          | [PEP 508](https://peps.python.org/pep-0508/)                                                                     |

## Development Guidelines

### Setting Up Development Environment

```sh
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install package with development dependencies
pip install -e ".[dev]"
```

### Code Style

- [PEP 8](https://peps.python.org/pep-0008/) for code style
- [PEP 484](https://peps.python.org/pep-0484/) for type hints
- Black for code formatting
- isort for import sorting

### Type Checking

- All code must be type-hinted
- Use mypy for static type checking

```sh
mypy src/
```

## Package Architecture

### Core Package (`ai_dev_console`)

The main package is designed to be reusable and independent of any specific
interface:

| Component   | Purpose                                  | Design Pattern               |
| ----------- | ---------------------------------------- | ---------------------------- |
| `models/`   | Domain models and data structures        | Domain-Driven Design         |
| `services/` | Business logic and external integrations | Service Layer Pattern        |
| `config/`   | Configuration management                 | Configuration Object Pattern |

### Entry Points (`cmd/`)

Different ways to use the package:

| Entry Point | Purpose                   | Implementation        |
| ----------- | ------------------------- | --------------------- |
| `app/`      | Streamlit web application | UI Components Pattern |
| `cli/`      | Command-line interface    | Command Pattern       |

## Dependency Management

### Core Dependencies

- Required for basic package functionality
- Must be lightweight and well-maintained

### Optional Dependencies

```toml
[project.optional-dependencies]
app = ["streamlit>=1.30.0"]
cli = ["click>=8.0.0"]
dev = ["pytest>=7.0.0", "black>=23.0.0", "mypy>=1.5.1"]
```

## Contributing Process

- **Fork and Clone**

  ```sh
  git clone https://github.com/yourusername/ai-dev-console.git
  ```

- **Create Feature Branch**

  ```sh
  git checkout -b feature/your-feature-name
  ```

- **Development Workflow**

  - Write tests first (TDD)
  - Implement feature
  - Run tests and type checking
  - Format code

  ```sh
  # Run full check
  make check
  ```

1. **Submit Pull Request**

   - Clear description of changes
   - Reference any related issues
   - Ensure all checks pass

## Release Process

| Stage      | Actions                              | Tools                                        |
| ---------- | ------------------------------------ | -------------------------------------------- |
| Versioning | Follow semantic versioning           | [PEP 440](https://peps.python.org/pep-0440/) |
| Building   | Build source and wheel distributions | `build`                                      |
| Publishing | Upload to PyPI                       | `twine`                                      |

## Documentation

- Use Google-style docstrings
- Keep README.md updated
- Include type hints in documentation
- Provide usage examples
