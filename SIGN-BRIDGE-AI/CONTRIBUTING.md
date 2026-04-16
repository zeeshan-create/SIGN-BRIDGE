# Contributing to SignBridge

Thank you for your interest in contributing to SignBridge! This document outlines the guidelines for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We welcome contributors from all backgrounds and experience levels.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, check the existing issues to avoid duplicates. When filing a bug report, include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Node version, Python version)

### Suggesting Features

We welcome feature suggestions! Please include:

- **Clear description** of the feature
- **Use cases** and why it would be beneficial
- **Potential implementation approaches** (optional)

### Pull Requests

1. **Fork the repository**
2. **Create your feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes** with clear messages
   ```bash
   git commit -m 'Add: Description of changes'
   ```
4. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request** with a detailed description

### Coding Standards

#### JavaScript/React
- Use functional components with hooks
- Follow existing naming conventions
- Run linting before committing
- Add comments for complex logic

#### Python
- Follow PEP 8 style guide
- Use type hints where applicable
- Add docstrings for functions and classes

#### General
- Keep functions small and focused
- Write meaningful commit messages
- Test your changes before submitting

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ASL-Detection.git

# Add upstream
git remote add upstream https://github.com/signbridge/asl-detection.git

# Create a feature branch
git checkout -b feature/your-feature-name
```

## Testing

Before submitting a PR:

1. Test the frontend:
   ```bash
   cd web && npm run build
   ```

2. Test the backend:
   ```bash
   cd server && node index.js
   ```

3. Test the Python bridge:
   ```bash
   python bridge.py
   ```

## Review Process

- All submissions require review
- We aim to respond within 48 hours
- Feedback will be constructive and actionable
- Changes may be requested before merging

## Recognition

Contributors will be added to the README.md credits section.

---

Thank you for helping make SignBridge better!
