# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/OuyangWenyu/hydromodel/issues>.

When reporting a bug, please include:

- **Operating system** name and version
- **Python version** you're using
- **hydromodel version** (run `python -c "import hydromodel; print(hydromodel.__version__)")
- **Detailed steps** to reproduce the bug
- **Error messages** (full traceback if available)
- **Expected behavior** vs. actual behavior
- **Code snippet** to reproduce (if applicable)

**Example bug report:**
```markdown
**Environment:**
- OS: Windows 11
- Python: 3.11.5
- hydromodel: 0.2.11

**Steps to reproduce:**
1. Load CAMELS-US dataset
2. Run calibration with SCE-UA
3. Error occurs during parameter saving

**Error message:**
```
FileNotFoundError: calibration_results.json not found
```

**Expected:** Results should be saved to output directory
```

### Fix Bugs

Look through the GitHub [issues](https://github.com/OuyangWenyu/hydromodel/issues) for bugs. Issues tagged with:
- `bug` - Confirmed bugs
- `help wanted` - Good for contributors
- `good first issue` - Great for first-time contributors

### Implement Features

Look through the GitHub [issues](https://github.com/OuyangWenyu/hydromodel/issues) for features. Issues tagged with:
- `enhancement` - New features or improvements
- `help wanted` - Open for contributions
- `documentation` - Documentation improvements

### Add New Models

hydromodel welcomes new hydrological model implementations! When adding a model:

1. Create a new file in `hydromodel/models/` (e.g., `your_model.py`)
2. Inherit from base classes or follow existing model patterns
3. Implement required methods:
   - `__init__`: Model initialization
   - `run`: Main simulation function
   - Define `param_limits` (parameter ranges)
4. Add model tests in `test/`
5. Document the model in `docs/models/your_model.md`
6. Update `MODEL_PARAM_DICT` in `model_config.py`

**Example:**
```python
class YourModel:
    def __init__(self, params, **kwargs):
        self.params = params

    def run(self, inputs, **kwargs):
        # Your model logic
        return outputs
```

### Write Documentation

hydromodel can always use more documentation! You can contribute by:

- Improving existing documentation
- Writing tutorials and examples
- Creating Jupyter notebook demonstrations
- Translating documentation to other languages
- Writing blog posts or articles about hydromodel

Documentation is built using [MkDocs](https://www.mkdocs.org/) with the Material theme.

### Submit Feedback

To propose a feature:

1. Open an issue at <https://github.com/OuyangWenyu/hydromodel/issues>
2. Tag it with `enhancement`
3. Explain in detail:
   - **What** the feature would do
   - **Why** it's needed
   - **How** it should work (if you have ideas)
4. Keep the scope narrow to make implementation easier

## Development Setup

### Prerequisites

- **Python 3.9 or higher**
- **Git** for version control
- **uv** (recommended) or pip for package management

### Setting Up Development Environment

#### Method 1: Using uv (Recommended)

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/your_username/hydromodel.git
cd hydromodel

# 3. Add upstream remote
git remote add upstream https://github.com/OuyangWenyu/hydromodel.git

# 4. Install uv (if not already installed)
pip install uv

# 5. Create development environment with all dependencies
uv sync --all-extras

# 6. Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 7. Verify installation
pytest test/
```

#### Method 2: Using pip and venv

```bash
# 1. Fork and clone (same as above)
git clone https://github.com/your_username/hydromodel.git
cd hydromodel

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install in editable mode with dev dependencies
pip install -e ".[dev,docs]"

# 5. Verify installation
pytest test/
```

### Development Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. **Make your changes** following our coding standards (see below)

3. **Write tests** for your changes:
   ```bash
   # Add tests to test/ directory
   # Run tests to ensure they pass
   pytest test/
   ```

4. **Update documentation** if needed:
   ```bash
   # Edit docs in docs/ directory
   # Build docs locally to preview
   mkdocs serve
   # View at http://127.0.0.1:8000/
   ```

5. **Format and lint your code**:
   ```bash
   # Format with black (if available)
   black hydromodel/ test/

   # Check code style (if flake8 available)
   flake8 hydromodel/ test/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description

   Detailed description of what changed and why.

   Fixes #123"  # Reference issue number if applicable
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **88 characters** (Black default)
- Use **type hints** where appropriate
- Write **docstrings** for all public functions, classes, and modules

### Docstring Format

Use **NumPy-style** docstrings:

```python
def calibrate(config, param_range_file=None):
    """
    Calibrate hydrological model with specified configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing data_cfgs, model_cfgs,
        training_cfgs, and evaluation_cfgs.
    param_range_file : str, optional
        Path to parameter range YAML file. If None, uses default ranges.

    Returns
    -------
    dict
        Dictionary mapping basin IDs to calibration results containing
        best parameters and objective values.

    Raises
    ------
    ValueError
        If configuration is invalid or missing required fields.

    Examples
    --------
    >>> config = {...}
    >>> results = calibrate(config)
    >>> print(results['01013500']['best_params'])

    See Also
    --------
    evaluate : Evaluate calibrated model
    UnifiedSimulator : Run model simulation

    Notes
    -----
    Results are saved to {output_dir}/{experiment_name}/ directory.
    """
    pass
```

### Naming Conventions

- **Functions/methods**: `lowercase_with_underscores`
- **Classes**: `CapitalizedWords`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private methods**: `_leading_underscore`
- **Protected methods**: `_single_leading_underscore`

### Code Organization

- Keep functions focused and small (ideally < 50 lines)
- Use meaningful variable names
- Add comments for complex logic
- Avoid deep nesting (max 3-4 levels)
- Extract repeated code into functions

## Testing Guidelines

### Writing Tests

- Use `pytest` for all tests
- Place tests in `test/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`

**Example test:**
```python
import pytest
import numpy as np
from hydromodel.models.xaj import xaj

def test_xaj_basic_run():
    """Test basic XAJ model execution."""
    # Arrange
    params = np.array([0.5, 0.3, 0.05, 15.0, 75.0, 90.0,
                       0.1, 50.0, 1.2, 0.3, 0.3, 0.5,
                       5.0, 0.7, 0.99])
    p_and_e = np.random.rand(100, 1, 2) * 10  # 100 days, 1 basin, 2 vars

    # Act
    results = xaj(p_and_e, params)

    # Assert
    assert results.shape == (100, 1)
    assert not np.any(np.isnan(results))
    assert np.all(results >= 0)

def test_xaj_parameter_validation():
    """Test XAJ parameter validation."""
    with pytest.raises(ValueError):
        xaj(np.random.rand(100, 1, 2), params=None)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_xaj.py

# Run specific test
pytest test/test_xaj.py::test_xaj_basic_run

# Run with coverage
pytest --cov=hydromodel --cov-report=html

# Run with verbose output
pytest -v

# Run only fast tests (skip slow)
pytest -m "not slow"
```

### Test Coverage

- Aim for **>80% code coverage** for new code
- Write tests for:
  - Normal cases
  - Edge cases
  - Error conditions
  - Boundary values

## Pull Request Guidelines

### Before Submitting

1. **Tests pass**: Run `pytest` and ensure all tests pass
2. **Code is formatted**: Use `black` or similar formatter
3. **Documentation updated**: Update relevant docs
4. **CHANGELOG updated**: Add entry to `docs/changelog.md`
5. **Commits are clean**: Use meaningful commit messages

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issue
Fixes #123

## How Has This Been Tested?
Describe the tests you ran.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed my own code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] Added tests that prove fix/feature works
- [ ] New and existing tests pass locally
- [ ] CHANGELOG.md updated
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in the release notes

## Project Structure

```
hydromodel/
├── hydromodel/              # Main package
│   ├── models/              # Model implementations
│   ├── trainers/            # Calibration, evaluation, simulation
│   ├── datasets/            # Data loading and preprocessing
│   └── utils/               # Utility functions
├── test/                    # Tests
├── docs/                    # Documentation
├── scripts/                 # Command-line scripts
├── configs/                 # Example configurations
└── pyproject.toml           # Project configuration
```

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Example: `0.2.11` → `0.3.0` (new features) → `1.0.0` (major release)

## Code Review Process

### What Reviewers Look For

- **Correctness**: Does the code work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Style**: Does it follow coding standards?
- **Performance**: Are there any performance concerns?
- **Maintainability**: Is the code easy to understand and maintain?

### Response Time

- Initial review: within 1 week
- Follow-up reviews: within 3-5 days
- For urgent fixes: within 1-2 days

## Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions
- **Email**: wenyuouyang@outlook.com for private matters

## Recognition

Contributors are credited in:
- Release notes
- `CONTRIBUTORS.md` file
- GitHub contributors page

## Getting Help

- **Documentation**: <https://OuyangWenyu.github.io/hydromodel>
- **Issues**: <https://github.com/OuyangWenyu/hydromodel/issues>
- **Discussions**: <https://github.com/OuyangWenyu/hydromodel/discussions>

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of:
- Age, body size, disability
- Ethnicity, gender identity and expression
- Level of experience
- Nationality, personal appearance
- Race, religion, sexual identity and orientation

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior:**
- Use of sexualized language or imagery
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information

### Enforcement

Project maintainers have the right to remove, edit, or reject contributions that do not align with this Code of Conduct.

## Questions?

Don't hesitate to ask! We're here to help:

1. Check existing [issues](https://github.com/OuyangWenyu/hydromodel/issues)
2. Search [discussions](https://github.com/OuyangWenyu/hydromodel/discussions)
3. Open a new issue with the `question` label
4. Email: wenyuouyang@outlook.com

---

Thank you for contributing to hydromodel! 🎉
