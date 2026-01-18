# Contributing to Vanguard-Alpha

Thank you for your interest in contributing to Vanguard-Alpha! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Pull Request Process](#pull-request-process)
8. [Issue Guidelines](#issue-guidelines)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

### Our Standards

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other contributors

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of trading concepts
- Familiarity with Python libraries (pandas, numpy)

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/Vanguard-Alpha.git
cd Vanguard-Alpha

# Add upstream remote
git remote add upstream https://github.com/hab2025/Vanguard-Alpha.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

### 4. Verify Installation

```bash
python -c "from main import VanguardAlpha; print('Setup successful!')"
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Examples**: Create usage examples
5. **Tests**: Add or improve test coverage
6. **Performance**: Optimize existing code
7. **Refactoring**: Improve code quality

### Contribution Workflow

1. **Find or Create an Issue**
   - Check existing issues
   - Create new issue if needed
   - Discuss approach before major changes

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make Changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation

4. **Test Your Changes**
   ```bash
   pytest tests/
   python examples.py  # Run examples
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in the template
   - Link related issues

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear, descriptive names
def calculate_position_size(entry_price: float, stop_loss: float) -> int:
    """Calculate optimal position size based on risk."""
    risk_amount = self.capital * RISK_PER_TRADE
    return int(risk_amount / (entry_price - stop_loss))

# Bad: Unclear names, no types
def calc(p, s):
    r = self.c * 0.02
    return int(r / (p - s))
```

### Naming Conventions

- **Classes**: PascalCase (`TradingEngine`, `RiskEngine`)
- **Functions**: snake_case (`calculate_sharpe_ratio`, `fetch_data`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_CAPITAL`, `RISK_PER_TRADE`)
- **Private**: Prefix with underscore (`_internal_method`)

### Documentation

All public functions must have docstrings:

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio for a series of returns.
    
    The Sharpe Ratio measures risk-adjusted returns by comparing
    excess returns to volatility.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio value (annualized)
        
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
    excess_returns = returns - (risk_free_rate / 252)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
```

### Type Hints

Use type hints for all function parameters and returns:

```python
from typing import List, Dict, Optional, Tuple

def analyze_trades(trades: List[Dict]) -> Tuple[float, float]:
    """Analyze list of trades and return metrics."""
    pass

def get_position(symbol: str) -> Optional[Dict]:
    """Get position for symbol, or None if not found."""
    pass
```

### Error Handling

```python
# Good: Specific exceptions with context
try:
    data = fetch_market_data(symbol)
except ConnectionError as e:
    logger.error(f"Failed to fetch data for {symbol}: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid symbol {symbol}: {e}")
    return None

# Bad: Bare except
try:
    data = fetch_market_data(symbol)
except:
    pass
```

### Logging

```python
# Good: Appropriate log levels with context
logger.info(f"Starting backtest for {symbol}")
logger.debug(f"Fetched {len(data)} records")
logger.warning(f"High volatility detected: {volatility:.2%}")
logger.error(f"Trade execution failed: {error}")

# Bad: Print statements
print("Starting backtest")
print(data)
```

## Testing Guidelines

### Writing Tests

```python
import pytest
from trading_engine import TradingEngine

def test_signal_generation():
    """Test that signal generation works correctly."""
    engine = TradingEngine(initial_capital=100000)
    
    # Test buy signal
    signal = engine.generate_signal(symbol='AAPL', ...)
    assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal['confidence'] <= 1

def test_position_sizing():
    """Test position sizing calculation."""
    from risk_engine import RiskEngine
    
    engine = RiskEngine(initial_capital=100000)
    size = engine.calculate_position_size(
        entry_price=150,
        stop_loss_price=145
    )
    
    assert size > 0
    assert size * 150 <= 100000 * 0.1  # Max 10% position
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_trading_engine.py

# Run specific test
pytest tests/test_trading_engine.py::test_signal_generation
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Mock external dependencies (APIs, databases)

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing

## Related Issues
Fixes #123
```

### Review Process

1. Automated checks run (linting, tests)
2. Code review by maintainers
3. Address feedback
4. Approval and merge

## Issue Guidelines

### Bug Reports

```markdown
**Describe the bug**
Clear description of the issue

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.9.0]
- Vanguard-Alpha version: [e.g., 1.0.0]

**Additional context**
Any other relevant information
```

### Feature Requests

```markdown
**Feature Description**
Clear description of proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches considered

**Additional Context**
Any other relevant information
```

## Development Tips

### Useful Commands

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .

# Run examples
python examples.py

# Generate documentation
pdoc --html --output-dir docs .
```

### Debugging

```python
# Use logging instead of print
logger.debug(f"Variable value: {var}")

# Use breakpoints
import pdb; pdb.set_trace()

# Or use ipdb for better interface
import ipdb; ipdb.set_trace()
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Areas for Contribution

### High Priority

1. **Test Coverage**: Increase test coverage to >90%
2. **Documentation**: Improve API documentation
3. **Examples**: Add more usage examples
4. **Performance**: Optimize data processing

### Medium Priority

1. **Multi-Asset Support**: Portfolio-level features
2. **Real-time Data**: WebSocket integration
3. **Advanced Strategies**: More built-in strategies
4. **Web Dashboard**: Interactive UI

### Low Priority

1. **Additional Indicators**: More technical indicators
2. **Export Formats**: Additional output formats
3. **Internationalization**: Multi-language support
4. **Mobile App**: Mobile interface

## Getting Help

### Resources

- **Documentation**: Check README.md and ARCHITECTURE.md
- **Examples**: Review examples.py
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions

### Contact

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions
- **Email**: For security issues

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Vanguard-Alpha! 🚀
