# FailMapper Framework - Failure-Aware Monte Carlo Bug Detection Architecture

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

FailMapper (Failure-Aware Monte Carlo Bug Detection Architecture) is an intelligent Java code analysis and test generation framework. This framework combines static code analysis, failure scenario model extraction, and Monte Carlo Tree Search (MCTS)-based test generation techniques, specifically designed for detecting bugs in Java code and generating high-quality unit tests.

## Core Features

### 🔍 **Multi-dimensional Static Analysis**
- **Data Flow Graph Analysis** - Track variable lifecycle and data flow direction
- **Dependency Relationship Analysis** - Analyze inter-class dependencies and method call relationships
- **Indirect Dependency Analysis** - Identify deep-level code dependencies
- **Boundary Condition Detection** - Automatically identify potential boundary value issues

### 🧠 **Intelligent Failure Scenario Modeling**
- **Model Extraction** - Extract failure scenario causes from source code
- **Failure Scenarios Detection** - Identify common failure scenarios
- **Conditional Branch Analysis** - Deep analysis of conditional statements and branch
- **Exception Handling Analysis** - Check completeness of exception handling

### 🎯 **Enhanced Test Generation**
- **Failure-Aware MCTS** - Intelligent test generation based on failure scenario models
- **Adaptive Test Strategy** - Adjust test strategies based on code characteristics
- **Coverage-Driven Optimization** - Intelligently improve code coverage
- **Bug-Driven Testing** - Prioritize generating test cases that can detect potential errors

### 🔬 **Intelligent Error Verification**
- **LLM-Assisted Verification** - Use large language models to verify detected errors
- **Multi-mode Verification** - Support immediate verification, batch verification, and other modes
- **Error Pattern Classification** - Automatic classification and prioritization of errors

## Project Architecture

```
LAMBDA Framework
├── Static Analysis Layer
│   ├── file_analyzer.py          # File parsing and AST analysis
│   ├── dependency_analyzer.py    # Dependency relationship analysis
│   ├── data_flow_analyzer.py     # Data flow analysis
│   └── boundary_exception_analyzer.py # Boundary condition and exception analysis
│
├── Modeling Layer
│   ├── model_extractor.py        # Failure scenario model extraction
│   ├── failure_scenario.py       # Failure scenario detection
│   ├── business_logic_analyzer.py # Business logic analysis
│   └── semantic_analyzer.py      # Semantic analysis
│
├── Test Generation Layer
│   ├── fa_mcts.py                # Failure-aware MCTS algorithm
│   ├── enhanced_mcts_test_generator.py # Adapted MCTS generator
│   ├── test_state.py             # Test state management
│   └── enhanced_test_state.py    # Enhanced test state
│
├── Verification & Feedback Layer
│   ├── bug_verifier.py           # Bug verifier
│   ├── verify_bug_with_llm.py    # LLM-assisted verification
│   ├── feedback.py               # Feedback mechanism
│   └── validation_engine.py      # validation engine
│
└── Framework Entry
    ├── run.py                    # Main runtime interface
    ├── main.py                   # Static analysis entry point
    ├── prompt_generator.py       # Test prompt generation for the root node
    └── failmapper_framework.py   # FailMapper framework core
```

## Installation Requirements

### Python Dependencies
```bash
python >= 3.7
```

### Required Python Packages
```bash
pip install javalang
pip install openai
pip install anthropic
pip install beautifulsoup4
pip install lxml
pip install requests
```

### Java Environment
- Java 8 or higher
- Maven (for Java project building and testing)

## Quick Start

### 1. Basic Usage

Analyze a single Java class and generate tests:

```bash
python run.py /path/to/java/project \
    --output_dir ./results \
    --class_name YourClassName \
    --package com.example.package
```

### 2. Step-by-step Usage

#### Step 1: Static Analysis
```bash
python main.py /path/to/java/project --output_dir ./analysis_results
```

#### Step 2: Generate Test Prompts
```bash
python prompt_generator.py ./analysis_results/project_name/project_name_combined_analysis.json \
    --output_dir ./analysis_results/project_name/prompts
```

#### Step 3: Run LAMBDA Framework
```bash
python failmapper_framework.py \
    --project /path/to/java/project \
    --prompt ./analysis_results/project_name/prompts \
    --class YourClassName \
    --package com.example.package
```

## Detailed Configuration Options

### run.py Parameters

| Parameter | Description | Required | Default |
|------|------|------|--------|
| `project_path` | Java project path | ✅ | - |
| `--output_dir` | Output directory | ✅ | - |
| `--class_name` | Target class name | ✅ | - |
| `--package` | Target package name | ✅ | - |

### lambda_framework.py Advanced Parameters

| Parameter | Description | Default |
|------|------|--------|
| `--max_iterations` | MCTS maximum iterations | 20 |
| `--target_coverage` | Target coverage (%) | 100.0 |
| `--verify_mode` | Verification mode (immediate/batch/none) | batch |
| `--failure_weight` | Failure awareness weight | 2.0 |
| `--bugs_threshold` | Bug threshold | 15 |
| `--verbose` | Verbose output | False |

## Output Description

### Static Analysis Results
- `{project_name}_dfg.json` - Data flow graph analysis results
- `{project_name}_dependency.json` - Dependency relationship analysis results
- `{project_name}_IDC.json` - Indirect dependency analysis results
- `{project_name}_combined_analysis.json` - Combined analysis results

### Test Generation Results
- `prompts/` - Generated test prompt files
- `generated_tests/` - Generated test code
- `bug_reports/` - Discovered error reports
- `coverage_reports/` - Coverage reports

## Core Algorithms

### 1. Failure-Aware MCTS
```python
# Core algorithm concept
reward = base_reward + bug_weight * bug_score
```
- Adjust search strategy based on models
- Prioritize exploring code paths that may contain errors
- Dynamically adjust test generation direction

### 2. Failure Scenario Detection
Supported error patterns:
- Conditional Errors
- Boundary Value Errors
- Null Pointer References
- Loop Errors
- Missing Exception Handling

## Usage Examples

### Example 1: Analyze Spring Boot Project
```bash
python run.py ~/projects/spring-boot-app \
    --output_dir ./spring_analysis \
    --class_name UserService \
    --package com.example.service
```

### Example 2: Batch Processing Mode
```bash
python failmapper_framework.py \
    --project ~/projects/my-java-app \
    --prompt ./prompts \
    --batch \
    --max_iterations 50 \
    --bug_weight 3.0
```

## Performance Optimization Recommendations

### 1. Memory Optimization
- For large projects, recommend increasing JVM heap memory
- Use `--verbose` parameter to monitor memory usage

### 2. Time Optimization
- Adjust `--max_iterations` parameter to balance quality and speed
- Use `--target_coverage` to set reasonable coverage targets

### 3. Quality Optimization
- Increase `--bug_weight` parameter to focus on error detection
- Use `verify_mode=immediate` for real-time verification

## Troubleshooting

### Common Issues

1. **Java Parsing Failed**
   ```
   Solution: Check Java syntax or exclude using module-info.java
   ```

2. **Out of Memory**
   ```bash
   export JAVA_OPTS="-Xmx4g"
   ```

3. **Dependency Analysis Failed**
   ```
   Ensure project structure is complete and dependencies are correct
   ```

## Contributing Guidelines

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Contact

For questions or suggestions, please contact us through:
- Create a [GitHub Issue](issues)
- Send email to the development team

## Acknowledgments

Thanks to all developers and researchers who have contributed to this project.

---

**LAMBDA Framework** - Making Java code analysis smarter and error detection more precise! 