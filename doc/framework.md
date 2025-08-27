# FailMapper Framework: Failure-Aware Monte Carlo Tree Search for Unit Test Generation

## Abstract

This paper presents LAMBDA (Failure-Aware Monte Carlo Bug Detection Architecture), an automated tool specialized in unit test generation and bug detection for Java projects. The framework integrates static code analysis, failure model extraction, and an enhanced Monte Carlo Tree Search (MCTS) algorithm to automatically generate high-quality unit tests, especially optimized for detecting bugs.

## 1. System Architecture Overview

### 1.1 Overall Architecture Design

LAMBDA adopts a three-stage pipeline architecture, as shown below:

```
Input: Java Project → [Stage 1: Static Analysis] → [Stage 2: Prompt Generation] → [Stage 3: Failure-Aware Test Generation] → Output: Test Cases
```

**Core Components:**
1. **Static Analysis Engine**: Extracts project structure, dependencies, and data flow information
2. **Prompt Generator**: Converts static analysis results into structured prompts for test generation
3. **Failure-Aware MCTS Engine**: Intelligent test generation and defect detection based on failure models

### 1.2 Technology Stack

- **Primary Language**: Python 3.8+
- **Java Code Parsing**: javalang library + regex fallback mechanism
- **AI Model Integration**: Supports Anthropic Claude, OpenAI GPT, DeepSeek, and others
- **Build Tool Support**: Maven and Gradle projects
- **Coverage Tool**: JaCoCo integration

## 2. Stage One: Static Analysis Engine

### 2.1 Multi-layer Parsing Strategy

The static analysis engine employs a progressive parsing strategy to handle complex Java code:

```python
def analyze_java_file(file_path):
    """
    Three-layer parsing strategy:
    1. Standard javalang parsing (preferred)
    2. Preprocessed javalang parsing (handles complex syntax)
    3. Regex fallback parsing (last resort)
    """
    try:
        # Layer 1: Direct javalang parsing
        result = parse_with_javalang(content)
    except Exception:
        try:
            # Layer 2: Parsing after preprocessing
            preprocessed = preprocess_java_content(content)
            result = parse_with_javalang(preprocessed)
        except Exception:
            # Layer 3: Regex fallback
            result = extract_basic_info_with_regex(content, file_path)
```

**Preprocessing rules include:**
- Simplify complex generics: `<List<Map<String, Object>>> → <...>`
- Simplify lambda expressions: `→ { complex body } → -> {...}`
- Normalize method references: `::specificMethod → ::method`

### 2.2 Data Flow Graph Construction

The system extracts a Data Flow Graph (DFG) for each method with the following key information:

```json
{
  "data_flow_graph": {
    "methodName": [
      {"type": "assignment", "from": "variable", "to": "expression", "line": 15},
      {"type": "condition", "from": "variable", "to": "condition_expr", "line": 18},
      {"type": "return", "details": "return_value", "line": 22}
    ]
  }
}
```

**Data flow node types:**
- `assignment`: Variable assignment
- `condition`: Conditional checks (if, while, for, etc.)
- `throw`: Exception throwing
- `return`: Return statement
- `method_call`: Method invocation

### 2.3 Dependency Analysis

#### 2.3.1 Direct Dependency Analysis
Analyze direct invocation relationships between internal classes of the project:

```python
def analyze_java_project(project_path):
    dependencies = {
        "testable_units": {},  # Detailed information of testable units
        "dependencies": [],    # List of direct dependencies
        "inheritance_map": {}, # Inheritance mapping
        "call_graph": {}       # Method call graph
    }
```

#### 2.3.2 Indirect Dependency Analysis
Use an enhanced dependency analyzer to trace indirect dependencies:

```python
class EnhancedJavaDependencyAnalyzer:
    def analyze(self):
        """
        Analysis steps:
        1. Parse all Java files
        2. Build symbol tables (classes, methods, fields)
        3. Analyze import statements and type references
        4. Build the transitive dependency graph
        """
```

### 2.4 Boundary Conditions and Exception Handling Analysis

Identify boundary conditions and exception handling patterns in code:

```python
boundary_patterns = [
    r'if\s*\([^)]*[<>=!]+\s*0\s*\)',     # Comparisons with zero
    r'if\s*\([^)]*\.length\s*[<>=!]',    # Length checks
    r'if\s*\([^)]*\.isEmpty\(\)',        # Empty checks
    r'if\s*\([^)]*null\s*[!=]',           # Null checks
]
```

## 3. Stage Two: Intelligent Prompt Generation

### 3.1 Structured Prompt Design

The prompt generator converts static analysis results into highly structured prompts for test generation:

```
===============================
JAVA CLASS UNIT TEST GENERATION
===============================

CRITICAL TESTING REQUIREMENTS:
1. DO NOT use any mocking frameworks (Mockito, EasyMock, PowerMock, etc.)
2. Use only real objects and direct instantiation for testing
3. Focus on testing actual behavior with real object interactions

-----------
1. STRUCTURE
-----------
[Class structure details]

--------------------  
2. DATA FLOW SUMMARY
--------------------
[Data flow, boundary conditions, exception handling details]

-------------
3. DEPENDENCIES
-------------
[Dependency relationships and API reference details]
```

### 3.2 Dependency API Resolution and Injection

The system intelligently resolves type dependencies and injects relevant API information:

```python
def _resolve_dep_fqns(import_types, package, testable_units, indirect_deps):
    """
    Dependency resolution priority:
    1. Already FQN and exists in testable_units
    2. simpleName matches indirect_deps  
    3. simpleName has a unique match in testable_units
    """
    resolved = set()
    for t in import_types:
        candidates = simple_to_fqns.get(t, [])
        if indirect_deps_for_class:
            matched = [c for c in candidates if c in indirect_deps_for_class]
            if matched:
                resolved.update(matched)
    return resolved
```

### 3.3 Anti-Mock Design Philosophy

LAMBDA explicitly forbids the use of mocking frameworks and enforces testing with real objects because:

1. **Authenticity**: Mocks may conceal real integration issues
2. **Integrity**: Real object tests can expose more defects
3. **Behavioral Accuracy**: Avoids inconsistency between mock behavior and real implementation

## 4. Stage Three: Logic-Aware MCTS Test Generation

### 4.1 Logical Model Extraction

#### 4.1.1 Core Functions of Extractor

```python
class Extractor:
    def __init__(self, source_code, class_name, package_name):
        # Core failure scenario components
        self.boundary_conditions = []    # Boundary conditions
        self.operations = []     # Logical operations
        self.control_flow_paths = []     # Control flow paths
        self.data_dependencies = []      # Data dependencies
        self.decision_points = []        # Decision points
        self.nested_conditions = []      # Nested conditions
```

#### 4.1.2 Logical Element Identification

The system automatically identifies the following logical elements:

**Boundary condition patterns:**
```python
boundary_patterns = [
    r'if\s*\([^)]*[<>=!]+\s*[01]\s*\)',           # Comparisons with 0/1
    r'if\s*\([^)]*\.size\(\)\s*[<>=!]',          # Collection size checks
    r'if\s*\([^)]*\.length\s*[<>=!]',              # Array length checks
    r'if\s*\([^)]*\b(MIN|MAX)_VALUE\b',            # Min/Max value checks
]
```

**Logical operation patterns:**
```python  
logical_patterns = [
    r'if\s*\([^)]*&&[^)]*\)',                      # AND operations
    r'if\s*\([^)]*\|\|[^)]*\)',                  # OR operations
    r'if\s*\(\s*!\s*[^)]+\)',                    # NOT operations
    r'[^=!<>]==[^=]',                                 # Equality comparison
]
```

### 4.2 Logical Defect Pattern Detection

#### 4.2.1 FS_Detector

```python
class FS_Detector:
    """Detects common logical defect patterns"""
    
    def detect_patterns(self):
        patterns = []
        patterns.extend(self._detect_off_by_one_errors())      # Off-by-one errors
        patterns.extend(self._detect_boundary_logic_errors())   # Boundary logic errors
        patterns.extend(self._detect_condition_logic_errors())  # Conditional logic errors
        patterns.extend(self._detect_null_pointer_risks())      # Null pointer risks
        patterns.extend(self._detect_arithmetic_overflow())     # Arithmetic overflow
        return patterns
```

#### 4.2.2 Common Types of Logical Defects

| Defect Type | Detection Pattern | Risk Level |
|---------|---------|----------|
| Off-by-One | `for(i=0; i<=array.length)` | HIGH |
| Boundary Logic Error | `if (x > 0 && x < 10)` vs `if (x >= 0 && x <= 10)` | MEDIUM |
| Conditional Logic Error | `if (a && b || c)` precedence issue | HIGH |
| Null Pointer Risk | Use before null check | MEDIUM |
| Arithmetic Overflow | `int result = a * b` without overflow check | LOW |

### 4.3 Failure-Aware MCTS Algorithm

#### 4.3.1 Enhanced State Representation

```python
class FATestState(TestState):
    def __init__(self, test_code, f_model, failures):
        super().__init__(test_code)
        self.f_model = f_model              # Failure model
        self.failures = failures        # Detected failure scenarios
        self.boundary_coverage = 0.0                # Boundary coverage
        self.logic_scenario_coverage = 0.0           # Logical scenario coverage
        self.verified_bugs = []             # Verified defects
```

#### 4.3.2 Failure-Aware Reward Function

The MCTS algorithm uses a multi-dimensional reward function:

```python
def calculate_reward(self, state):
    """
    Reward = α × code coverage + β × Failure Scenario Coverage + γ × bug discovery reward δ
    """
    base_reward = state.coverage / 100.0                    # Base coverage reward
    logic_reward = self.f_weight * (
        state.boundary_coverage * 0.4 +                    # Boundary condition coverage
        state.logic_scenario_coverage * 0.3 +               # Logical pattern coverage  
        len(state.verified_logical_bugs) * 0.3             # Logical defect discovery
    )
    
    total_reward = base_reward + logic_reward
    return min(total_reward, 10.0)  # Reward cap
```

#### 4.3.3 Logic-Guided Strategy Selection

```python
class TestStrategySelector:
    """Selects testing strategies based on logical patterns"""
    
    def select_strategy(self, current_state, iteration):
        """
        Strategy selection priority:
        1. High-risk logical patterns → Boundary testing strategy
        2. Uncovered conditional branches → Condition coverage strategy  
        3. Complex logical operations → Logical combination strategy
        4. Data dependency chains → Data flow testing strategy
        """
```

### 4.4 Dynamic Defect Verification

#### 4.4.1 BugVerifier

```python
class BugVerifier:
    """Uses LLM to verify logical defects"""
    
    def verify_logical_bug(self, test_method, source_code, error_details):
        """
        Verification steps:
        1. Analyze the root cause of the test failure
        2. Check whether it is a real logical defect
        3. Classify the defect type and severity
        4. Generate a detailed defect report
        """
        verification_prompt = f"""
        Analyze whether the following test failure reveals a real logical defect:
        
        Source code: {source_code}
        Failed test: {test_method}  
        Error details: {error_details}
        
        Please determine:
        1. Is this a real logical error?
        2. Defect type: [Boundary Condition / Conditional Logic / Arithmetic Operation / State Management / Other]
        3. Severity: [Low / Medium / High / Critical]
        4. Root cause analysis
        """
```

#### 4.4.2 Defect Classification System

| Category | Subtype | Typical Example |
|---------|--------|----------|
| Boundary Condition Error | Off-by-one, Missing boundary checks | `array[i]` where `i = array.length` |
| Conditional Logic Error | Wrong logical operator, precedence issue | `if(a && b || c && d)` |
| State Management Error | Inconsistent state, race condition | Concurrent state update conflicts |
| Arithmetic Operation Error | Overflow, divide-by-zero, precision loss | `int result = Integer.MAX_VALUE + 1` |
| Data Processing Error | Null pointer, type casting, encoding issues | Calling methods without null checks |

## 5. Experimental Validation and Evaluation Metrics

### 5.1 Evaluation Metrics System

**Primary metrics:**
- **Defect Detection Rate**: Number of real defects found / Actual defects in project
- **Logical Defect Detection Rate**: Number of logical defects found / Actual logical defects  
- **False Positive Rate**: Number of false positives / Total reported defects
- **Code Coverage**: Statement coverage, branch coverage, path coverage

**Efficiency metrics:**
- **Generation Efficiency**: Test generation time / Number of target classes
- **LLM Call Efficiency**: Total token consumption, number of API calls
- **Convergence Speed**: Number of MCTS iterations required to reach target coverage

### 5.2 Benchmark Datasets

Experiments use the following standard defect datasets:

1. **Defects4J**: 357 real Java defects
2. **Bears**: 251 defects collected from GitHub  
3. **Proprietary dataset**: Logical defect samples extracted from open-source projects

### 5.3 Baselines for Comparison

- **Traditional MCTS**: Standard MCTS without logic awareness
- **Random Test Generation**: Strategy based on random generation
- **EvoSuite**: Evolutionary algorithm test generation tool
- **Randoop**: Feedback-directed random testing

## 6. Key Innovations

### 6.1 Logic-Aware Test Generation

Traditional test generation focuses mainly on structural coverage. LAMBDA introduces the concept of logic awareness for the first time:

1. **Logic Model-Driven**: Test generation based on code logical structure rather than just syntax
2. **Defect Pattern-Oriented**: Targeted tests for known logical defect patterns  
3. **Multi-dimensional Reward Mechanism**: Balances coverage and defect discovery capability

### 6.2 Progressive Parsing Strategy

Addresses the lack of robustness in existing Java code parsing tools:

1. **Three-layer fallback**: Standard parsing → Preprocessed parsing → Regex fallback
2. **Syntax compatibility**: Supports Java 8-21 language features
3. **Fault tolerance**: Overall analysis is resilient even if some files fail to parse

### 6.3 Anti-Mock Testing Design

Unlike mainstream test generation tools, LAMBDA insists on using real objects for testing:

1. **Realistic integration testing**: Reveals integration issues that mocks cannot cover
2. **Behavioral consistency**: Ensures test behavior aligns with production behavior
3. **Enhanced defect discovery**: Real object interactions reveal more potential issues

## 7. Implementation Details

### 7.1 System Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Static Analysis │───→│   Prompt Builder  │───→│ Logic-Aware MCTS │
│                 │    │                  │    │                 │
│ • file_analyzer │    │ • prompt_generator│    │ • f_model   │
│ • dependency    │    │ • api_resolver    │    │ • pattern_detect│
│ • data_flow     │    │ • struct_format   │    │ • mcts_search   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 7.2 Key Algorithm Implementations

**Core MCTS loop:**
```python  
def run_search(self):
    for iteration in range(self.max_iterations):
        # 1. Selection (with logic bias)
        leaf_node = self.select_with_logic_bias(self.root)
        
        # 2. Expansion (logic-guided strategy)
        child_node = self.expand_with_logic_strategy(leaf_node)
        
        # 3. Simulation (logic-aware reward)
        reward = self.simulate_with_logic_reward(child_node)
        
        # 4. Backpropagation (update logic metrics)
        self.backpropagate_logic_aware(child_node, reward)
        
        # 5. Defect verification (if potential defect is detected)
        if self.detect_potential_bug(child_node):
            self.verify_bug_with_llm(child_node)
```

### 7.3 Extensibility Design

**Plugin-based architecture:**
- **Analyzer plugins**: Easily add new static analyzers
- **Strategy plugins**: Support custom test generation strategies
- **Verifier plugins**: Support different defect verification methods
- **LLM adapters**: Support multiple large language model backends

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Dependence on LLM quality**: Test generation and defect verification rely on LLM capabilities
2. **Java language constraint**: Currently supports only Java projects
3. **Computational overhead**: MCTS search incurs higher cost than simple random generation
4. **Scalability for large projects**: May require further optimization for very large projects

### 8.2 Future Research Directions

1. **Multi-language support**: Extend to C++, Python, C#, and others
2. **Stronger logical reasoning**: Integrate symbolic execution and constraint solving
3. **Adaptive learning**: Automatically tune search strategies based on project characteristics
4. **Distributed parallelism**: Support parallel test generation for large-scale projects

## 9. Conclusion

The LAMBDA framework achieves automated unit test generation targeting logical defects through a logic-aware MCTS algorithm. Its main contributions include:

1. **Innovative logic-aware test generation**: Integrates logical models and defect patterns into MCTS for the first time
2. **Robust static analysis pipeline**: Addresses engineering challenges in Java code parsing  
3. **Real object testing philosophy**: Avoids limitations of mock-based testing
4. **Comprehensive defect detection and verification**: Combines pattern detection with LLM verification to improve accuracy

Experimental results show that the LAMBDA framework significantly improves logical defect detection compared with traditional approaches, providing new research directions and practical tools for automated test generation.

---

*Note: This document is based on LAMBDA framework v1.0. For detailed source code implementations, please refer to the project repository.*