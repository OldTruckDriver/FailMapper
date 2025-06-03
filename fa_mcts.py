#!/usr/bin/env python3
"""
Logic-Aware Monte Carlo Tree Search

This module implements a Logic-Aware Monte Carlo Tree Search (MCTS) algorithm
for generating tests that specifically target logical vulnerabilities in Java code.
The algorithm enhances traditional MCTS with logic-aware components to improve
the detection of logical bugs.
"""

import os
import re
import time
import json
import logging
import random
import traceback
import numpy as np
from collections import defaultdict
from datetime import datetime

# Import from base MCTS implementation
from enhanced_mcts_test_generator import (
    EnhancedMCTSTestGenerator, 
    TestValidator, 
    TestMethodExtractor,
    handle_false_positive_tests
)

# Import from other LAMBDA components
from logic_test_state import LogicAwareTestState

# Import from feedback module
from feedback import (
    call_anthropic_api, call_gpt_api, call_deepseek_api, extract_java_code,
    run_tests_with_jacoco, save_test_code, get_coverage_percentage,
    get_class_uncovered_details, strip_java_comments, run_maven_command,
    find_source_code, read_source_code, generate_test_summary,
    reset_llm_metrics, get_llm_metrics_summary,
    read_test_prompt_file
)

# Import from verify_bug_with_llm module
from verify_bug_with_llm import (
    verify_bug_with_llm, filter_verified_bug_methods, merge_verified_bug_tests, attempt_to_fix_test_expectations
)

# Import from logic_bug_verifier module
from logic_bug_verifier import LogicBugVerifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logic_aware_mcts")

class LogicAwareMCTSNode:
    """
    Node in the Logic-Aware MCTS tree
    
    Enhanced with logic-specific rewards and heuristics to improve
    the detection of logical vulnerabilities.
    """
    
    def __init__(self, state, parent=None, action=None):
        """
        Initialize node with state and parent
        
        Parameters:
        state (LogicAwareTestState): Current test state
        parent (LogicAwareMCTSNode): Parent node (None for root)
        action (str): Action taken to reach this state
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0.0
        self.visits = 0
        
        # Logic-specific rewards
        self.logic_bug_rewards = 0.0  # Additional rewards for finding logical bugs
        self.logic_coverage_rewards = 0.0  # Rewards for covering logical constructs
        self.high_risk_pattern_rewards = 0.0  # Rewards for covering high-risk patterns
        
        # Logic-specific metrics
        self.logical_bugs_found = 0
        self.covered_patterns = set()
        self.covered_branch_conditions = set()
        
        # Track bug types found by this node and its children
        self.bug_types_found = set()
        
        # Track whether this node found a new test path or behavior
        self.is_novel = False
        self.expanded = False
        self.used_action = []
    
    def has_compilation_errors(self):
        """
        Check if the current state has compilation errors
        
        Returns:
        bool: True if there are compilation errors, False otherwise
        """
        return (self.state and 
                hasattr(self.state, "compilation_errors") and 
                self.state.compilation_errors)
    
    def generate_possible_actions(self, test_prompt, source_code, uncovered_data=None, 
                               logic_model=None, logic_patterns=None, strategy_selector=None):
        """
        Generate possible actions (test methods) from current state
        
        Parameters:
        test_prompt (str): Test generation prompt
        source_code (str): Source code being tested
        uncovered_data (dict): Information about uncovered code
        logic_model (LogicModelExtractor): Logical model
        logic_patterns (list): Detected logic bug patterns
        strategy_selector (LogicTestStrategySelector): Strategy selector
        
        Returns:
        list: List of possible actions
        """
        possible_actions = []
        business_logic_issues = []
        
        # Check for compilation errors and prioritize fixing them
        if self.has_compilation_errors():
            # If we have compilation errors, only return the fix_compilation_errors action
            logger.info("Detected compilation errors, prioritizing compilation error fixing")
            action = {
                "type": "fix_compilation_errors",
                "description": "Fix compilation errors in test code",
                "errors": self.state.compilation_errors  # Include a few errors for context
            }

            possible_actions.append(action)
            return possible_actions

        if hasattr(self.state, 'business_logic_analysis') and self.state.business_logic_analysis:
            business_logic_issues = self.state.business_logic_analysis.get('potential_bugs', [])
            # print("--------------------------------")
            # print("business_logic_issues in generate_possible_actions:")
            # print(business_logic_issues)
            # print("--------------------------------")
            

        # Get strategies from selector based on current state
        if strategy_selector:
            strategies = strategy_selector.select_strategies(
                self.state, 
                self.covered_patterns, 
                self.covered_branch_conditions,
                business_logic_issues
            )
        else:
            # Default strategy if no selector provided
            strategies = [
                {"id": "boundary_testing", "name": "Boundary Value Testing", "weight": 1.0},
                {"id": "logical_expression", "name": "Logical Expression Testing", "weight": 1.0},
                {"id": "exception_handling", "name": "Exception Path Testing", "weight": 0.7}
            ]
        
        for issue in business_logic_issues:
            print("--------------------------------")
            print("issue in generate_possible_actions:")
            print(issue)
            print("--------------------------------")
            
            action = {
                "type": "business_logic_test",
                "issue_type": issue.get('type', 'unknown'),
                "method": issue.get('method', ''),
                "description": f"Test for potential business logic issue: {issue.get('description', '')}",
                "confidence": issue.get('confidence', 0),
                "business_logic": True  # Flag to identify these special actions
            }
            possible_actions.append(action)

        # Get currently uncovered code if available
        uncovered_lines = []
        if uncovered_data and "uncovered_lines" in uncovered_data:
            uncovered_lines = uncovered_data["uncovered_lines"]
        
        # Include state-specific actions targeting interesting lines
        if uncovered_lines:
            # Select some random uncovered lines to focus on
            selected_lines = random.sample(
                uncovered_lines, 
                min(5, len(uncovered_lines))
            )
            
            for line_info in selected_lines:
                line_num = line_info.get("line", 0)
                content = line_info.get("content", "").strip()
                
                # Skip empty or irrelevant lines
                if not content or content in ["}", "{", "//", "/*", "*/"]:
                    continue
                
                # Create targeted action for this line
                line_action = {
                    "type": "target_line",
                    "line": line_num,
                    "content": content,
                    "description": f"Target uncovered line {line_num}: {content[:40]}..."
                }
                possible_actions.append(line_action)
        
        # Add strategy-based actions
        for strategy in strategies:
            strategy_id = strategy.get("id", "unknown")
            strategy_weight = strategy.get("weight", 1.0)
            
            # Skip strategies with very low weight
            if strategy_weight < 0.1:
                continue
                
            # Add focused strategy actions
            if strategy_id == "boundary_testing" and logic_model:
                # Add boundary condition testing actions
                boundary_conditions = logic_model.boundary_conditions
                if boundary_conditions:
                    # Select a few boundary conditions to target
                    selected_conditions = random.sample(
                        boundary_conditions,
                        min(2, len(boundary_conditions))
                    )
                    
                    for condition in selected_conditions:
                        condition_str = condition.get("condition", "")
                        line_num = condition.get("line", 0)
                        
                        if not condition_str:
                            continue
                            
                        action = {
                            "type": "boundary_test",
                            "condition": condition_str,
                            "line": line_num,
                            "strategy": strategy_id,
                            "description": f"Test boundary condition at line {line_num}: {condition_str[:40]}..."
                        }
                        possible_actions.append(action)
            
            elif strategy_id == "logical_expression" and logic_model:
                # Add logical operation testing actions
                logical_operations = logic_model.logical_operations
                if logical_operations:
                    # Select a few logical operations to target
                    selected_operations = random.sample(
                        logical_operations,
                        min(2, len(logical_operations))
                    )
                    
                    for operation in selected_operations:
                        operation_str = operation.get("condition", "")
                        line_num = operation.get("line", 0)
                        
                        if not operation_str:
                            continue
                            
                        action = {
                            "type": "logical_expression_test",
                            "operation": operation_str,
                            "line": line_num,
                            "strategy": strategy_id,
                            "description": f"Test logical expression at line {line_num}: {operation_str[:40]}..."
                        }
                        possible_actions.append(action)
            
            elif strategy_id == "exception_handling":
                # Add exception handling test actions
                action = {
                    "type": "exception_test",
                    "strategy": strategy_id,
                    "description": "Generate tests for exception paths"
                }
                possible_actions.append(action)
            
            elif strategy_id == "data_validation":
                # Add data validation test actions
                action = {
                    "type": "data_validation_test",
                    "strategy": strategy_id,
                    "description": "Generate tests for data validation edge cases"
                }
                possible_actions.append(action)
            
            elif strategy_id == "resource_management":
                # Add resource management test actions
                action = {
                    "type": "resource_management_test",
                    "strategy": strategy_id,
                    "description": "Generate tests for resource management issues"
                }
                possible_actions.append(action)
            
            elif strategy_id == "state_transition":
                # Add state transition test actions
                action = {
                    "type": "state_transition_test",
                    "strategy": strategy_id, 
                    "description": "Generate tests for state transitions"
                }
                possible_actions.append(action)
        
        # Add targeted actions for specific logic bug patterns
        if logic_patterns:
            # Filter to high-risk patterns
            high_risk_patterns = [p for p in logic_patterns if p.get("risk_level") == "high"]
            if high_risk_patterns:
                # Select a few high-risk patterns to target
                selected_patterns = random.sample(
                    high_risk_patterns,
                    min(2, len(high_risk_patterns))
                )
                
                for pattern in selected_patterns:
                    pattern_type = pattern.get("type", "unknown")
                    line_num = pattern.get("location", 0)
                    description = pattern.get("description", "")
                    
                    action = {
                        "type": "bug_pattern_test",
                        "pattern_type": pattern_type,
                        "line": line_num,
                        "description": f"Test for {pattern_type} bug pattern at line {line_num}: {description[:40]}..."
                    }
                    possible_actions.append(action)
        
        # Add general exploration action
        if not possible_actions or random.random() < 0.2:  # 20% chance to add exploration
            action = {
                "type": "general_exploration",
                "description": "General test exploration"
            }
            possible_actions.append(action)
            
        # Avoid empty action list
        if not possible_actions:
            action = {
                "type": "fallback",
                "description": "Fallback test generation"
            }
            possible_actions.append(action)
        print("--------------------------------")
        print("used_action in generate_possible_actions:")
        print(self.used_action)
        print("--------------------------------")
        possible_actions = [x for x in possible_actions if x not in self.used_action]
        return possible_actions

    def is_fully_expanded(self):
        """Check if all possible child actions have been explored"""
        return self.expanded
    
    def best_child(self, exploration_weight=1.0, logic_weight=1.0):
        """
        Select best child node using UCB1 formula with logic enhancements
        
        Parameters:
        exploration_weight (float): Weight for exploration term
        logic_weight (float): Weight for logic-specific rewards
        
        Returns:
        LogicAwareMCTSNode: Best child node
        """
        if not self.children:
            return None
            
        def ucb_score(child):
            # Base UCB1 score
            exploitation = child.wins / child.visits if child.visits > 0 else 0.0
            exploration = exploration_weight * (2 * (self.visits / child.visits) ** 0.5) if child.visits > 0 else float('inf')
            
            # Logic-specific rewards with decay factor
            logic_bonus = 0.0
            if child.visits > 0:
                # Add reward for logical bugs found
                logic_bug_term = child.logic_bug_rewards / child.visits
                
                # Add reward for logic coverage
                logic_coverage_term = child.logic_coverage_rewards / child.visits
                
                # Add reward for high-risk pattern coverage
                high_risk_term = child.high_risk_pattern_rewards / child.visits
                
                # Add novelty bonus for finding new paths
                novelty_bonus = 0.2 if child.is_novel else 0.0
                
                # 添加：访问次数衰减因子 - 随着访问次数增加而减少奖励
                visits_decay = 1.0 / (1.0 + 0.1 * child.visits)
                
                # 添加：连续失败惩罚
                failure_penalty = 1.0
                if hasattr(child, 'consecutive_failures') and child.consecutive_failures > 0:
                    failure_penalty = max(0.3, 1.0 - (0.2 * child.consecutive_failures))
                
                # 添加：策略多样性奖励
                diversity_bonus = 0.0
                if hasattr(child, 'action') and hasattr(self, 'last_action_type'):
                    if isinstance(child.action, dict) and 'type' in child.action:
                        if child.action['type'] != self.last_action_type:
                            diversity_bonus = 0.15  # 奖励选择不同类型的策略
                
                # Combined logic bonus with decay and penalties
                logic_bonus = logic_weight * (
                    (logic_bug_term + logic_coverage_term + high_risk_term + novelty_bonus) * 
                    visits_decay * failure_penalty
                ) + diversity_bonus
            
            # Return combined score
            return exploitation + exploration + logic_bonus
            
        # Return child with highest UCB score
        return max(self.children, key=ucb_score)

    def add_child(self, state, action):
        """
        Add a new child node
        
        Parameters:
        state (LogicAwareTestState): New state
        action (dict): Action taken to reach the state
        
        Returns:
        LogicAwareMCTSNode: New child node
        """
        # Create new child node
        child = LogicAwareMCTSNode(state, self, action)
        
        # Add to children list
        self.children.append(child)
        
        # Check if the action leads to novel state
        if state:
            # Check if this action found new bugs
            if state.has_logical_bugs:
                child.is_novel = True
                
                # Update found bug types
                for bug in state.logical_bugs:
                    bug_type = bug.get("logic_bug_type", "unknown")
                    self.bug_types_found.add(bug_type)
                    
            # Check if this action covered new patterns
            if hasattr(state, "covered_logic_patterns") and state.covered_logic_patterns:
                new_patterns = state.covered_logic_patterns - self.covered_patterns
                if new_patterns:
                    child.is_novel = True
                    
            # Check if this action covered new branch conditions
            if hasattr(state, "covered_branch_conditions") and state.covered_branch_conditions:
                new_conditions = state.covered_branch_conditions - self.covered_branch_conditions
                if new_conditions:
                    child.is_novel = True
        
        # Return the new child node
        return child
    
    def update(self, reward, bug_type=None, pattern_coverage=None, branch_coverage=None, has_error=False):
        """
        Update node statistics after simulation
        
        Parameters:
        reward (float): Reward value
        bug_type (str): Type of bug found (optional)
        pattern_coverage (int): Number of covered patterns (optional)
        branch_coverage (int): Number of covered branch conditions (optional)
        has_error (bool): Whether there was an error in test execution (new parameter)
        """
        self.visits += 1
        self.wins += reward
        
        # 添加：跟踪连续失败
        if not hasattr(self, 'consecutive_failures'):
            self.consecutive_failures = 0
            
        # 添加：检测失败和错误，更新失败计数
        if has_error or reward < 0.1:  # 认为极低奖励也是失败的信号
            self.consecutive_failures += 1
            # 添加：对已累积的奖励进行惩罚性衰减
            self.wins *= 0.9  # 轻微衰减累积奖励
        else:
            self.consecutive_failures = 0  # 重置连续失败计数
        
        # Update coverage data if provided
        if pattern_coverage is not None and hasattr(self, 'covered_patterns'):
            self.covered_patterns = pattern_coverage
        
        if branch_coverage is not None and hasattr(self, 'covered_branch_conditions'):
            self.covered_branch_conditions = branch_coverage
        
        # Update logic-specific rewards
        if bug_type:
            if bug_type.startswith("logical_"):
                # Higher reward for logical bugs
                self.logic_bug_rewards += 1.0
                self.logical_bugs_found += 1
            elif bug_type.startswith("high_risk_"):
                # Reward for finding high-risk bugs
                self.high_risk_pattern_rewards += 0.8
                
        # Update logic coverage rewards if available from state
        if self.state:
            # Reward for covering logical patterns
            if hasattr(self.state, "covered_logic_patterns"):
                pattern_coverage = len(self.state.covered_logic_patterns) / 10.0  # Normalize
                self.logic_coverage_rewards += min(pattern_coverage, 1.0)
                
                # Update covered patterns - use actual set content from state
                if hasattr(self, 'covered_patterns'):
                    self.covered_patterns = self.state.covered_logic_patterns
                    
            # Reward for covering branch conditions
            if hasattr(self.state, "covered_branch_conditions"):
                branch_coverage = len(self.state.covered_branch_conditions) / 20.0  # Normalize
                self.logic_coverage_rewards += min(branch_coverage, 1.0)
                
                # Update covered branch conditions - use actual set content from state
                if hasattr(self, 'covered_branch_conditions'):
                    self.covered_branch_conditions = self.state.covered_branch_conditions

class LogicAwareMCTS(EnhancedMCTSTestGenerator):
    """
    Logic-Aware Monte Carlo Tree Search for test generation.
    
    Enhances the base MCTS algorithm with logic-awareness to improve
    the detection of logical vulnerabilities in Java code.
    """
    
    def __init__(self, project_dir, prompt_dir, class_name, package_name, 
            initial_test_code, source_code, test_prompt, 
            logic_model, logic_patterns, strategy_selector,
            max_iterations=20, exploration_weight=1.0,
            verify_bugs_mode="batch", focus_on_bugs=True, 
            logic_weight=2.0, initial_coverage=0.0,
            logical_bugs_threshold=100):
        """
        初始化 Logic-Aware MCTS
        
        Parameters:
        project_dir (str): 项目目录
        prompt_dir (str): 提示目录
        class_name (str): 类名
        package_name (str): 包名
        initial_test_code (str): 初始测试代码
        source_code (str): 源代码
        test_prompt (str): 测试生成提示
        logic_model (LogicModelExtractor): 逻辑模型
        logic_patterns (list): 检测到的逻辑bug模式
        strategy_selector (LogicTestStrategySelector): 策略选择器
        max_iterations (int): 最大迭代次数
        exploration_weight (float): 探索权重
        verify_bugs_mode (str): 何时验证bug (immediate/batch/none)
        focus_on_bugs (bool): 是否专注于寻找bug
        logic_weight (float): 与逻辑相关的奖励权重
        initial_coverage (float): 初始代码覆盖率
        logical_bugs_threshold (int): 终止搜索的逻辑bug数阈值
        """
        # 初始化父类
        super().__init__(
            project_dir=project_dir,
            prompt_dir=prompt_dir,
            class_name=class_name,
            package_name=package_name,
            initial_test_code=initial_test_code,
            source_code=source_code,
            test_prompt=test_prompt,
            max_iterations=max_iterations,
            exploration_weight=exploration_weight,
            verify_bugs_mode=verify_bugs_mode,
            focus_on_bugs=focus_on_bugs
        )
        
        # 设置逻辑特定属性
        self.logic_model = logic_model
        self.logic_patterns = logic_patterns
        self.strategy_selector = strategy_selector
        self.logic_weight = logic_weight
        self.logical_bugs_threshold = logical_bugs_threshold
        
        # 统计和指标
        self.logical_bugs_found = 0
        self.verified_bug_methods = []
        self.best_logic_coverage = 0.0
        self.best_pattern_coverage = 0
        self.best_branch_coverage = 0
        self.current_coverage = initial_coverage
        
        # 跟踪当前测试状态
        self.root_state = None
        self.best_state = None
        self.best_test = initial_test_code
        self.best_reward = 0.0
        
        # 跟踪算法执行历史
        self.history = []
        
        # 分支条件和风险模式到测试的映射
        self.targeted_conditions = defaultdict(list)
        self.targeted_patterns = defaultdict(list)
        
        # 用于学术评估的跟踪指标
        self.metrics = {
            "logical_bug_types_found": set(),
            "boundary_conditions_covered": 0,
            "logical_operations_covered": 0,
            "high_risk_patterns_covered": 0,
            "iterations_to_first_logical_bug": None,
            "iterations_to_high_coverage": None,
            "total_test_methods": 0,
            "total_logical_bug_tests": 0,
            "ucb_score_distribution": [],
            "strategy_effectiveness": defaultdict(lambda: {"used": 0, "bugs_found": 0})
        }
        
        # 新增: 用于推迟验证的bug收集
        self.potential_bugs = []
        self.potential_bug_signatures = set()
        self.unique_bugs = []  # 用于记录唯一的bug
        
        # 新增: 存储不同覆盖率的测试代码
        self.high_coverage_tests = {}
        
        logger.info(f"初始化 Logic-Aware MCTS，logic_weight={logic_weight}, logical_bugs_threshold={logical_bugs_threshold}")

    def process_initial_state(self, initial_state):
        """
        Process the initial test state
        
        Parameters:
        initial_state (LogicAwareTestState): Initial state
        
        Returns:
        LogicAwareTestState: Processed initial state
        """
        logger.info("Processing initial test state")
        
        # If no initial state provided, create one
        if not initial_state:
            initial_state = LogicAwareTestState(
                test_code=self.initial_test_code,
                class_name=self.class_name,
                package_name=self.package_name,
                project_dir=self.project_dir,
                source_code=self.source_code,
                logic_model=self.logic_model,
                logic_patterns=self.logic_patterns
            )
            
            # Evaluate the initial state
            initial_state.evaluate()
        
        from business_logic_analyzer import BusinessLogicAnalyzer
        analyzer = BusinessLogicAnalyzer()
        
        # Identify target methods to analyze (limit to most complex or error-prone methods)
        target_methods = self._identify_target_methods()

        # Initialize business logic analysis results
        business_logic_results = {
            "analyzed_methods": [],
            "potential_bugs": []
        }

        print("--------------------------------")
        print("Target Methods:")
        print(target_methods)
        print("--------------------------------")

        # Analyze each target method
        for method in target_methods:
            logger.info(f"Analyzing business logic for method: {method}")
            analysis = analyzer.analyze_code_for_logic_bugs(
                source_code=self.source_code,
                class_name=self.class_name,
                method_name=method
            )
            
            if "error" not in analysis:
                # print("--------------------------------")
                # print("Analyzed Method:")
                # print(method)
                # print("--------------------------------")

                # print("--------------------------------")
                # print("business_logic_results:")
                # print(business_logic_results)
                # print("--------------------------------")

                business_logic_results["analyzed_methods"].append(method)
                

                
                # Extract potential bugs with sufficient confidence
                if "potential_bugs" in analysis:
                    for bug in analysis["potential_bugs"]:
                        print("--------------------------------")
                        print("Bug:")
                        print(bug)
                        print("--------------------------------")
                        business_logic_results["potential_bugs"].append({
                            "method": method,
                            "type": bug.get("type", "unknown"),
                            "description": bug.get("description", ""),
                            "confidence": bug.get("confidence", 0.0),
                            "semantic_signals": bug.get("semantic_signals", {}),
                            "implementation_features": bug.get("implementation_features", {}),
                            "test_strategy": bug.get("test_strategy", "")
                        })
        
        
        
        
        # Log business logic analysis results
        if business_logic_results["potential_bugs"]:
            logger.info(f"Identified {len(business_logic_results['potential_bugs'])} potential business logic issues")
            for bug in business_logic_results["potential_bugs"]:
                logger.info(f"  - Method: {bug['method']}, Type: {bug['type']}, Confidence: {bug['confidence']:.2f}")

            # Store the business logic analysis in the state
            initial_state.business_logic_analysis = business_logic_results
        else:
            logger.info("No potential business logic issues identified")
        # Store as root state
        self.root_state = initial_state
        
        # Check for compilation errors in the initial state
        if hasattr(initial_state, "compilation_errors") and initial_state.compilation_errors:
            logger.warning(f"Initial state has compilation errors: {initial_state.compilation_errors[:2]}")
            # Ensure the compilation_errors attribute is properly set
            initial_state.compilation_errors = initial_state.compilation_errors
        
        # Set as current best state
        self.best_state = initial_state
        self.best_test = initial_state.test_code
        
        # Update metrics
        if initial_state.coverage > 0:
            self.current_coverage = initial_state.coverage
            
        if hasattr(initial_state, "covered_logic_patterns"):
            self.best_pattern_coverage = len(initial_state.covered_logic_patterns)
            
        if hasattr(initial_state, "covered_branch_conditions"):
            self.best_branch_coverage = len(initial_state.covered_branch_conditions)
        
        if initial_state.has_logical_bugs:
            self.logical_bugs_found = initial_state.count_logical_bugs()
            logical_bug_methods = initial_state.get_logical_bug_finding_methods()
            
            # Add to verified bug methods
            for bug_method in logical_bug_methods:
                if bug_method not in self.verified_bug_methods:
                    self.verified_bug_methods.append(bug_method)
                    
                    # Update metrics
                    bug_type = bug_method.get("bug_type", "unknown")
                    self.metrics["logical_bug_types_found"].add(bug_type)
                    self.metrics["total_logical_bug_tests"] += 1
        
        # Calculate initial reward
        initial_reward = self.calculate_logic_aware_reward(initial_state)
        self.best_reward = initial_reward
        
        logger.info(f"Initial state processed: coverage={self.current_coverage:.2f}%, " + 
                  f"logical bugs={self.logical_bugs_found}, reward={initial_reward:.4f}")
        
        return initial_state
    
    def _identify_target_methods(self):
        """
        Identify target methods for business logic analysis
        
        Returns:
        list: Method names to analyze
        """
        target_methods = []
        
        # If logic model is available, use it to identify complex methods
        if self.logic_model and hasattr(self.logic_model, 'method_complexity'):
            # Sort methods by complexity
            complex_methods = sorted(
                self.logic_model.method_complexity.items(),
                key=lambda x: x[1].get("cognitive", 0) + x[1].get("cyclomatic", 0),
                reverse=True
            )
            
            # Take top 3-5 most complex methods
            target_methods = [method for method, _ in complex_methods[:min(5, len(complex_methods))]]
        
        # Fall back to regex-based method extraction if needed
        if not target_methods:
            import re
            method_pattern = r'(?:public|private|protected)\s+(?:<.*?>)?\s*\w+\s+(\w+)\s*\([^)]*\)'
            matches = re.findall(method_pattern, self.source_code)
            
            # Filter out common utility methods
            exclude_patterns = ['equals', 'hashCode', 'toString', 'clone', 'finalize', 'main']
            target_methods = [m for m in matches if not any(ex == m for ex in exclude_patterns)]
            
            # Limit to reasonable number
            target_methods = target_methods[:min(5, len(target_methods))]
        
        # Add class name to potential methods list if it's a constructor
        if target_methods and self.class_name not in target_methods:
            target_methods.append(self.class_name)
        
        return target_methods
    
    def _analyze_business_logic(self, state, source_code=None, class_name=None):
        """
        Perform business logic analysis to enhance test generation
        
        Parameters:
        state (LogicAwareTestState): Current test state
        source_code (str): Source code (optional)
        class_name (str): Class name (optional)
        
        Returns:
        dict: Logic insights for test generation
        """
        if not hasattr(self, 'business_logic_analyzer'):
            # Import here to avoid circular imports
            from business_logic_analyzer import BusinessLogicAnalyzer
            self.business_logic_analyzer = BusinessLogicAnalyzer()
        
        # Use source_code and class_name from parameters or instance variables
        source_code = source_code or self.source_code
        class_name = class_name or self.class_name
        
        # Get target methods to analyze
        target_methods = self._get_target_methods()
        logic_insights = {}
        
        for method in target_methods:
            # Skip methods that have already been analyzed
            if hasattr(state, 'logic_insights') and method in state.logic_insights:
                logic_insights[method] = state.logic_insights[method]
                continue
            
            # Analyze method's business logic
            try:
                logic_analysis = self.business_logic_analyzer.analyze_code_for_logic_bugs(
                    source_code, class_name, method)
                
                # Only add results with sufficient confidence
                if logic_analysis.get("confidence", 0.0) >= 0.7:
                    logic_insights[method] = logic_analysis
                    logger.info(f"Found potential logic bug in method {method} with confidence {logic_analysis['confidence']:.2f}")
            except Exception as e:
                logger.error(f"Error analyzing method {method}: {str(e)}")
        
        # Update state with business logic insights
        state.logic_insights = logic_insights
        
        # Use insights to adjust test generation strategy
        if logic_insights:
            # There are potential logic bugs identified, adjust test generation
            return self._adjust_test_generation_strategy(state, logic_insights)
        else:
            return state

    def _get_target_methods(self):
        """Get target methods for business logic analysis"""
        import re
        
        # Default to methods extracted from source code
        methods = []
        
        # Extract method names from source code
        method_pattern = r'(?:public|private|protected)?\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\('
        method_matches = re.finditer(method_pattern, self.source_code)
        
        for match in method_matches:
            method_name = match.group(1)
            # Skip constructor and common utility methods
            if method_name != self.class_name and method_name not in ['toString', 'hashCode', 'equals']:
                methods.append(method_name)
        
        # Prioritize public methods
        public_methods = []
        for method in methods:
            public_pattern = r'public\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]+>)?)\s+' + re.escape(method) + r'\s*\('
            if re.search(public_pattern, self.source_code):
                public_methods.append(method)
        
        # If we have too many methods, focus on public ones
        if len(methods) > 5 and public_methods:
            return public_methods
        
        return methods

    def _adjust_test_generation_strategy(self, state, logic_insights):
        """
        Adjust test generation strategy based on business logic insights
        
        Parameters:
        state (LogicAwareTestState): Current test state
        logic_insights (dict): Business logic insights
        
        Returns:
        LogicAwareTestState: Updated state with strategy adjustments
        """
        # Add target methods and insights to state for test generation
        state.logic_bug_targets = []
        
        for method, analysis in logic_insights.items():
            for bug in analysis.get("potential_bugs", []):
                state.logic_bug_targets.append({
                    "method": method,
                    "bug_description": bug.get("description", ""),
                    "confidence": bug.get("confidence", 0.0),
                    "intended_behavior": analysis.get("llm_analysis", {}).get("intended_behavior", ""),
                    "actual_behavior": analysis.get("llm_analysis", {}).get("actual_behavior", "")
                })
        
        # Set state flags to help guide test generation
        state.has_potential_logic_bugs = len(state.logic_bug_targets) > 0
        
        # Log analysis results
        logger.info(f"Identified {len(state.logic_bug_targets)} potential logic bug targets")
        
        return state


    def run_search(self):
        """
        运行MCTS搜索算法
        
        Returns:
        tuple: (best_test_code, best_coverage)
        """
        logger.info(f"start Logic-Aware MCTS search, iteration times: {self.max_iterations}")
        
        # Store start time for performance metrics
        self.start_time = time.time()
        
        # Initialize tracking list for all states
        self.all_states = []
        
        # If not processed, process initial state
        if not self.root_state:
            self.root_state = self.process_initial_state(None)
        
        # Add initial state to tracking list
        if self.root_state:
            self.all_states.append(self.root_state)
        
        # NEW: Add business logic analysis to initial state
        # This will enhance the test generation with awareness of potential logic bugs
        # self.root_state = self._analyze_business_logic(self.root_state)

        # Create root node
        self.root = LogicAwareMCTSNode(self.root_state)  # Store as instance attribute
        
        # Main MCTS loop
        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration  # Track current iteration
            logger.info(f"iteration {iteration}/{self.max_iterations}")
            
            try:
                # 1. Selection - use stored root node
                selected_node = self.selection(self.root)
                
                # 2. Expansion
                if selected_node.is_fully_expanded():
                    expanded_node = selected_node
                else:
                    expanded_node = self.expansion(selected_node)
                    # If new state created, add to tracking list
                    if expanded_node != selected_node and expanded_node.state:
                        self.all_states.append(expanded_node.state)
                
                # 3. Simulation
                reward = self.simulation(expanded_node)
                
                # 4. Backpropagation
                self.backpropagation(expanded_node, reward)
                
                # Update best test
                if reward > self.best_reward:
                    self.update_best_tests(expanded_node.state, reward, iteration)
                
                # Record history - use most promising child node rather than last expanded node
                best_node = self.root.best_child(
                    exploration_weight=self.exploration_weight,
                    logic_weight=self.logic_weight
                )
                self.record_history(best_node, iteration, reward)
                
                # Check termination condition
                if self.check_termination(iteration):
                    logger.info(f"iteration {iteration} satisfies termination condition")
                    break
                
            except Exception as e:
                logger.error(f"iteration {iteration} error: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save metrics for academic evaluation
        self.save_logic_metrics()
        
        # After MCTS search, verify all collected potential bugs
        logger.info(f"MCTS search completed, start verifying {len(self.potential_bugs)} potential bugs")
        logger.info(f"collected {len(self.all_states)} test states")
        self.verified_bug_methods = self.verify_all_potential_bugs()
        
        # Save test summary to JSON file
        self.save_test_summary()
        
        # Generate integrated test code with all verified bugs
        logger.info("Generate integrated test code with all verified bugs...")
        integrated_test_code = self.generate_integrated_test_code()
        
        # If there is integrated test code, use it as final result
        if integrated_test_code and integrated_test_code != self.best_test:
            logger.info("Use integrated test code as final result")
            final_test_code = integrated_test_code
        else:
            logger.info("Use best test code as final result (no bug integration)")
            final_test_code = self.best_test
        
        logger.info(f"MCTS search completed, best coverage={self.current_coverage:.2f}%, " +
                f"found logical bugs={len([m for m in self.verified_bug_methods if m.get('is_real_bug', False)])}")
        
        return final_test_code, self.current_coverage


    def verify_all_potential_bugs(self):
        """
        Verify all collected potential bugs at once
        
        Returns:
        list: Verified bug method list
        """
        from logic_bug_verifier import LogicBugVerifier
        
        # Create bug verifier
        verifier = LogicBugVerifier(self.source_code, self.class_name, self.package_name)
        
        # Group potential bugs by method name
        bugs_by_method = {}
        for bug in self.potential_bugs:
            method_name = bug.get("test_method", "unknown")
            if method_name not in bugs_by_method:
                bugs_by_method[method_name] = []
            
            # Only add unverified bugs
            if not bug.get("verified", False):
                # Generate robust bug signature for each bug (if not already)
                if not bug.get("bug_signature"):
                    bug["bug_signature"] = self._create_robust_bug_signature(bug)
                    
                bugs_by_method[method_name].append(bug)
        
        if not bugs_by_method:
            logger.info("No bugs to verify")
            return []
            
        logger.info(f"Start verifying {len(bugs_by_method)} test methods with potential bugs")
        
        # Prepare method list for verification
        methods_to_verify = []
        
        for method_name, bugs in bugs_by_method.items():
            # If no bug to verify, skip
            if not bugs:
                continue
                
            # Use method code extracted during simulation
            # If multiple bugs correspond to the same method, use method code from the first bug with code
            method_code = None
            for bug in bugs:
                if bug.get("method_code"):
                    method_code = bug.get("method_code")
                    logger.info(f"Use method code extracted during simulation: {method_name}")
                    break
            
            # If no method code, try to extract from multiple sources
            if not method_code:
                # Try various methods to extract code...
                
                pass
            
            # Merge bug information for the same method
            bug_descriptions = []
            for bug in bugs:
                bug_type = bug.get("bug_type", "unknown")
                error = bug.get("error", "")
                
                # Simplify error information
                if len(error) > 100:
                    error = error[:100] + "..."
                    
                bug_descriptions.append(f"{bug_type}: {error}")
            
            # Create method verification information, using robust bug signature
            method_info = {
                "method_name": method_name,
                "code": method_code,
                "bug_info": bugs,
                "bug_descriptions": bug_descriptions,
                "bug_signature": bugs[0].get("bug_signature")  # Use signature from the first bug
            }
            
            methods_to_verify.append(method_info)
        
        # Verify all methods
        verified_methods = verifier.verify_bugs(methods_to_verify)
        
        # Save original verification results
        self.original_verified_methods = verified_methods.copy()
        
        # Add robust signature for verified methods
        for method in verified_methods:
            if not method.get("bug_signature"):
                method["bug_signature"] = self._create_robust_bug_signature(method)
        
        # Ensure verification results correctly reflect real/false positive status
        for method in verified_methods:
            method_name = method.get("method_name", "")
            is_real_bug = method.get("is_real_bug", False)
            logger.info(f"Verification result: method {method_name}, is_real_bug={is_real_bug}")
        
        # Calculate and record verification results
        real_bugs = [m for m in verified_methods if m.get("is_real_bug", False)]
        false_positives = [m for m in verified_methods if not m.get("is_real_bug", False)]
        
        # Explicitly set verification statistics
        self.real_bugs_count = len(real_bugs)
        self.false_positives_count = len(false_positives) 
        self.total_verified_methods = len(verified_methods)
        
        # Update logical bug count - now only count real bugs
        self.logical_bugs_found = self.real_bugs_count
        
        logger.info(f"Verification completed: found {len(real_bugs)} real bugs, {len(false_positives)} false positives, from {len(verified_methods)} test methods")
        
        return verified_methods


    def process_bug_findings(self, state, iteration):
        """
        Process bug findings in state
        
        Parameters:
        state (LogicAwareTestState): Test state
        iteration (int): Current iteration
        """
        # Skip state with no logical bugs
        if not state or not hasattr(state, 'has_logical_bugs') or not state.has_logical_bugs:
            return
            
        # Get logical bug methods
        logical_bug_methods = state.get_logical_bug_finding_methods()
        
        if not logical_bug_methods:
            logger.warning("State reports logical bugs but get_logical_bug_finding_methods returned empty list")
            return
            
        logger.info(f"Found {len(logical_bug_methods)} logical bug methods at iteration {iteration}")
        
        # Track new found bugs
        found_new_bugs = False
        verified_bug_count = 0
        total_detected_bugs = len(logical_bug_methods)
        
        # For tracking unique bug methods found in this iteration
        current_iteration_bug_methods = set()
        current_iteration_verified_bug_methods = set()
            
        # Process each bug method
        for bug_method in logical_bug_methods:
            method_name = bug_method.get("method_name", "")
            bug_type = bug_method.get("bug_type", "unknown")
            bug_category = bug_method.get("bug_category", "general")
            
            # Add iteration information
            bug_method["found_in_iteration"] = iteration
            
            # Skip non-logical bugs
            if bug_category != "logical":
                continue
            
            # Generate robust bug signature
            if not bug_method.get("bug_signature"):
                bug_method["bug_signature"] = self._create_robust_bug_signature(bug_method)
                
            bug_signature = bug_method["bug_signature"]
            
            # Add to set of bugs found in this iteration
            current_iteration_bug_methods.add(bug_signature)
                    
            # Check if it is a new bug type
            if bug_type not in self.metrics["logical_bug_types"]:
                # 记录首次发现逻辑bug的迭代
                if self.metrics["iterations_to_first_logical_bug"] is None:
                    self.metrics["iterations_to_first_logical_bug"] = iteration
                
                # 添加到bug类型集合
                self.metrics["logical_bug_types"].add(bug_type)
                logger.info(f"Found new logical bug type: {bug_type}")
            
            # Check if it is a new bug discovery - use unique signature comparison
            method_exists = any(m.get("bug_signature", "") == bug_signature for m in self.verified_bug_methods)
            is_new_bug = not method_exists
            
            # Extract complete test method code
            if not bug_method.get("method_code") and state.test_code:
                # Try to extract complete method from test code
                method_code = self._extract_method_from_test_code(state.test_code, method_name)
                if method_code:
                    bug_method["method_code"] = method_code
                    bug_method["found_in_iteration"] = iteration_number
                    logger.info(f"Successfully extracted method code: {method_name}")
            
            # Verify bug - even if already verified, still record iteration number
            try:
                # 如果bug尚未验证，则进行验证
                if not bug_method.get("verified", False):
                    verifier = LogicBugVerifier(self.source_code, self.class_name, self.package_name)
                    verified_result = verifier.verify_bugs([bug_method])
                    
                    # 如果验证返回了结果，使用验证信息
                    if verified_result and len(verified_result) > 0:
                        # 更新bug_method内容，保留迭代号和方法代码
                        verified_bug = verified_result[0]
                        iteration_number = bug_method.get("found_in_iteration")
                        method_code = bug_method.get("method_code", "")
                        bug_signature = bug_method.get("bug_signature", "")
                        
                        for key, value in verified_bug.items():
                            bug_method[key] = value
                            
                        # 恢复迭代号、方法代码和bug签名
                        bug_method["found_in_iteration"] = iteration_number
                        if method_code:
                            bug_method["method_code"] = method_code
                        if bug_signature:
                            bug_method["bug_signature"] = bug_signature
                    else:
                        # 如果验证没有返回结果，标记为已验证
                        bug_method["verified"] = True
                        bug_method["is_real_bug"] = bug_method.get("is_real_bug", False)
                        logger.warning(f"Bug verification returned no results for {method_name}")
                else:
                    logger.info(f"Bug already verified: {method_name}")
                    # 确保即使已验证的bug也有迭代号
                    if "found_in_iteration" not in bug_method:
                        bug_method["found_in_iteration"] = iteration
            except Exception as e:
                logger.error(f"Error verifying bug {method_name}: {str(e)}")
                logger.error(traceback.format_exc())
                # 错误时，标记为已验证但保留原始is_real_bug值
                bug_method["verified"] = True
            
            # 添加到已验证bug方法（如果是新bug）
            if is_new_bug:
                self.verified_bug_methods.append(bug_method)
                self.logical_bugs_found += 1
                found_new_bugs = True
                
                # 更新指标
                self.metrics["total_logical_bug_tests"] += 1
                
                # 更新策略有效性
                if hasattr(state, 'metadata') and state.metadata and "action" in state.metadata and "strategy" in state.metadata["action"]:
                    strategy = state.metadata["action"]["strategy"]
                    self.metrics["strategy_effectiveness"][strategy]["bugs_found"] += 1
                
                logger.info(f"Added new logical bug method: {method_name} (type: {bug_type}, is_real_bug: {bug_method.get('is_real_bug', False)})")
            
            # 计算已验证的真实bug
            if bug_method.get("verified", False) and bug_method.get("is_real_bug", False):
                verified_bug_count += 1
                current_iteration_verified_bug_methods.add(bug_signature)
        
        # 记录本次迭代发现的唯一bug数量
        unique_bugs_in_iteration = len(current_iteration_bug_methods)
        unique_verified_bugs_in_iteration = len(current_iteration_verified_bug_methods)
        
        # 更新bug趋势 - 确保使用本次迭代发现的唯一bug数量而不是重复计算
        self._update_bug_trend(iteration, unique_bugs_in_iteration, unique_verified_bugs_in_iteration)
        
        # 更新状态中的bug计数，用于奖励计算
        state.unique_bugs_count = unique_bugs_in_iteration
        state.unique_verified_bugs_count = unique_verified_bugs_in_iteration
        
        # 如果发现新bug且此状态不是最佳状态，考虑保留它
        if found_new_bugs and state != self.best_state:
            # 重新计算包含新bug影响的奖励
            reward = self.calculate_logic_aware_reward(state)
            
            # 如果包含bug的版本奖励高于当前最佳，更新最佳版本
            if reward > self.best_reward:
                self.update_best_tests(state, reward, iteration)

    def _extract_method_from_test_code(self, test_code, method_name):
        """
        从完整测试代码中提取指定方法的代码
        
        Parameters:
        test_code (str): 完整测试代码
        method_name (str): 方法名
        
        Returns:
        str: 提取的方法代码，如果未找到则返回空字符串
        """
        import re
        try:
            # 匹配方法定义和整个方法体，考虑不同的访问修饰符和返回类型
            pattern = r'(public|private|protected)?\s+(?:static\s+)?(?:final\s+)?(?:[\w\<\>\[\]]+\s+)?' + re.escape(method_name) + r'\s*\([^\)]*\)\s*(?:throws\s+[\w\.,\s]+)?\s*\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}'
            
            match = re.search(pattern, test_code)
            if match:
                return match.group(0)
            
            return ""
        except Exception as e:
            logger.error(f"提取方法时出错: {str(e)}")
            return ""
    
    def _rename_variables(self, code, used_vars, suffix):
        """
        更全面的变量重命名，避免冲突
        
        Parameters:
        code (str): 源代码
        used_vars (set): 已使用的变量名
        suffix (int): 方法区分后缀
        
        Returns:
        tuple: (修改后的代码, 新使用的变量名)
        """
        import re
        new_used_vars = set()
        modified_code = code
        
        # 保留字和常用类型，不应该被误认为是变量
        java_keywords = {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
            "const", "continue", "default", "do", "double", "else", "enum", "extends", "false",
            "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
            "int", "interface", "long", "native", "new", "null", "package", "private", "protected",
            "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized",
            "this", "throw", "throws", "transient", "true", "try", "void", "volatile", "while",
            "String", "Integer", "Double", "Float", "Long", "Boolean", "Character", "Byte", "Short",
            "Object", "Class", "System", "Exception", "RuntimeException", "Throwable", "Error",
            "List", "Map", "Set", "Collection", "Arrays", "ArrayList", "HashMap", "HashSet",
            "StringBuilder", "StringBuffer", "Math", "Thread", "Runnable"
        }
        
        # 为每个测试方法增加唯一前缀
        method_prefix = f"m{suffix}_"
        
        # 扩展匹配模式以捕获更多变量声明场景
        var_patterns = [
            # 基本变量赋值
            r'(\w+(?:<[^>]+>)?)\s+(\w+)\s*=',
            
            # 不带初始化的变量声明
            r'(\w+(?:<[^>]+>)?)\s+(\w+)\s*;',
            
            # for-each循环
            r'for\s*\(\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*:',
            
            # 普通for循环
            r'for\s*\(\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*=',
            
            # catch语句
            r'catch\s*\(\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*\)',
            
            # 方法参数
            r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(',
            
            # 增加对Lambda表达式参数的支持
            r'(?:\(|,)\s*(\w+)\s*->'
        ]
        
        # 收集所有变量和它们的类型
        all_vars = []
        for pattern in var_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                try:
                    # 对于大多数模式，变量类型在组1，变量名在组2
                    if len(match.groups()) > 1:
                        var_type = match.group(1)
                        var_name = match.group(2)
                    else:
                        # 对于Lambda表达式，只有变量名
                        var_type = "unknown"
                        var_name = match.group(1)
                    
                    # 跳过关键字和基本类型
                    if var_name in java_keywords:
                        continue
                        
                    all_vars.append((var_type, var_name))
                except Exception as e:
                    logger.debug(f"处理变量匹配时出错: {str(e)}")
                    continue
        
        # 创建变量重命名映射 - 为所有变量添加特定前缀
        var_map = {}
        for var_type, var_name in all_vars:
            # 所有变量都添加方法特定前缀
            new_name = f"{method_prefix}{var_name}"
            var_map[var_name] = new_name
            new_used_vars.add(new_name)
        
        # 系统性替换所有变量名
        # 按照变量名长度从长到短排序，防止部分替换问题
        sorted_vars = sorted(var_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for old_name, new_name in sorted_vars:
            # 使用正则替换变量名，确保只替换完整的标识符
            modified_code = re.sub(
                r'\b' + re.escape(old_name) + r'\b',
                new_name,
                modified_code
            )
        
        # 检查并重命名常用测试框架方法以避免冲突
        helper_method_patterns = [
            (r'void\s+when\s*\(', f'void when_{suffix}('),
            (r'void\s+then\s*\(', f'void then_{suffix}('),
            (r'void\s+given\s*\(', f'void given_{suffix}('),
            (r'(?:void|boolean)\s+assert\w+\s*\(', f'\\g<0>{suffix}')
        ]
        
        for pattern, replacement in helper_method_patterns:
            modified_code = re.sub(pattern, replacement, modified_code)
        
        # 修改方法内的调用
        for pattern, _ in helper_method_patterns:
            base_name = re.search(r'(?:void|boolean)\s+(\w+)', pattern)
            if base_name:
                method_name = base_name.group(1)
                modified_code = re.sub(
                    r'\b' + re.escape(method_name) + r'\s*\(',
                    f"{method_name}_{suffix}(",
                    modified_code
                )
        
        # 替换可能的lambda表达式
        lambda_pattern = r'(?:->\s*\{|\(\)\s*->\s*\{)'
        if re.search(lambda_pattern, modified_code):
            # 提取lambda内部变量并替换
            lambda_matches = re.finditer(r'->\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', modified_code)
            for lambda_match in lambda_matches:
                lambda_body = lambda_match.group(1)
                lambda_vars = re.finditer(r'\b(\w+)\b', lambda_body)
                for var_match in lambda_vars:
                    var_name = var_match.group(1)
                    if var_name in var_map:
                        lambda_body = re.sub(
                            r'\b' + re.escape(var_name) + r'\b',
                            var_map[var_name],
                            lambda_body
                        )
                # 替换修改后的lambda体
                modified_code = modified_code.replace(
                    lambda_match.group(0),
                    "-> {" + lambda_body + "}"
                )
        
        return modified_code, new_used_vars

    def _add_missing_helper_methods(self, code):
        """
        查找并添加缺失的辅助方法
        
        Parameters:
        code (str): 集成测试代码
        
        Returns:
        str: 更新后的代码
        """
        import re
        
        # 检测代码中的辅助方法调用
        helper_calls = set()
        helper_patterns = [
            r'(\w+)_(\d+)\s*\((?:\s*[\w\[\],.<>]+\s+\w+\s*(?:,)?)*\s*\)',  # 带参数的方法调用
            r'assert\w+_(\d+)\s*\(',  # 带后缀的assert方法
        ]
        
        for pattern in helper_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                if match.group(0).startswith("assert"):
                    # 处理assert方法
                    helper_calls.add(f"assert{match.group(1)}")
                else:
                    # 处理其他辅助方法
                    helper_calls.add(f"{match.group(1)}_{match.group(2)}")
        
        # 定义辅助方法模板
        helper_templates = {
            "when": """
        // 辅助方法：when
        private void when_{suffix}(Object obj) {
            // 测试辅助方法
        }
        
        private void when_{suffix}(double value) {
            // 测试辅助方法
        }
        """,
            "then": """
        // 辅助方法：then
        private void then_{suffix}(Object obj) {
            // 测试辅助方法
        }
        
        private void then_{suffix}(double value) {
            // 测试辅助方法
        }
        """,
            "given": """
        // 辅助方法：given
        private void given_{suffix}(Object obj) {
            // 测试辅助方法
        }
        
        private void given_{suffix}(double value) {
            // 测试辅助方法
        }
        """,
            "assert": """
        // 辅助方法：断言
        private <T extends Throwable> void assertThrows_{suffix}(Class<T> expectedType, Runnable code, String message) {
            try {
                code.run();
                fail("Expected exception: " + expectedType.getName() + " but nothing was thrown");
            } catch (Throwable t) {
                if (!expectedType.isInstance(t)) {
                    fail(message + " - Expected: <" + expectedType.getName() + "> but was: <" + t.getClass().getName() + ">");
                }
            }
        }
        """
        }
        
        # 构建缺失的辅助方法
        missing_methods = []
        for helper in helper_calls:
            for base_name, template in helper_templates.items():
                if helper.startswith(base_name):
                    suffix = helper.split('_')[-1]
                    missing_methods.append(template.format(suffix=suffix))
                    break
        
        # 将缺失的方法添加到类的末尾
        if missing_methods:
            class_end = code.rfind('}')
            if class_end != -1:
                code = (
                    code[:class_end] + 
                    "\n    // ===== 自动生成的辅助方法 ===== \n" +
                    "".join(missing_methods) +
                    code[class_end:]
                )
        
        return code

    def _update_bug_trend(self, iteration, detected_bugs, verified_bugs):
        """
        Update the bug trend records with cumulative bug counts
        
        Parameters:
        iteration (int): Current iteration
        detected_bugs (int or list): Number of detected bugs or list of bug methods in current iteration
        verified_bugs (int or list): Number of verified bugs or list of verified bug methods in current iteration
        """
        # Make sure bug_trend exists
        if not hasattr(self, "bug_trend"):
            self.bug_trend = []
            self.unique_detected_bug_signatures = set()
            self.unique_verified_bug_signatures = set()
        
        # Extract current iteration bug methods from history
        current_detected_methods = []
        current_verified_methods = []
        
        # Find the history entry for this iteration
        history_entry = None
        for entry in self.history:
            if entry.get("iteration") == iteration:
                history_entry = entry
                break
        
        # Process the history entry if found
        if history_entry:
            # If we have bug_details, use those for more accurate tracking
            if "bug_details" in history_entry and isinstance(history_entry["bug_details"], list):
                current_detected_methods = []
                current_verified_methods = []
                
                for bug in history_entry["bug_details"]:
                    # 使用新的健壮签名函数创建唯一bug签名
                    bug_info = {
                        "method_name": bug.get("method", "unknown"),
                        "bug_type": bug.get("type", "unknown"),
                        "error": bug.get("description", ""),
                        "found_in_iteration": iteration
                    }
                    
                    bug_signature = self._create_robust_bug_signature(bug_info)
                    
                    current_detected_methods.append(bug_signature)
                    if bug.get("verified", False) and bug.get("is_real_bug", False):
                        current_verified_methods.append(bug_signature)
            else:
                # Handle detected_bugs - for scenarios without detailed bug_details
                if "detected_bugs" in history_entry:
                    if isinstance(history_entry["detected_bugs"], list):
                        for bug_name in history_entry["detected_bugs"]:
                            # 创建bug信息并生成签名
                            bug_info = {
                                "method_name": bug_name,
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_detected_methods.append(self._create_robust_bug_signature(bug_info))
                    elif isinstance(history_entry["detected_bugs"], (int, float)) and history_entry["detected_bugs"] > 0:
                        for i in range(int(history_entry["detected_bugs"])):
                            bug_info = {
                                "method_name": f"bug_method_{iteration}_{i}",
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_detected_methods.append(self._create_robust_bug_signature(bug_info))
                
                # Handle verified_bugs
                if "verified_bugs" in history_entry:
                    if isinstance(history_entry["verified_bugs"], list):
                        for bug_name in history_entry["verified_bugs"]:
                            bug_info = {
                                "method_name": bug_name,
                                "bug_type": "unknown", 
                                "found_in_iteration": iteration
                            }
                            current_verified_methods.append(self._create_robust_bug_signature(bug_info))
                    elif isinstance(history_entry["verified_bugs"], (int, float)) and history_entry["verified_bugs"] > 0:
                        for i in range(int(history_entry["verified_bugs"])):
                            bug_info = {
                                "method_name": f"verified_method_{iteration}_{i}",
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_verified_methods.append(self._create_robust_bug_signature(bug_info))
        else:
            # If no history entry found, use the direct parameters
            logger.info(f"No history entry found for iteration {iteration}, using direct parameters")
            
            # Handle detected_bugs parameter - could be list or number
            if isinstance(detected_bugs, list):
                for i, bug in enumerate(detected_bugs):
                    bug_name = bug if isinstance(bug, str) else f"bug_method_{iteration}_{i}"
                    bug_info = {
                        "method_name": bug_name,
                        "bug_type": "unknown",
                        "found_in_iteration": iteration
                    }
                    current_detected_methods.append(self._create_robust_bug_signature(bug_info))
            elif detected_bugs > 0:
                for i in range(detected_bugs):
                    bug_info = {
                        "method_name": f"bug_method_{iteration}_{i}",
                        "bug_type": "unknown",
                        "found_in_iteration": iteration
                    }
                    current_detected_methods.append(self._create_robust_bug_signature(bug_info))
            
            # Handle verified_bugs parameter - could be list or number
            if isinstance(verified_bugs, list):
                for i, bug in enumerate(verified_bugs):
                    bug_name = bug if isinstance(bug, str) else f"verified_method_{iteration}_{i}"
                    bug_info = {
                        "method_name": bug_name,
                        "bug_type": "unknown",
                        "found_in_iteration": iteration
                    }
                    current_verified_methods.append(self._create_robust_bug_signature(bug_info))
            elif verified_bugs > 0:
                for i in range(verified_bugs):
                    bug_info = {
                        "method_name": f"verified_method_{iteration}_{i}",
                        "bug_type": "unknown",
                        "found_in_iteration": iteration
                    }
                    current_verified_methods.append(self._create_robust_bug_signature(bug_info))
        
        # Update the unique bug sets
        self.unique_detected_bug_signatures.update(current_detected_methods)
        self.unique_verified_bug_signatures.update(current_verified_methods)
        
        # Calculate cumulative counts
        cumulative_detected = len(self.unique_detected_bug_signatures)
        cumulative_verified = len(self.unique_verified_bug_signatures)
        
        # Convert detected_bugs/verified_bugs to integers if they're lists
        detected_bugs_count = len(detected_bugs) if isinstance(detected_bugs, list) else detected_bugs
        verified_bugs_count = len(verified_bugs) if isinstance(verified_bugs, list) else verified_bugs
        
        # Add a new trend point with both current and cumulative data
        self.bug_trend.append({
            "iteration": iteration,
            "detected_bugs": detected_bugs_count,              # Bugs found in this iteration
            "verified_bugs": verified_bugs_count,              # Verified bugs in this iteration
            "cumulative_detected_bugs": cumulative_detected,   # Total unique bugs found so far
            "cumulative_verified_bugs": cumulative_verified    # Total unique verified bugs so far
        })
        
        # Add bug trend to metrics
        self.metrics["bug_trend"] = self.bug_trend
        
        logger.info(f"Updated bug trend - iteration {iteration}: " +
                f"detected={detected_bugs_count}, verified={verified_bugs_count}, " +
                f"cumulative detected={cumulative_detected}, cumulative verified={cumulative_verified}")
    

    def _generate_bug_trend(self):
        """
        从历史记录中生成bug趋势数据，用于测试摘要
        
        Returns:
        list: bug趋势数据列表
        """
        # 如果没有历史记录，返回空列表
        if not hasattr(self, "history") or not self.history:
            return []
        
        # 获取原始验证方法中真实bug的数量
        real_bugs_count = 0
        verified_bug_methods = []
        
        # 首先从原始验证结果中获取真实bug
        if hasattr(self, "original_verified_methods") and self.original_verified_methods:
            for bug in self.original_verified_methods:
                if bug.get("is_real_bug", False):
                    method_name = bug.get("method_name", "")
                    if method_name and method_name not in verified_bug_methods:
                        verified_bug_methods.append(method_name)
        
        # 也从verified_bug_methods中获取
        if hasattr(self, "verified_bug_methods") and self.verified_bug_methods:
            for bug in self.verified_bug_methods:
                if bug.get("is_real_bug", False):
                    method_name = bug.get("method_name", "")
                    if method_name and method_name not in verified_bug_methods:
                        verified_bug_methods.append(method_name)
        
        real_bugs_count = len(verified_bug_methods)
        logger.info(f"为趋势图找到 {real_bugs_count} 个真实bug")
        
        # 初始化趋势数据和累积bug跟踪
        bug_trend = []
        unique_detected_bugs = set()
        unique_verified_bugs = set()
        
        # 创建每个迭代的趋势数据点
        for entry in sorted(self.history, key=lambda x: x.get("iteration", 0)):
            iteration = entry.get("iteration", 0)
            
            # 获取当前迭代检测到的bug的唯一集合（去重）
            detected_bugs_set = set()
            if "detected_bugs" in entry and isinstance(entry["detected_bugs"], list):
                detected_bugs_set.update(entry["detected_bugs"])
            elif "bugs_found" in entry and entry["bugs_found"] > 0:
                # 如果只有数字指示的bug数量，则使用占位符
                for i in range(entry["bugs_found"]):
                    detected_bugs_set.add(f"bug_{iteration}_{i}")
            
            # 获取当前迭代验证的bug的唯一集合（去重）
            verified_bugs_set = set()
            if "verified_bugs" in entry and isinstance(entry["verified_bugs"], list):
                verified_bugs_set.update(entry["verified_bugs"])
            
            # 如果存在bug_details字段，也从其中提取信息
            if "bug_details" in entry and isinstance(entry["bug_details"], list):
                for bug in entry["bug_details"]:
                    method = bug.get("method", "")
                    if method:
                        detected_bugs_set.add(method)
                        if bug.get("verified", False) and bug.get("is_real_bug", False):
                            verified_bugs_set.add(method)
            
            # 当前迭代的唯一bug数量
            detected_bugs_count = len(detected_bugs_set)
            verified_bugs_count = len(verified_bugs_set)
            
            # 更新累积唯一bug集合
            unique_detected_bugs.update(detected_bugs_set)
            unique_verified_bugs.update(verified_bugs_set)
            
            # 创建带有累积统计的趋势点
            current_entry = {
                "iteration": iteration,
                "detected_bugs": detected_bugs_count,  # 当前迭代检测到的唯一bug数
                "verified_bugs": verified_bugs_count,  # 当前迭代验证的唯一bug数
                "cumulative_detected_bugs": len(unique_detected_bugs),  # 累计唯一检测bug数
                "cumulative_verified_bugs": len(unique_verified_bugs)   # 累计唯一验证bug数
            }
            
            bug_trend.append(current_entry)
        
        # 如果没有创建任何趋势数据，添加一个空的趋势点防止出错
        if not bug_trend:
            bug_trend.append({
                "iteration": 1,
                "detected_bugs": 0,
                "verified_bugs": 0,
                "cumulative_detected_bugs": 0,
                "cumulative_verified_bugs": 0
            })
        
        return bug_trend

    def _generate_coverage_trend(self):
        """
        从历史记录中生成覆盖率趋势数据，用于测试摘要
        
        Returns:
        list: 覆盖率趋势数据列表
        """
        # 如果没有历史记录，返回空列表
        if not hasattr(self, "history") or not self.history:
            return []
        
        # 从历史记录中提取覆盖率信息
        coverage_trend = []
        unique_bug_signatures = set()
        unique_verified_bug_signatures = set()
        
        for i, entry in enumerate(sorted(self.history, key=lambda x: x.get("iteration", 0))):
            iteration = entry.get("iteration", i+1)
            
            # 提取当前迭代发现的bug
            current_iteration_bugs = set()
            current_iteration_verified_bugs = set()
            
            # 处理bug_details字段中的详细信息
            if "bug_details" in entry and isinstance(entry["bug_details"], list):
                for bug in entry["bug_details"]:
                    # 创建完整的bug信息，用于生成签名
                    bug_info = {
                        "method_name": bug.get("method", "unknown"),
                        "bug_type": bug.get("type", "unknown"),
                        "error": bug.get("description", ""),
                        "found_in_iteration": iteration
                    }
                    
                    # 使用健壮的签名函数创建唯一签名
                    bug_signature = self._create_robust_bug_signature(bug_info)
                    
                    # 添加到当前迭代的bug集合
                    current_iteration_bugs.add(bug_signature)
                    
                    # 如果是已验证的真实bug，添加到已验证bug集合
                    if bug.get("verified", False) and bug.get("is_real_bug", False):
                        current_iteration_verified_bugs.add(bug_signature)
            else:
                # 处理detected_bugs字段
                detected_bugs = entry.get("detected_bugs", None)
                if detected_bugs is not None:
                    if isinstance(detected_bugs, list):
                        # 为每个bug创建健壮签名
                        for bug in detected_bugs:
                            bug_info = {
                                "method_name": bug if isinstance(bug, str) else f"unknown_bug_{iteration}",
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_iteration_bugs.add(self._create_robust_bug_signature(bug_info))
                    elif isinstance(detected_bugs, (int, float)) and detected_bugs > 0:
                        # 为每个计数创建占位符bug并生成签名
                        for j in range(int(detected_bugs)):
                            bug_info = {
                                "method_name": f"unknown_bug_{iteration}_{j}",
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_iteration_bugs.add(self._create_robust_bug_signature(bug_info))
                
                # 处理verified_bugs字段
                verified_bugs = entry.get("verified_bugs", None)
                if verified_bugs is not None:
                    if isinstance(verified_bugs, list):
                        # 为每个已验证bug创建健壮签名
                        for bug in verified_bugs:
                            bug_info = {
                                "method_name": bug if isinstance(bug, str) else f"unknown_verified_bug_{iteration}",
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_iteration_verified_bugs.add(self._create_robust_bug_signature(bug_info))
                    elif isinstance(verified_bugs, (int, float)) and verified_bugs > 0:
                        # 为每个计数创建占位符已验证bug并生成签名
                        for j in range(int(verified_bugs)):
                            bug_info = {
                                "method_name": f"unknown_verified_bug_{iteration}_{j}",
                                "bug_type": "unknown",
                                "found_in_iteration": iteration
                            }
                            current_iteration_verified_bugs.add(self._create_robust_bug_signature(bug_info))
                
                # 也检查bugs_found字段作为后备
                bugs_found = entry.get("bugs_found", 0)
                if isinstance(bugs_found, (int, float)) and bugs_found > 0 and not current_iteration_bugs:
                    # 如果其他字段没有提供信息，使用bugs_found
                    for j in range(int(bugs_found)):
                        bug_info = {
                            "method_name": f"bugs_found_{iteration}_{j}",
                            "bug_type": "unknown",
                            "found_in_iteration": iteration
                        }
                        current_iteration_bugs.add(self._create_robust_bug_signature(bug_info))
            
            # 更新唯一bug集合
            unique_bug_signatures.update(current_iteration_bugs)
            unique_verified_bug_signatures.update(current_iteration_verified_bugs)
            
            # 当前迭代检测到的bug数量
            current_detected_count = len(current_iteration_bugs)
            current_verified_count = len(current_iteration_verified_bugs)
            
            # 累计唯一bug数量
            cumulative_detected = len(unique_bug_signatures)
            cumulative_verified = len(unique_verified_bug_signatures)
            
            # 构建趋势条目
            trend_entry = {
                "iteration": iteration,
                "coverage": entry.get("coverage", 0.0),
                "best_coverage": entry.get("current_best_coverage", entry.get("coverage", 0.0)),
                "reward": entry.get("reward", 0.0),
                "detected_bugs": current_detected_count,       # 当前迭代发现的bug数
                "verified_bugs": current_verified_count,       # 当前迭代验证的真实bug数
                "cumulative_detected_bugs": cumulative_detected,  # 累计唯一bug数
                "cumulative_verified_bugs": cumulative_verified   # 累计唯一验证bug数
            }
            coverage_trend.append(trend_entry)
            
        # 日志记录趋势统计
        logger.info(f"生成覆盖率趋势数据: {len(coverage_trend)} 个数据点，" +
                f"累计检测到 {cumulative_detected} 个唯一bug，" +
                f"累计验证 {cumulative_verified} 个唯一bug")
        
        return coverage_trend

    def _calculate_coverage_improvement_rate(self):
        """
        根据历史记录计算覆盖率改进率
        
        Returns:
        float: 覆盖率改进率
        """
        # 如果没有历史记录，返回0
        if not hasattr(self, "history") or not self.history:
            return 0.0
        
        # 获取初始覆盖率和最终覆盖率
        initial_coverage = self.history[0].get("coverage", 0.0)
        final_coverage = self.current_coverage
        
        # 计算覆盖率改进率
        improvement = final_coverage - initial_coverage
        iterations = len(self.history)
        
        if iterations <= 1:
            return improvement
        
        # 计算每次迭代的平均改进率
        improvement_rate = improvement / (iterations - 1)
        return round(improvement_rate, 4)
    
    # def update_best_tests(self, state, reward, iteration):
    #     """
    #     根据奖励更新最佳测试代码
        
    #     Parameters:
    #     state (LogicAwareTestState): 当前状态
    #     reward (float): 奖励值
    #     iteration (int): 当前迭代
    #     """
    #     # 如果状态为 None，跳过
    #     if not state:
    #         return
            
    #     # 检查此状态的覆盖率
    #     current_state_coverage = 0.0
    #     if hasattr(state, "coverage") and state.coverage > 0:
    #         current_state_coverage = state.coverage
            
    #     # 首先检查是否有更高的覆盖率
    #     if current_state_coverage > self.current_coverage:
    #         logger.info(f"Found higher coverage at iteration {iteration}: {current_state_coverage:.2f}% > {self.current_coverage:.2f}%")
    #         self.current_coverage = current_state_coverage
            
    #         # 这种情况下无论奖励值如何都应该更新最佳测试
    #         if self.best_state is None or self.best_reward < reward:
    #             self.best_state = state
    #             self.best_test = state.test_code
    #             self.best_reward = reward
    #             logger.info(f"Updated best test due to higher coverage: {self.current_coverage:.2f}%")
        
    #     # 接着检查奖励更高的情况
    #     elif reward > self.best_reward:
    #         logger.info(f"Found better test at iteration {iteration}: reward={reward:.4f} > {self.best_reward:.4f}")
            
    #         # 更新最佳状态和测试
    #         self.best_state = state
    #         self.best_test = state.test_code
    #         self.best_reward = reward
        
    #     # 记录高覆盖率迭代
    #     if (self.current_coverage >= 80.0 and 
    #         self.metrics["iterations_to_high_coverage"] is None):
    #         self.metrics["iterations_to_high_coverage"] = iteration
            
    #         # 更新逻辑覆盖率指标
    #         if hasattr(state, "covered_logic_patterns"):
    #             self.best_pattern_coverage = len(state.covered_logic_patterns)
    #             self.metrics["high_risk_patterns_covered"] = len([
    #                 pattern_id for pattern_id in state.covered_logic_patterns
    #                 if any(p.get("risk_level") == "high" for p in self.logic_patterns
    #                     if f"{p['type']}_{p['location']}" == pattern_id)
    #             ])
                
    #         if hasattr(state, "covered_branch_conditions"):
    #             self.best_branch_coverage = len(state.covered_branch_conditions)
                
    #             # 计算已覆盖的边界条件
    #             boundary_conditions = set(
    #                 cond_id for cond_id in state.covered_branch_conditions
    #                 if any(c.get("type") in ["if_condition", "while_loop", "for_loop"] 
    #                     for c in self.logic_model.boundary_conditions
    #                     if f"{c['method']}_{c['line']}" == cond_id)
    #             )
    #             self.metrics["boundary_conditions_covered"] = len(boundary_conditions)
                
    #             # 计算已覆盖的逻辑操作
    #             logical_operations = set(
    #                 cond_id for cond_id in state.covered_branch_conditions
    #                 if any(c.get("operation") in ["&&", "||", "!=", "=="] 
    #                     for c in self.logic_model.logical_operations
    #                     if f"{c['method']}_{c['line']}" == cond_id)
    #             )
    #             self.metrics["logical_operations_covered"] = len(logical_operations)


    def update_best_tests(self, state, reward, iteration):
        """
        根据奖励更新最佳测试代码
        
        Parameters:
        state (LogicAwareTestState): 当前状态
        reward (float): 奖励值
        iteration (int): 当前迭代
        """
        # 如果状态为 None，跳过
        if not state:
            return
            
        # 检查此状态的覆盖率
        current_state_coverage = 0.0
        if hasattr(state, "coverage") and state.coverage > 0:
            current_state_coverage = state.coverage
            
        # 首先检查是否有更高的覆盖率
        if current_state_coverage > self.current_coverage:
            logger.info(f"Found higher coverage at iteration {iteration}: {current_state_coverage:.2f}% > {self.current_coverage:.2f}%")
            self.current_coverage = current_state_coverage
            
            # 确保保存最高覆盖率对应的状态和测试代码
            self.best_state = state
            self.best_test = state.test_code
            self.best_reward = max(reward, self.best_reward)  # 使用当前奖励和最佳奖励的最大值
            logger.info(f"Updated best test due to higher coverage: {self.current_coverage:.2f}%")
        
        # 如果覆盖率相同但奖励更高，也更新
        elif current_state_coverage == self.current_coverage and reward > self.best_reward:
            logger.info(f"Found better test at iteration {iteration} with same coverage: reward={reward:.4f} > {self.best_reward:.4f}")
            self.best_state = state
            self.best_test = state.test_code
            self.best_reward = reward
        
        # 接着检查奖励更高的情况（只有当覆盖率不低于最高覆盖率的80%时才考虑）
        elif reward > self.best_reward and current_state_coverage >= self.current_coverage * 0.8:
            logger.info(f"Found better test at iteration {iteration}: reward={reward:.4f} > {self.best_reward:.4f}")
            
            # 更新最佳状态和测试
            self.best_state = state
            self.best_test = state.test_code
            self.best_reward = reward
        
        # 记录高覆盖率迭代
        if (self.current_coverage >= 80.0 and 
            self.metrics["iterations_to_high_coverage"] is None):
            self.metrics["iterations_to_high_coverage"] = iteration
            
            # 更新逻辑覆盖率指标
            if hasattr(state, "covered_logic_patterns"):
                self.best_pattern_coverage = len(state.covered_logic_patterns)
                self.metrics["high_risk_patterns_covered"] = len([
                    pattern_id for pattern_id in state.covered_logic_patterns
                    if any(p.get("risk_level") == "high" for p in self.logic_patterns
                        if f"{p['type']}_{p['location']}" == pattern_id)
                ])
                
            if hasattr(state, "covered_branch_conditions"):
                self.best_branch_coverage = len(state.covered_branch_conditions)
                
                # 计算已覆盖的边界条件
                boundary_conditions = set(
                    cond_id for cond_id in state.covered_branch_conditions
                    if any(c.get("type") in ["if_condition", "while_loop", "for_loop"] 
                        for c in self.logic_model.boundary_conditions
                        if f"{c['method']}_{c['line']}" == cond_id)
                )
                self.metrics["boundary_conditions_covered"] = len(boundary_conditions)
                
                # 计算已覆盖的逻辑操作
                logical_operations = set(
                    cond_id for cond_id in state.covered_branch_conditions
                    if any(c.get("operation") in ["&&", "||", "!=", "=="] 
                        for c in self.logic_model.logical_operations
                        if f"{c['method']}_{c['line']}" == cond_id)
                )
                self.metrics["logical_operations_covered"] = len(logical_operations)
        
        # 保存最高覆盖率测试代码的副本
        if current_state_coverage >= self.current_coverage * 0.9:
            # 只保存足够高覆盖率的测试代码
            coverage_str = f"{current_state_coverage:.2f}".replace(".", "_")
            self.high_coverage_tests[coverage_str] = state.test_code

    def record_history(self, node, iteration, reward):
        """
        记录执行历史以进行分析
        
        Parameters:
        node (LogicAwareMCTSNode): 当前节点
        iteration (int): 当前迭代
        reward (float): 奖励值
        """
        # 检查节点有效性
        if not node or not node.state:
            logger.warning(f"Attempted to record history for invalid node at iteration {iteration}")
            return
        
        # 获取最新的覆盖率 - 优先使用当前最佳覆盖率
        coverage = 0.0
        if hasattr(self, "current_coverage") and self.current_coverage > 0:
            coverage = self.current_coverage
        elif hasattr(self, "best_state") and self.best_state and hasattr(self.best_state, "coverage"):
            coverage = self.best_state.coverage
        elif hasattr(node.state, "coverage"):
            coverage = node.state.coverage
        
        # 获取bug信息 - 优先从verified_bug_methods获取
        bugs_found = 0
        bug_details = []
        detected_bugs = []
        verified_bugs = []
        
        # 从已验证的bug列表获取
        if hasattr(self, "verified_bug_methods") and self.verified_bug_methods:
            # print("--------------------------------")
            # print("verified_bug_methods")
            # print(self.verified_bug_methods)
            # print("--------------------------------")
            # print("--------------------------------")
            for bug_method in self.verified_bug_methods:
                method_name = bug_method.get("method_name", "unknown")
                bug_info = {
                    "method": method_name,
                    "type": bug_method.get("logic_bug_type", "unknown"),
                    "verified": bug_method.get("verified", False),
                    "is_real_bug": bug_method.get("is_real_bug", False)
                }
                
                # print("bug_info")
                # print(bug_info)
                
                bug_details.append(bug_info)
                detected_bugs.append(method_name)
                
                # 如果bug已验证为真实bug
                if bug_info["verified"] and bug_info["is_real_bug"]:
                    verified_bugs.append(method_name)
            # print("--------------------------------")     
            bugs_found = len(detected_bugs)
        
        # 如果还没有已验证的bug，但当前节点状态有bug信息
        if bugs_found == 0 and hasattr(node.state, "logical_bugs") and node.state.logical_bugs:
            # 收集逻辑bug详情
            for bug in node.state.logical_bugs:
                method_name = bug.get("test_method", bug.get("method_name", "unknown"))
                
                bug_info = {
                    "method": method_name,
                    "type": bug.get("logic_bug_type", "unknown"),
                    "verified": bug.get("verified", False),
                    "is_real_bug": bug.get("is_real_bug", False)
                }
                bug_details.append(bug_info)
                detected_bugs.append(method_name)
                
                # 如果是已验证的真实bug
                if bug_info["verified"] and bug_info["is_real_bug"]:
                    verified_bugs.append(method_name)
                    
            bugs_found = len(detected_bugs)
        
        # 获取逻辑模式和分支条件覆盖数据
        logic_pattern_coverage = 0
        branch_condition_coverage = 0
        
        if hasattr(node.state, "covered_logic_patterns"):
            logic_pattern_coverage = len(node.state.covered_logic_patterns)
        
        if hasattr(node.state, "covered_branch_conditions"):
            branch_condition_coverage = len(node.state.covered_branch_conditions)
        
        # 获取总模式和分支数，以提供更多信息
        total_logic_patterns = len(self.logic_patterns) if hasattr(self, "logic_patterns") and self.logic_patterns else 0
        total_branch_conditions = len(self.logic_model.boundary_conditions) if hasattr(self.logic_model, "boundary_conditions") else 10
        
        # 记录节点统计信息，包含更完整的信息
        logger.info(f"Recording history for node (iteration {iteration}): visits={node.visits}, " + 
                f"logic_patterns={logic_pattern_coverage}/{total_logic_patterns}, " +
                f"branch_conditions={branch_condition_coverage}/{total_branch_conditions}, " +
                f"coverage={coverage:.2f}%, bugs={bugs_found}")
        
        # 创建历史条目
        entry = {
            "iteration": iteration,
            "reward": round(float(reward), 5),
            "coverage": coverage,  # 使用正确的覆盖率
            "bugs_found": bugs_found,  # 设置正确的bugs_found数量
            "action": node.action if node.action else "root",
            "logic_pattern_coverage": logic_pattern_coverage,
            "branch_condition_coverage": branch_condition_coverage,
            "visits": node.visits,
            "wins": round(float(hasattr(node, 'wins') and node.wins or (hasattr(node, 'value') and node.value or 0.0)), 5),
            "logic_bug_rewards": round(float(hasattr(node, 'logic_bug_rewards') and node.logic_bug_rewards or 0.0), 5),
            "logic_coverage_rewards": round(float(hasattr(node, 'logic_coverage_rewards') and node.logic_coverage_rewards or 0.0), 5),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detected_bugs": detected_bugs,  # 添加已检测到的bug
            "verified_bugs": verified_bugs,  # 添加已验证的bug
            "bug_details": bug_details,  # 添加bug详情
            "test_code": node.state.test_code if hasattr(node.state, "test_code") else None  # 保存当前节点的测试代码
        }
        
        # 添加到历史
        self.history.append(entry)
        
        # 记录UCB分数分布以供分析
        if node.parent and node.parent.children:
            scores = []
            for child in node.parent.children:
                if child.visits > 0:
                    # 根据节点类型选择正确的属性
                    win_value = hasattr(child, 'wins') and child.wins or (hasattr(child, 'value') and child.value or 0.0)
                    exploitation = win_value / child.visits
                    exploration = self.exploration_weight * (2 * (node.parent.visits / child.visits) ** 0.5)
                    
                    # 针对LogicAwareMCTSNode的逻辑奖励
                    logic_bonus = 0.0
                    if hasattr(child, 'logic_bug_rewards') and hasattr(child, 'logic_coverage_rewards') and hasattr(child, 'high_risk_pattern_rewards'):
                        logic_bonus = self.logic_weight * (
                            (child.logic_bug_rewards / child.visits) + 
                            (child.logic_coverage_rewards / child.visits) +
                            (child.high_risk_pattern_rewards / child.visits) +
                            (0.2 if hasattr(child, 'is_novel') and child.is_novel else 0.0)
                        )
                    
                    scores.append(exploitation + exploration + logic_bonus)
            
            if scores and hasattr(self, "metrics"):
                self.metrics["ucb_score_distribution"].append({
                    "iteration": iteration,
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores),
                    "count": len(scores)
                })



    def check_termination(self, iteration):
        """
        Check if search should terminate early
        
        Parameters:
        iteration (int): Current iteration
        
        Returns:
        bool: True if search should terminate
        """
        # Check if maximum iterations reached
        if iteration >= self.max_iterations:
            return True
            
        # Check if we've reached target coverage with bugs
        target_coverage = 101.0  # Very high coverage threshold
        if self.current_coverage >= target_coverage and self.logical_bugs_found > 0:
            logger.info(f"Reached high coverage ({self.current_coverage}%) with bugs found, terminating early")
            return True
            
        # Check if we've found enough logical bugs
        # Higher threshold to allow more iterations
        if self.logical_bugs_found >= self.logical_bugs_threshold:
            logger.info(f"Found {self.logical_bugs_found} logical bugs (threshold: {self.logical_bugs_threshold}), terminating early")
            return True
            
        # Check if no progress in last 5 iterations
        if iteration > 5 and len(self.history) >= 5:
            last_rewards = [entry["reward"] for entry in self.history[-5:]]
            if all(abs(last_rewards[0] - r) < 0.001 for r in last_rewards[1:]):
                logger.info("No progress in last 5 iterations, terminating early")
                return True
            
        # Continue search
        return False
    
    def save_logic_metrics(self):
        """Save logic-specific metrics for academic evaluation"""
        # Calculate final metrics
        self.metrics["total_test_methods"] = sum(1 for m in self.best_state.test_methods) if self.best_state else 0
        
        # Calculate logical bug detection rate
        if self.metrics["total_test_methods"] > 0:
            self.metrics["logical_bug_detection_rate"] = (
                self.metrics["total_logical_bug_tests"] / self.metrics["total_test_methods"]
            )
        else:
            self.metrics["logical_bug_detection_rate"] = 0.0
        
        # Convert any sets to lists for JSON serialization
        metrics_copy = {}
        for key, value in self.metrics.items():
            if isinstance(value, set):
                metrics_copy[key] = list(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                metrics_copy[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, set):
                        metrics_copy[key][subkey] = list(subvalue)
                    else:
                        metrics_copy[key][subkey] = subvalue
            else:
                metrics_copy[key] = value
                
        # Save metrics to file
        metrics_file = os.path.join(self.project_dir, f"{self.class_name}_logic_metrics.json")
        try:
            import json
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_copy, f, indent=2)
            logger.info(f"Saved logic metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save logic metrics: {str(e)}")
    
    def selection(self, node):
        """
        Select a promising node for expansion
        
        Returns:
        LogicAwareMCTSNode: Selected node
        """
        # Start from provided node (usually root)
        current = node
        
        # Record path for logging
        path = []
        path.append("root")
        
        # Add: Strategy history tracking
        action_type_history = []
        
        # Add diversity factor
        current_iteration = getattr(self, 'current_iteration', 0)
        force_exploration = (current_iteration % 3 == 0)  # Force exploration every 3 iterations
        
        # Select child node with highest UCB score until reaching leaf or partially expanded node
        while current.is_fully_expanded() and current.children:
            # If forced exploration, use different selection strategy
            if force_exploration and len(current.children) > 1:
                # Calculate UCB scores for all children
                child_scores = []
                for child in current.children:
                    if child.visits > 0:
                        exploitation = child.wins / child.visits
                        exploration = self.exploration_weight * (2 * (current.visits / child.visits) ** 0.5)
                        
                        # Logic bonus for LogicAwareMCTSNode
                        logic_bonus = 0.0
                        if hasattr(child, 'logic_bug_rewards') and hasattr(child, 'logic_coverage_rewards'):
                            logic_bug_term = child.logic_bug_rewards / child.visits
                            logic_coverage_term = child.logic_coverage_rewards / child.visits
                            logic_bonus = self.logic_weight * (logic_bug_term + logic_coverage_term)
                        
                        # Add random perturbation to increase diversity
                        random_factor = random.random() * 0.3
                        
                        # Get action type
                        action_type = "unknown"
                        if child.action and isinstance(child.action, dict) and 'type' in child.action:
                            action_type = child.action['type']
                        
                        # Add reward for unused action types
                        diversity_bonus = 0.0
                        if action_type not in action_type_history[-2:] if action_type_history else True:
                            diversity_bonus = 0.2
                        
                        score = exploitation + exploration + logic_bonus + random_factor + diversity_bonus
                        child_scores.append((child, score))
                    else:
                        child_scores.append((child, float('inf')))  # Unvisited nodes have highest score
                
                # Sort by score, but randomly select one of the top 3, not always the best
                sorted_children = sorted(child_scores, key=lambda x: x[1], reverse=True)
                if len(sorted_children) >= 3:
                    # Randomly select one of the top 3
                    idx = random.randint(0, min(2, len(sorted_children)-1))
                    current = sorted_children[idx][0]
                else:
                    # If less than 3, randomly select one
                    current = random.choice([c[0] for c in sorted_children])
            else:
                # Regular selection - check recent strategy selection history
                if len(action_type_history) >= 2:
                    # If two consecutive selections are of the same strategy type, try to encourage diversity
                    recent_actions = action_type_history[-2:]
                    if len(set(recent_actions)) == 1:  # Most recent actions are the same
                        # Temporarily increase exploration weight to encourage diversity
                        temp_exploration_weight = self.exploration_weight * 1.5
                        current = current.best_child(
                            exploration_weight=temp_exploration_weight,
                            logic_weight=self.logic_weight
                        )
                    else:
                        # Regular selection
                        current = current.best_child(
                            exploration_weight=self.exploration_weight,
                            logic_weight=self.logic_weight
                        )
                else:
                    # Regular selection
                    current = current.best_child(
                        exploration_weight=self.exploration_weight,
                        logic_weight=self.logic_weight
                    )
            
            # Add: Record current node's action type
            if current.action and isinstance(current.action, dict) and 'type' in current.action:
                action_type = current.action['type']
                action_type_history.append(action_type)
                # Save last action type for diversity bonus in UCB calculation
                current.parent.last_action_type = action_type
            
            # Add to
            if current.action and isinstance(current.action, dict) and 'type' in current.action:
                action_type = current.action['type']
                path.append(action_type)
            elif current.action:
                path.append(str(current.action))
            else:
                path.append("unknown")
        
        # Log execution path
        path_str = " -> ".join(path)
        logger.info(f"Node execution path: {path_str}")
            
        return current

    def expansion(self, node):
        """
        Expand a node by selecting an unexplored action
        
        Parameters:
        node (LogicAwareMCTSNode): Node to expand
        
        Returns:
        LogicAwareMCTSNode: New expanded node
        """
        # Get possible actions
        possible_actions = node.generate_possible_actions(
            test_prompt=self.test_prompt,
            source_code=self.source_code,
            uncovered_data={"uncovered_lines": node.state.uncovered_lines} if hasattr(node.state, "uncovered_lines") else None,
            logic_model=self.logic_model,  # Add logic model parameter
            logic_patterns=self.logic_patterns,  # Add logic pattern parameter
            strategy_selector=self.strategy_selector  # Add strategy selector parameter
        )
        
        # If no actions, mark as fully expanded and return
        if not possible_actions:
            node.expanded = True
            logger.info("No possible actions, node marked as fully expanded")
            return node
            
        # Select action (random selection for expansion)
        # print("--------------------------------")
        # print(f"possible_actions: {possible_actions}")
        # print("--------------------------------")
        action = random.choice(possible_actions)

        # print("--------------------------------")
        # print(f"action: {action}")
        # print("--------------------------------")
        
        # Log the selected action in detail
        if isinstance(action, dict):
            action_type = action.get("type", "unknown")
            action_desc = action.get("description", "No description")
            
            # Log more details based on action type
            if action_type == "boundary_test" and "condition" in action and "line" in action:
                logger.info(f"Selected action: {action_type} - Target condition at line {action['line']}: {action['condition']}")
            elif action_type == "logical_expression_test" and "operation" in action and "line" in action:
                logger.info(f"Selected action: {action_type} - Target operation at line {action['line']}: {action['operation']}")
            elif action_type == "target_line" and "line" in action and "content" in action:
                logger.info(f"Selected action: {action_type} - Target line {action['line']}: {action['content']}")
            elif action_type == "bug_pattern_test" and "pattern_type" in action:
                logger.info(f"Selected action: {action_type} - Target pattern: {action['pattern_type']}")
            else:
                logger.info(f"Selected action: {action_type} - {action_desc}")
        else:
            logger.info(f"Selected action: {action}")
        
        # Create new test state
        node.used_action.append(action)
        new_state = self._apply_action(node.state, action)
        
        # Check if state creation failed
        if not new_state:
            logger.warning(f"Failed to create new state for action: {action}")
            # Mark as expanded if all actions have been tried
            if len(node.children) >= len(possible_actions):
                node.expanded = True
            return node
            
        # Create child node
        child_node = node.add_child(new_state, action)
        
        # Mark node as fully expanded if all actions have been tried
        if len(node.children) >= len(possible_actions):
            node.expanded = True
            
        # Update strategy effectiveness metrics
        if "strategy" in action:
            strategy = action["strategy"]
            self.metrics["strategy_effectiveness"][strategy]["used"] += 1
            
        return child_node
        
    def _apply_action(self, state, action):
        """
        Apply an action to create a new test state
        
        Parameters:
        state (LogicAwareTestState): Current state
        action (dict): Action to apply
        
        Returns:
        LogicAwareTestState: New state or None if failed
        """
        if not state:
            return None
            
        try:
            # Log start of action application
            action_type = action.get("type", "unknown") if isinstance(action, dict) else str(action)
            logger.info(f"Applying action: {action_type}")
            
            # More detailed logging based on action type
            if isinstance(action, dict):
                if "strategy" in action:
                    logger.info(f"Using strategy: {action['strategy']}")
                if "description" in action:
                    logger.info(f"Action description: {action['description']}")
            
            # Generate prompt for the action
            if action_type == "business_logic_test":
                # Create a special prompt for business logic issues
                prompt = self._create_business_logic_test_prompt(state, action)
            else:
                # Regular actions use the normal prompt creation
                prompt = self.create_logic_aware_action_prompt(state, action)
            
            # Use LLM to generate new test code
            from feedback import call_anthropic_api, call_gpt_api, call_deepseek_api, extract_java_code
            
            # Log LLM call
            logger.info(f"Calling LLM API to generate test code for action: {action_type}")
            
            # Call LLM API
            llm_response = call_anthropic_api(prompt)
            # llm_response = call_gpt_api(prompt)
            # llm_response = call_deepseek_api(prompt)
            # Extract test code
            new_test_code = extract_java_code(llm_response)
            
            # Check if code extraction failed
            if not new_test_code:
                logger.warning(f"Failed to extract test code from LLM response: {action}")
                return None
                
            # Log code size instead of entire code
            code_size = len(new_test_code)
            logger.info(f"Generated test code size: {code_size} characters")
                
            # Save current coverage to restore in new state
            previous_coverage = getattr(state, "coverage", 0.0)
            previous_patterns = getattr(state, "covered_logic_patterns", set()) if hasattr(state, "covered_logic_patterns") else set()
            previous_conditions = getattr(state, "covered_branch_conditions", set()) if hasattr(state, "covered_branch_conditions") else set()
                
            # Create new test state
            new_state = LogicAwareTestState(
                test_code=new_test_code,
                class_name=self.class_name,
                package_name=self.package_name,
                project_dir=self.project_dir,
                source_code=self.source_code,
                logic_model=self.logic_model,
                logic_patterns=self.logic_patterns
            )
            
            # Add metadata about the action that generated this state
            new_state.metadata = {
                "action": action,
                "parent_coverage": previous_coverage,
                "generation_method": "logic_aware_mcts"
            }

            # NEW: Pass business logic analysis to new state
            if hasattr(state, 'business_logic_analysis'):
                new_state.business_logic_analysis = state.business_logic_analysis
            
            # Handle compilation errors in parent state so new state knows about them
            if hasattr(state, "compilation_errors") and state.compilation_errors:
                new_state.previous_compilation_errors = state.compilation_errors
            
            # Initialize coverage and pattern coverage sets before evaluation
            if previous_coverage > 0:
                new_state.coverage = previous_coverage
            
            if previous_patterns:
                new_state.covered_logic_patterns = previous_patterns.copy()
                
            if previous_conditions:
                new_state.covered_branch_conditions = previous_conditions.copy()
            
            # Evaluate new state
            logger.info(f"Evaluating new state for action: {action_type}")
            new_state.evaluate(verify_bugs=self.verify_bugs_mode == "immediate")
            
            # Check if we've successfully fixed compilation errors
            if action_type == "fix_compilation_errors":
                if hasattr(new_state, "compilation_errors") and new_state.compilation_errors:
                    logger.warning(f"Compilation errors still exist after fix attempt: {new_state.compilation_errors[:2]}")
                else:
                    logger.info("Successfully fixed compilation errors!")
            
            # Ensure coverage is not lost after evaluation
            if not hasattr(new_state, "coverage") or new_state.coverage <= 0:
                new_state.coverage = previous_coverage
                logger.debug(f"Restored previous coverage {previous_coverage} after evaluation")
            
            # Log evaluation results for new state
            new_coverage = getattr(new_state, "coverage", 0.0)
            has_bugs = hasattr(new_state, "has_logical_bugs") and new_state.has_logical_bugs
            bug_count = getattr(new_state, "count_logical_bugs", lambda: 0)() if has_bugs else 0
            
            logger.info(f"Action {action_type} result: coverage={new_coverage:.2f}%, " +
                      f"found logical bugs: {has_bugs} (count: {bug_count})")
            
            return new_state
                
        except Exception as e:
            logger.error(f"Error applying action: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    

    def _create_business_logic_test_prompt(self, state, action):
        """
        Create a specialized prompt for testing business logic issues
        
        Parameters:
        state (LogicAwareTestState): Current state
        action (dict): Business logic action
        
        Returns:
        str: Generated prompt
        """
        issue_type = action.get("issue_type", "unknown")
        issue_method = action.get("method", "")
        issue_description = action.get("description", "")
        
        # Extract more details about the issue from business logic analysis
        issue_details = {}
        if hasattr(state, 'business_logic_analysis'):
            for issue in state.business_logic_analysis.get('potential_bugs', []):
                if issue.get('method') == issue_method and issue.get('type') == issue_type:
                    issue_details = issue
                    break
        
        semantic_signals = issue_details.get('semantic_signals', {})
        implementation_features = issue_details.get('implementation_features', {})
        
        # Build specialized prompt
        prompt = f"""
CRITICAL REQUIREMENTS:
1. DO NOT use @Nested annotations or nested test classes - they cause coverage tracking issues
2. Generate a COMPLETE test class with ALL methods intact - do not omit any code
3. DO NOT use placeholders like "... existing code ..." or similar comments
4. Your response MUST contain the ENTIRE test class that can compile without modifications

You are an expert Java test engineer focusing on detecting BUSINESS LOGIC BUGS.
You need to extend the following test class for {self.class_name} to find a specific business logic bug.

BUSINESS LOGIC ISSUE DETAILS:
- Method with potential issue: {issue_method}
- Issue type: {issue_type}
- Description: {issue_description}
- Expected behavior: {semantic_signals.get('expected_behavior', 'Not specified')}
- Actual behavior: {semantic_signals.get('actual_behavior', 'Not specified')}
- Specifically test: {issue_details.get('test_strategy', 'all edge cases and logical conditions')}

Current test coverage: {state.coverage:.2f}%

Here is the existing test code:
```java
{state.test_code}
```
Here is the source code being tested:
```java
{self.source_code}
```

Your task is to:

Add ONE new test method specifically targeting the business logic inconsistency described above
Design test cases that would reveal the issue
Create detailed assertions that verify correct business logic
Add clear comments explaining your test strategy

IMPORTANT TIPS:

Test edge cases and boundary conditions
Test with inputs that could trigger the logical inconsistency
Verify method outputs against expected business logic, not just against arbitrary values
Include assertions that would fail if the business logic is incorrect

Return the COMPLETE test class including all existing methods plus your new test.
"""

        return prompt



    def create_logic_aware_action_prompt(self, state, action):
        """
        Create a prompt for the given action with logic awareness
        
        Parameters:
        state (LogicAwareTestState): Current state
        action (dict): Action to apply
        
        Returns:
        str: Generated prompt
        """
        action_type = action.get("type", "fallback")
        
        # Base prompt with strong warnings about nested classes and complete code
        prompt = f"""
CRITICAL REQUIREMENTS:
1. DO NOT use @Nested annotations or nested test classes - they cause coverage tracking issues
2. Generate a COMPLETE test class with ALL methods intact - do not omit any code
3. DO NOT use placeholders like "... existing code ..." or similar comments
4. Your response MUST contain the ENTIRE test class that can compile without modifications

You are an expert Java test engineer focusing on detecting logical bugs.
You need to extend the following test class for {self.class_name} to find bugs.

Focus specifically on finding logical bugs related to:
1. Boundary conditions
2. Boolean logic errors
3. Operator precedence issues
4. Off-by-one errors
5. Null handling problems
6. Resource management issues
7. Exception handling defects
8. Data operation bugs

Current test coverage: {state.coverage:.2f}%

Here is the existing test code:
```java
{state.test_code}
```
"""

        # Add source code context
        prompt += f"""
Here is the source code being tested:
```java
{self.source_code}
```
"""

        # Add action-specific instructions
        if action_type == "fix_compilation_errors":
            errors = action.get("errors", [])
            error_text = "\n".join([f"- {err}" for err in errors[:5]])  # Show up to 5 errors
            
            prompt += f"""
SPECIFIC FOCUS: Fix the compilation errors in the test code.

The current code has the following compilation errors:
{error_text}

Your task is to:
1. Analyze the errors carefully
2. Fix the compilation issues while maintaining the original test functionality
3. Make sure the code follows correct Java syntax and JUnit conventions
4. Ensure the entire test class will compile without errors
5. Do not remove any existing test methods, just fix them
6. If you need to add imports, add them at the beginning of the file
7. Fix issues with incorrect method signatures, missing imports, syntax errors, etc.

Your priority is to make the test compile correctly. Do not worry about adding new tests in this step.
"""

        elif action_type == "boundary_test":
            condition = action.get("condition", "")
            line = action.get("line", 0)
            
            prompt += f"""
SPECIFIC FOCUS: Add a test method that thoroughly tests the boundary condition at line {line}:
`{condition}`

Create test cases that check behavior at and around the boundary with various inputs.
Focus on edge cases that might cause logical errors in this condition.
"""

        elif action_type == "logical_expression_test":
            operation = action.get("operation", "")
            line = action.get("line", 0)
            
            prompt += f"""
SPECIFIC FOCUS: Add a test method that thoroughly tests the logical expression at line {line}:
`{operation}`

Create test cases that exercise all paths through this logical expression.
Focus on combinations of inputs that might cause incorrect evaluations.
"""

        elif action_type == "exception_test":
            prompt += """
SPECIFIC FOCUS: Add a test method that tests exception handling paths.

Create test cases that trigger and verify exception handling behavior.
Focus on edge cases where exceptions might be incorrectly handled or not thrown when they should be.
"""

        elif action_type == "data_validation_test":
            prompt += """
SPECIFIC FOCUS: Add a test method that tests data validation logic.

Create test cases with invalid, boundary, and special input data.
Focus on finding cases where validation logic might be incorrect or incomplete.
"""

        elif action_type == "resource_management_test":
            prompt += """
SPECIFIC FOCUS: Add a test method that tests resource management.

Create test cases that verify resources are properly acquired, used, and released.
Focus on edge cases where resource leaks or improper cleanup might occur.
"""

        elif action_type == "state_transition_test":
            prompt += """
SPECIFIC FOCUS: Add a test method that tests state transitions.

Create test cases that verify object state changes correctly after operations.
Focus on sequences of operations that might leave objects in incorrect states.
"""

        elif action_type == "bug_pattern_test":
            pattern_type = action.get("pattern_type", "unknown")
            line = action.get("line", 0)
            
            prompt += f"""
SPECIFIC FOCUS: Add a test method that tests for potential {pattern_type} bug at line {line}.

Create test cases specifically designed to detect this type of logical bug.
Focus on inputs and scenarios that might trigger the issue.
"""

        elif action_type == "target_line":
            line = action.get("line", 0)
            content = action.get("content", "")
            
            prompt += f"""
SPECIFIC FOCUS: Add a test method that targets the uncovered code at line {line}:
`{content}`

Create test cases that execute this specific line of code.
Focus on inputs and scenarios that might reveal logical bugs in this section.
"""

        else:  # general_exploration or fallback
            prompt += """
SPECIFIC FOCUS: Add a test method that explores untested functionality.

Create test cases that increase code coverage and check for logical correctness.
Focus on areas that might contain logical bugs based on your expertise.
"""

        # Add information about any bugs found so far
        if hasattr(state, "logical_bugs") and state.logical_bugs:
            prompt += "\nNOTE: The following logical bugs have already been found:\n"
            for i, bug in enumerate(state.logical_bugs[:3]):  # Limit to first 3
                bug_type = bug.get("logic_bug_type", "unknown")
                prompt += f"- {bug_type} bug\n"

        # Add hints about uncovered branches if available
        if hasattr(state, "uncovered_branches") and state.uncovered_branches:
            prompt += "\nThe following branches remain uncovered:\n"
            for i, branch in enumerate(state.uncovered_branches[:3]):  # Limit to first 3
                prompt += f"- {branch}\n"

        # Add final instructions
        prompt += """
INSTRUCTIONS:
1. Create a SINGLE new test method with a meaningful name
2. Focus on finding logical bugs rather than just increasing coverage
3. Add detailed assertions that verify correct behavior
4. Return the COMPLETE test class, including both existing and new methods
5. Keep existing test methods unchanged - only add new ones
6. Make sure the code is valid Java and will compile without errors
7. Follow JUnit best practices for test method organization
8. Each assertion should have a descriptive message explaining what is being checked
9. DO NOT use @Nested annotations or nested test classes - they cause coverage tracking issues
10. Use straightforward test methods without nesting to ensure proper coverage tracking
11. Always provide the entire test class, do not omit any parts or use placeholders

IMPORTANT: YOUR RESPONSE MUST CONTAIN THE COMPLETE TEST CLASS CODE WITH ALL METHODS. 
DO NOT USE COMMENTS LIKE "... existing code ..." OR SIMILAR PLACEHOLDERS THAT OMIT PARTS OF THE CODE.
"""

        return prompt
    
    def simulation(self, node):
        """
        Simulate from node to estimate value, but only detect bugs without verification
        
        Parameters:
        node (LogicAwareMCTSNode): Node to simulate
        
        Returns:
        float: Reward value
        """
        # If node has state, use it to calculate reward
        if node.state:
            # Collect potential bugs, but do not verify immediately
            if hasattr(node.state, "detected_bugs") and node.state.detected_bugs:
                for bug in node.state.detected_bugs:
                    # Create bug information
                    bug_info = {
                        "test_method": bug.get("test_method", "unknown"),
                        "bug_type": bug.get("type", "unknown"),
                        "error": bug.get("error", ""),
                        "severity": bug.get("severity", "medium"),
                        "method_code": self._extract_method_from_test_code(node.state.test_code, bug.get("test_method", "")),
                        "found_in_iteration": getattr(self, "current_iteration", 0),
                        "original_test_code": node.state.test_code  # Save full original test code
                    }
                    
                    # Create bug signature for deduplication
                    bug_signature = self._create_bug_signature(bug_info)
                    
                    # If this is a new bug signature, add to candidate list
                    if bug_signature not in self.potential_bug_signatures:
                        bug_info["bug_signature"] = bug_signature
                        self.potential_bug_signatures.add(bug_signature)
                        self.potential_bugs.append(bug_info)
                        logger.info(f"Detected potential bug: {bug_info['test_method']} (type: {bug_info['bug_type']})")
                
                # Still consider unverified bug count when calculating reward
                reward = self.calculate_logic_aware_reward(node.state)
                return reward
            
            # No bugs but has state, calculate reward normally
            reward = self.calculate_logic_aware_reward(node.state)
            return reward
        
        # No state, return zero reward
        return 0.0




    def backpropagation(self, node, reward):
        """
        Backpropagate reward through the tree
        
        Parameters:
        node (LogicAwareMCTSNode): Node to start backpropagation from
        reward (float): Reward value
        """
        # Get bug type if available
        bug_type = None
        pattern_coverage = None
        branch_coverage = None
        
        if node.state:
            # Extract coverage data
            if hasattr(node.state, "covered_logic_patterns"):
                pattern_coverage = node.state.covered_logic_patterns
            
            if hasattr(node.state, "covered_branch_conditions"):
                branch_coverage = node.state.covered_branch_conditions
            
            # Extract bug type if available
            if hasattr(node.state, "has_logical_bugs") and node.state.has_logical_bugs:
                # Use the most severe bug type for backpropagation
                for bug in node.state.logical_bugs:
                    if bug.get("severity", "medium") == "high":
                        bug_type = f"logical_{bug.get('logic_bug_type', 'unknown')}"
                        break
            
                if not bug_type and node.state.logical_bugs:
                    bug_type = f"logical_{node.state.logical_bugs[0].get('logic_bug_type', 'unknown')}"
        
        # Log backpropagation data for debugging
        logger.info(f"Starting backpropagation with reward={reward:.4f}, patterns={len(pattern_coverage) if pattern_coverage else 0}, " +
                   f"branches={len(branch_coverage) if branch_coverage else 0}, bug_type={bug_type}")
        
        # Backpropagate reward and coverage data
        current = node
        path = []
        
        while current:
            # Track the path for logging
            if isinstance(current.action, dict) and 'type' in current.action:
                path.append(current.action['type'])
            elif current.action:
                path.append(str(current.action))
            else:
                path.append("root")
                
            # Use more comprehensive update method that passes coverage data
            old_visits = current.visits
            old_wins = current.wins
            old_logic_rewards = getattr(current, 'logic_bug_rewards', 0.0)
            
            if hasattr(current, 'logic_bug_rewards'):
                current.update(reward, bug_type, pattern_coverage, branch_coverage)
            else:
                # Basic update for non-logic nodes
                current.update(reward)
            
            # Log the update for this node
            if hasattr(current, 'logic_bug_rewards'):
                logger.debug(f"Node updated: visits {old_visits}->{current.visits}, " +
                           f"wins {old_wins:.4f}->{current.wins:.4f}, " +
                           f"logic_rewards {old_logic_rewards:.4f}->{current.logic_bug_rewards:.4f}")
            else:
                logger.debug(f"Node updated: visits {old_visits}->{current.visits}, " +
                           f"wins {old_wins:.4f}->{current.wins:.4f}")
            
            current = current.parent
        
        # Log the backpropagation path
        path.reverse()  # Reverse to show root-to-node path
        path_str = " -> ".join(path)
        logger.info(f"Backpropagation path: {path_str}")
    
    # def calculate_logic_aware_reward(self, state, parent_state=None):
    #     """
    #     Calculate reward with logic-specific components
        
    #     Parameters:
    #     state (LogicAwareTestState): Test state
    #     parent_state (LogicAwareTestState): Parent state for comparison
        
    #     Returns:
    #     float: Calculated reward
    #     """
    #     if not state:
    #         return 0.0
        
    #     # Check for compilation errors
    #     has_compilation_errors = hasattr(state, "compilation_errors") and state.compilation_errors
        
    #     # If this is a fix_compilation_errors action, we want to reward based on whether it fixed the errors
    #     if hasattr(state, "metadata") and state.metadata and state.metadata.get("action", {}).get("type") == "fix_compilation_errors":
    #         # If it had errors before but now they're fixed, give a high reward
    #         had_errors_before = (hasattr(state, "previous_compilation_errors") and state.previous_compilation_errors)
    #         if had_errors_before and not has_compilation_errors:
    #             logger.info("High reward for successfully fixing compilation errors")
    #             return 2.0  # High reward for fixing compilation errors
    #         elif has_compilation_errors:
    #             logger.info("Low reward for failing to fix compilation errors")
    #             return 0.1  # Low reward for failing
        
    #     # For normal testing actions, if there are compilation errors, give low reward
    #     if has_compilation_errors:
    #         logger.info("Low reward due to compilation errors")
    #         return 0.05  # Very low reward when there are compilation errors
            
    #     # Base reward components
    #     coverage_reward = state.coverage / 100.0  # 0.0 to 1.0
        
    #     # Additional coverage reward if parent is provided
    #     coverage_improvement = 0.0
    #     if parent_state and hasattr(parent_state, "coverage"):
    #         coverage_delta = state.coverage - parent_state.coverage
    #         if coverage_delta > 0:
    #             coverage_improvement = coverage_delta / 10.0  # Scale improvement
        
    #     # Bug detection rewards
    #     bug_reward = 0.0
    #     if state.detected_bugs:
    #         # Basic reward for any bugs
    #         bug_reward = 0.5
            
    #         # Bonus for logical bugs
    #         if state.has_logical_bugs:
    #             logical_bug_count = state.count_logical_bugs()
    #             bug_reward += 0.3 * logical_bug_count
                
    #             # Extra bonus for certain high-value bug types
    #             for bug in state.logical_bugs:
    #                 bug_type = bug.get("logic_bug_type", "")
    #                 if bug_type in ["boundary_error", "boolean_logic", "operator_logic"]:
    #                     bug_reward += 0.2
    #                 elif bug_type in ["resource_leak", "concurrency_issue", "state_corruption"]:
    #                     bug_reward += 0.3
                        
    #     # Logic coverage rewards
    #     logic_coverage_reward = 0.0
    #     if hasattr(state, "covered_logic_patterns"):
    #         pattern_coverage = len(state.covered_logic_patterns) / max(len(self.logic_patterns), 1)
    #         logic_coverage_reward += pattern_coverage * 0.5
            
    #         # Bonus for covering high-risk patterns
    #         high_risk_patterns = [p for p in self.logic_patterns if p.get("risk_level") == "high"]
    #         if high_risk_patterns:
    #             high_risk_covered = len([
    #                 pattern_id for pattern_id in state.covered_logic_patterns
    #                 if any(f"{p['type']}_{p['location']}" == pattern_id for p in high_risk_patterns)
    #             ])
    #             high_risk_ratio = high_risk_covered / len(high_risk_patterns)
    #             logic_coverage_reward += high_risk_ratio * 0.3
                
    #     # Branch condition rewards
    #     branch_reward = 0.0
    #     if hasattr(state, "covered_branch_conditions") and self.logic_model:
    #         boundary_conditions = [c for c in self.logic_model.boundary_conditions 
    #                            if c.get("type") in ["if_condition", "while_loop", "for_loop"]]
            
    #         if boundary_conditions:
    #             covered_ratio = len(state.covered_branch_conditions) / len(boundary_conditions)
    #             branch_reward = covered_ratio * 0.4
                
    #     # Test quality rewards
    #     quality_reward = 0.0
        
    #     # Reward test diversity
    #     if hasattr(state, "has_boundary_tests") and state.has_boundary_tests:
    #         quality_reward += 0.1
    #     if hasattr(state, "has_boolean_logic_tests") and state.has_boolean_logic_tests:
    #         quality_reward += 0.1
    #     if hasattr(state, "has_state_transition_tests") and state.has_state_transition_tests:
    #         quality_reward += 0.1
    #     if hasattr(state, "has_exception_path_tests") and state.has_exception_path_tests:
    #         quality_reward += 0.1
            
    #     # Combine rewards - adjust weights based on focus
    #     if self.focus_on_bugs:
    #         # When focusing on bugs, prioritize bug detection and logic coverage
    #         combined_reward = (
    #             0.3 * coverage_reward +
    #             0.1 * coverage_improvement +
    #             0.3 * bug_reward +
    #             0.2 * logic_coverage_reward +
    #             0.05 * branch_reward +
    #             0.05 * quality_reward
    #         )
    #     else:
    #         # When focusing on coverage, adjust weights accordingly
    #         combined_reward = (
    #             0.5 * coverage_reward +
    #             0.2 * coverage_improvement +
    #             0.1 * bug_reward +
    #             0.1 * logic_coverage_reward +
    #             0.05 * branch_reward +
    #             0.05 * quality_reward
    #         )
            
    #     return combined_reward

    def calculate_logic_aware_reward(self, state, parent_state=None):
        """
        Calculate logic-aware reward with improved exploration for stagnant coverage
        
        Parameters:
        state (LogicAwareTestState): Test state
        parent_state (LogicAwareTestState): Parent state for comparison
        
        Returns:
        float: Calculated reward
        """
        if not state:
            return 0.0
        
        # Check for compilation errors
        has_compilation_errors = hasattr(state, "compilation_errors") and state.compilation_errors
        
        # If this is a fix_compilation_errors action, reward based on fixing errors
        if hasattr(state, "metadata") and state.metadata and state.metadata.get("action", {}).get("type") == "fix_compilation_errors":
            # If had errors before but now fixed, give high reward
            had_errors_before = (hasattr(state, "previous_compilation_errors") and state.previous_compilation_errors)
            if had_errors_before and not has_compilation_errors:
                logger.info("High reward for successfully fixing compilation errors")
                return 2.0  # High reward for fixing compilation errors
            elif has_compilation_errors:
                logger.info("Low reward for failing to fix compilation errors")
                return 0.1  # Low reward for failing
        
        # For normal testing actions, low reward if compilation errors
        if has_compilation_errors:
            logger.info("Low reward due to compilation errors")
            return 0.05  # Very low reward when there are compilation errors
            
        # Base reward components
        coverage_reward = state.coverage / 100.0  # 0.0 to 1.0
        
        # Track stagnant coverage over iterations
        if not hasattr(state, 'stagnant_coverage_iterations'):
            state.stagnant_coverage_iterations = 0
        
        # NEW: Track if coverage is stagnant
        is_stagnant = False
        
        # Check for coverage improvement
        coverage_improvement = 0.0
        if parent_state and hasattr(parent_state, "coverage"):
            coverage_delta = state.coverage - parent_state.coverage
            if coverage_delta > 0:
                # Reset stagnant counter on improvement
                state.stagnant_coverage_iterations = 0
                coverage_improvement = coverage_delta / 5.0  # Increased scaling
            else:
                # Increment stagnant counter
                state.stagnant_coverage_iterations += 1
                if state.stagnant_coverage_iterations > 3:
                    is_stagnant = True
        

        # Business logic bug detection rewards
        business_logic_reward = 0.0     
        # Bug detection rewards
        bug_reward = 0.0
        if state.detected_bugs:
            for bug in state.detected_bugs:
                if hasattr(state, 'business_logic_analysis'):
                    for issue in state.business_logic_analysis.get('potential_bugs', []):
                        # If detected bug aligns with predicted business logic issue
                        if self._bug_matches_predicted_issue(bug, issue):
                            # Give major reward boost - this is a key success case!
                            business_logic_reward += 1.0 * issue.get('confidence', 0.5)
                            logger.info(f"Detected bug matches predicted business logic issue: +{business_logic_reward} reward")
                            break
            # Basic reward for any bugs
            bug_reward = 0.5
            
            
            # Bonus for logical bugs
            if state.has_logical_bugs:
                logical_bug_count = state.count_logical_bugs()
                bug_reward += 0.4 * logical_bug_count  # Increased from 0.3
                
                # Extra bonus for certain high-value bug types
                for bug in state.logical_bugs:
                    bug_type = bug.get("logic_bug_type", "")
                    if bug_type in ["boundary_error", "boolean_logic", "operator_logic"]:
                        bug_reward += 0.3
                    elif bug_type in ["resource_leak", "concurrency_issue", "state_corruption"]:
                        bug_reward += 0.4
        
        # Logic pattern coverage rewards - MAJOR CHANGES HERE
        logic_coverage_reward = 0.0
        if hasattr(state, "covered_logic_patterns"):
            # Get previous pattern coverage
            previous_pattern_count = 0
            if parent_state and hasattr(parent_state, "covered_logic_patterns"):
                previous_pattern_count = len(parent_state.covered_logic_patterns)
            
            current_pattern_count = len(state.covered_logic_patterns)
            
            # Base pattern coverage (as percentage of total)
            if self.logic_patterns:
                pattern_coverage_pct = current_pattern_count / len(self.logic_patterns)
                logic_coverage_reward += pattern_coverage_pct * 0.8
            
            # NEW: Major reward for new pattern discoveries
            new_patterns = current_pattern_count - previous_pattern_count
            if new_patterns > 0:
                # Reset stagnation counter when finding new patterns
                state.stagnant_coverage_iterations = 0
                
                # Significant reward for each new pattern discovered
                logic_coverage_reward += new_patterns * 0.6
                
                # Track which specific patterns were newly covered
                newly_covered = []
                if parent_state and hasattr(parent_state, "covered_logic_patterns"):
                    newly_covered = [p for p in state.covered_logic_patterns 
                                if p not in parent_state.covered_logic_patterns]
                
                # Extra reward for high risk patterns
                for pattern_id in newly_covered:
                    pattern_type = pattern_id.split('_')[0] if '_' in pattern_id else pattern_id
                    # Check if this is a high risk pattern
                    is_high_risk = any(p.get("risk_level") == "high" and 
                                    p.get("type") == pattern_type 
                                    for p in self.logic_patterns)
                    if is_high_risk:
                        logic_coverage_reward += 0.4
                        logger.info(f"Extra reward for covering high-risk pattern: {pattern_id}")
        
        # Branch condition rewards
        branch_reward = 0.0
        if hasattr(state, "covered_branch_conditions") and self.logic_model:
            # Get previous branch coverage
            previous_branch_count = 0
            if parent_state and hasattr(parent_state, "covered_branch_conditions"):
                previous_branch_count = len(parent_state.covered_branch_conditions)
            
            current_branch_count = len(state.covered_branch_conditions)
            
            # Base branch coverage
            if hasattr(self.logic_model, 'boundary_conditions') and self.logic_model.boundary_conditions:
                covered_ratio = current_branch_count / len(self.logic_model.boundary_conditions)
                branch_reward = covered_ratio * 0.5
                
            # NEW: Reward for newly covered branches
            new_branches = current_branch_count - previous_branch_count
            if new_branches > 0:
                # Additional reward for each new branch covered
                branch_reward += new_branches * 0.2
        
        # Test quality rewards
        quality_reward = 0.0
        
        # Reward test diversity
        if hasattr(state, "has_boundary_tests") and state.has_boundary_tests:
            quality_reward += 0.1
        if hasattr(state, "has_boolean_logic_tests") and state.has_boolean_logic_tests:
            quality_reward += 0.1
        if hasattr(state, "has_state_transition_tests") and state.has_state_transition_tests:
            quality_reward += 0.1
        if hasattr(state, "has_exception_path_tests") and state.has_exception_path_tests:
            quality_reward += 0.1
        
        # NEW: Exploration bonus for stagnant coverage
        exploration_bonus = 0.0
        if is_stagnant:
            # Add increasing exploration bonus based on stagnation length
            exploration_bonus = min(0.5, 0.1 * state.stagnant_coverage_iterations)
            logger.info(f"Adding exploration bonus of {exploration_bonus} after " +
                    f"{state.stagnant_coverage_iterations} stagnant iterations")
        
        # Combine rewards - adjust weights based on focus
        if self.focus_on_bugs:
            # When focusing on bugs, prioritize bug detection and logic coverage
            combined_reward = (
                0.2 * coverage_reward +
                0.15 * coverage_improvement +  # Increased from 0.1
                0.3 * bug_reward +
                0.20 * business_logic_reward +
                0.25 * logic_coverage_reward +  # Increased from 0.2
                0.05 * branch_reward +
                0.05 * quality_reward +
                exploration_bonus  # Add exploration bonus for stagnant coverage
            )
        else:
            # When focusing on coverage, adjust weights accordingly
            combined_reward = (
                0.35 * coverage_reward +
                0.2 * coverage_improvement +
                0.1 * bug_reward +
                0.2 * logic_coverage_reward +
                0.05 * branch_reward +
                0.05 * quality_reward +
                exploration_bonus  # Add exploration bonus for stagnant coverage
            )
        
        # Log detailed reward components for debugging
        if hasattr(self, 'current_iteration') and self.current_iteration % 5 == 0:
            logger.info(f"Reward components: coverage={coverage_reward:.2f}, " +
                    f"improvement={coverage_improvement:.2f}, bug={bug_reward:.2f}, " +
                    f"logic={logic_coverage_reward:.2f}, branch={branch_reward:.2f}, " +
                    f"quality={quality_reward:.2f}, exploration={exploration_bonus:.2f}")
        
        return combined_reward


    def _bug_matches_predicted_issue(self, bug, issue):
        """
        Check if a detected bug matches a predicted business logic issue
        
        Parameters:
        bug (dict): Detected bug
        issue (dict): Predicted business logic issue
        
        Returns:
        bool: True if match found
        """
        # Check method name match
        bug_method = bug.get("test_method", "")
        issue_method = issue.get("method", "")
        if not bug_method or not issue_method:
            return False
        
        # Simplify method name (remove "test" prefix)
        if bug_method.startswith("test"):
            simplified_bug_method = bug_method[4:]
        else:
            simplified_bug_method = bug_method
        
        # If methods don't match, not the same issue
        if issue_method.lower() not in simplified_bug_method.lower() and simplified_bug_method.lower() not in issue_method.lower():
            return False
        
        # Check error message for semantic similarity to issue description
        bug_error = bug.get("error", "") + " " + bug.get("description", "")
        issue_desc = issue.get("description", "")
        
        # Look for keywords from issue in bug error
        issue_keywords = set(re.findall(r'\b\w{4,}\b', issue_desc.lower()))
        if not issue_keywords:
            return False
        
        # Check how many issue keywords appear in the bug error
        error_text = bug_error.lower()
        matches = sum(1 for kw in issue_keywords if kw in error_text)
        
        # Return true if sufficient keyword matches
        return matches >= min(2, len(issue_keywords) // 2)


    def _create_bug_signature(self, bug_info):
        """
        Create unique bug signature for deduplication
        
        Parameters:
        bug_info (dict): Bug information
        
        Returns:
        str: Bug signature
        """
        import hashlib
        import re
        
        method_name = bug_info.get("test_method", "unknown")
        error_msg = bug_info.get("error", "")
        
        # Clean variable parts from error message (e.g., memory addresses)
        cleaned_error = re.sub(r'@[0-9a-f]+', '', error_msg)
        
        # Extract core information from error
        if "expected:" in cleaned_error and "but was:" in cleaned_error:
            # Assertion failure type error
            error_parts = re.search(r'expected:.*?<([^>]+)>.*?but was:.*?<([^>]+)>', cleaned_error)
            if error_parts:
                # Use only core part of error to create signature
                cleaned_error = f"expected:{error_parts.group(1)}_but_was:{error_parts.group(2)}"
        elif "Exception" in cleaned_error:
            # Exception type error
            exception_type = re.search(r'([A-Za-z]+Exception)', cleaned_error)
            if exception_type:
                # Use exception type as core part
                cleaned_error = exception_type.group(1)
        
        # Hash of method name and error core as signature
        signature = f"{method_name}:{hashlib.md5(cleaned_error.encode()).hexdigest()[:12]}"
        return signature

    def _create_robust_bug_signature(self, bug_info):
        """
        Create more robust unique bug signature
        
        Parameters:
        bug_info (dict): Bug information
        
        Returns:
        str: More robust bug signature
        """
        import hashlib
        import re
        
        method_name = bug_info.get("test_method", bug_info.get("method_name", "unknown"))
        bug_type = bug_info.get("bug_type", bug_info.get("logic_bug_type", "unknown"))
        error_msg = bug_info.get("error", "")
        iteration = bug_info.get("found_in_iteration", 0)
        
        # Clean error message
        cleaned_error = re.sub(r'@[0-9a-f]+', '', error_msg)
        cleaned_error = re.sub(r'line\s+\d+', 'line_num', cleaned_error)
        
        # Extract core information from error, use different extraction strategies for different types of errors
        error_essence = ""
        if "expected:" in cleaned_error and "but was:" in cleaned_error:
            # Assertion failure type error
            error_parts = re.search(r'expected:.*?<([^>]+)>.*?but was:.*?<([^>]+)>', cleaned_error)
            if error_parts:
                # Use only core part of error to create signature
                error_essence = f"expected:{error_parts.group(1)}_but_was:{error_parts.group(2)}"
        elif "Exception" in cleaned_error:
            # Exception type error
            exception_type = re.search(r'([A-Za-z]+Exception)', cleaned_error)
            if exception_type:
                # Use exception type as core part
                error_essence = exception_type.group(1)
        else:
            # Other type error, use first 50 characters
            error_essence = cleaned_error[:50]
        
        # Combine key information to create signature
        signature_base = f"{method_name}_{bug_type}_{error_essence}"
        
        # Add iteration number for further differentiation
        if iteration > 0:
            signature_base += f"_iter{iteration}"
        
        # Use hash to create a fixed-length signature
        signature = f"{method_name}_{hashlib.md5(signature_base.encode()).hexdigest()[:12]}"
        
        return signature

    def generate_test_summary(self):
        """
        Generate test summary, including more precise bug statistics
        
        Returns:
        dict: Test summary dictionary
        """
        # Get all verified bugs, including real bugs and false positives
        all_verified_bugs = []
        real_bugs = []
        false_positives = []
        
        # First get all bugs from original verified methods
        if hasattr(self, "original_verified_methods") and self.original_verified_methods:
            for bug in self.original_verified_methods:
                all_verified_bugs.append(bug)
                if bug.get("is_real_bug", False):
                    real_bugs.append(bug)
                else:
                    false_positives.append(bug)
        
        # Then get bugs from verified_bug_methods
        elif hasattr(self, "verified_bug_methods") and self.verified_bug_methods:
            for bug in self.verified_bug_methods:
                all_verified_bugs.append(bug)
                if bug.get("is_real_bug", False):
                    real_bugs.append(bug)
                else:
                    false_positives.append(bug)
        
        # If still no bugs, get from potential_bugs
        if not all_verified_bugs and hasattr(self, "potential_bugs"):
            for bug in self.potential_bugs:
                if bug.get("verified", False):
                    all_verified_bugs.append(bug)
                    if bug.get("is_real_bug", False):
                        real_bugs.append(bug)
                    else:
                        false_positives.append(bug)
        
        # Analyze and count bugs and test methods
        unique_bug_signatures = set()
        unique_test_methods = set()
        
        for bug in all_verified_bugs:
            method_name = bug.get("method_name", "")
            bug_type = bug.get("bug_type", bug.get("logic_bug_type", "unknown"))
            signature = f"{method_name}_{bug_type}"
            
            if method_name:
                unique_test_methods.add(method_name)
            if signature:
                unique_bug_signatures.add(signature)
        
        # Group by bug type
        bugs_by_type = {}
        for bug in all_verified_bugs:
            bug_type = bug.get("bug_type", bug.get("logic_bug_type", "unknown"))
            if bug_type not in bugs_by_type:
                bugs_by_type[bug_type] = {"total": 0, "real": 0, "false_positive": 0}
            
            bugs_by_type[bug_type]["total"] += 1
            if bug.get("is_real_bug", False):
                bugs_by_type[bug_type]["real"] += 1
            else:
                bugs_by_type[bug_type]["false_positive"] += 1
        
        # Check last test coverage
        coverage = self.current_coverage
        
        # Calculate execution time
        execution_time = time.time() - getattr(self, "start_time", time.time())
        
        # Get logical bug types
        logical_bug_types = set()
        if hasattr(self, "metrics") and "logical_bug_types_found" in self.metrics:
            logical_bug_types.update(self.metrics["logical_bug_types_found"])
        
        # Extract unique types from all bugs
        for bug in real_bugs:
            bug_type = bug.get("bug_type", bug.get("logic_bug_type", "unknown"))
            if bug_type != "unknown":
                logical_bug_types.add(bug_type)
        
        # If still no bug types, add a default one
        if not logical_bug_types and len(real_bugs) > 0:
            logical_bug_types.add("logical_error")
        
        # Record real bug and false positive counts
        real_bugs_count = len(real_bugs)
        false_positives_count = len(false_positives)
        
        logger.info(f"Generate test summary, found {real_bugs_count} real bugs and {false_positives_count} false positives")
        
        # Update logical bug count
        self.logical_bugs_found = real_bugs_count
        
        # Generate bug details (including all verified bugs)
        bug_details = self._generate_bug_details()
        
        # Generate bug trend
        bug_trend = self._generate_bug_trend()
        
        # Generate coverage trend
        coverage_trend = self._generate_coverage_trend()
        
        # Create complete summary
        summary = {
            "class_name": self.class_name,
            "package_name": self.package_name,
            "best_coverage": round(coverage, 2) if isinstance(coverage, (int, float)) else 0.0,
            "has_errors": False,
            "iterations": len(self.history),
            "status": "Success" if coverage >= 90 and real_bugs_count > 0 else 
                    "Partial Success" if coverage >= 70 or real_bugs_count > 0 else "Failed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "logical_bugs_found": real_bugs_count,  # 更新为真实bug数量
            "false_positives_found": false_positives_count,  # 添加误报数量
            "logical_bug_types": list(logical_bug_types),
            "bugs_found_iteration": self._get_first_bug_iteration(),
            "bug_details": bug_details,  # 包含所有已验证的bug，包括真实bug和误报
            "history": self.history,
            "coverage_trend": coverage_trend,
            "bug_trend": bug_trend,
            "performance_stats": {
                "avg_execution_time": round(execution_time / max(1, len(self.history)), 2),
                "max_execution_time": round(execution_time, 2),
                "coverage_improvement_rate": self._calculate_coverage_improvement_rate(),
                "final_iterations": len(self.history),
                "real_bugs_to_false_positives_ratio": round(real_bugs_count / max(1, false_positives_count), 2)
            }
        }
                
        return summary


    def _get_first_bug_iteration(self):
        """Get iteration number of first bug found"""
        for i, entry in enumerate(self.history):
            # Check if detected_bugs is a list or an integer
            detected_bugs = entry.get("detected_bugs", 0)
            bugs_found = entry.get("bugs_found", 0)
            
            # Handle detected_bugs being a list
            if isinstance(detected_bugs, list):
                detected_bugs_count = len(detected_bugs)
            else:
                detected_bugs_count = detected_bugs
                
            # Handle bugs_found being a list
            if isinstance(bugs_found, list):
                bugs_found_count = len(bugs_found)
            else:
                bugs_found_count = bugs_found
                
            if detected_bugs_count > 0 or bugs_found_count > 0:
                return entry.get("iteration", i+1)
        return None
        
    def _generate_bug_details(self):
        """
        Generate complete bug details list, including iteration number of each bug
        Include all verified bugs, including real bugs and false positives
        """
        bug_details = []
        processed_methods = set()  # Track processed methods
        
        # Check if there are original verified results
        if hasattr(self, "original_verified_methods") and self.original_verified_methods:
            logger.info(f"从 original_verified_methods 中生成 bug 详情，共有 {len(self.original_verified_methods)} 个方法")
            # print("--------------------------------")
            # print("original_verified_methods")
            # print("--------------------------------")
            for bug in self.original_verified_methods:
                # print(bug)
                # print("--------------------------------")
                
                method_name = bug.get("method_name", "")
                if not method_name or method_name in processed_methods:
                    continue
                
                # Include all verified bugs, whether real or false
                is_real_bug = bug.get("is_real_bug", False)
                # iteration = bug.get("found_in_iteration", 0)
                iteration = 0
                if bug.get("bug_info") and isinstance(bug["bug_info"], list) and len(bug["bug_info"]) > 0:
                    first_bug_info = bug["bug_info"][0]
                    iteration = first_bug_info.get("found_in_iteration", 0)
                else:
                    iteration = bug.get("found_in_iteration", 0)
                bug_type = bug.get("bug_type", bug.get("logic_bug_type", "unknown"))

                bug_details.append({
                    "iteration": iteration,
                    "method": method_name,
                    "type": bug_type,
                    "verified": True,
                    "is_real_bug": is_real_bug
                })
                processed_methods.add(method_name)
                logger.info(f"Add bug: {method_name}, type: {bug_type}, is real bug: {is_real_bug}")
        
        # Get verified bugs from verified_bug_methods
        if hasattr(self, "verified_bug_methods") and self.verified_bug_methods:
            logger.info(f"Generate bug details from verified_bug_methods, there are {len(self.verified_bug_methods)} methods")
            for bug in self.verified_bug_methods:
                # print(bug)
                method_name = bug.get("method_name", "")
                if not method_name or method_name in processed_methods:
                    continue
                
                # Include all verified bugs, whether real or false
                is_real_bug = bug.get("is_real_bug", False)
                iteration = 0
                if bug.get("bug_info") and isinstance(bug["bug_info"], list) and len(bug["bug_info"]) > 0:
                    first_bug_info = bug["bug_info"][0]
                    iteration = first_bug_info.get("found_in_iteration", 0)
                else:
                    iteration = bug.get("found_in_iteration", 0)
                bug_type = bug.get("bug_type", bug.get("logic_bug_type", "unknown"))

                bug_details.append({
                    "iteration": iteration,
                    "method": method_name,
                    "type": bug_type,
                    "verified": True,
                    "is_real_bug": is_real_bug
                })
                processed_methods.add(method_name)
                logger.info(f"Add bug: {method_name}, type: {bug_type}, is real bug: {is_real_bug}")
        
        # Add extra verified bugs from potential_bugs
        if hasattr(self, "potential_bugs") and self.potential_bugs:
            logger.info(f"Generate bug details from potential_bugs, there are {len(self.potential_bugs)} methods")
            for bug in self.potential_bugs:
                test_method = bug.get("test_method", "")
                if not test_method or test_method in processed_methods:
                    continue
                
                # 包含所有已验证的bug，无论是否是真实bug
                if bug.get("verified", False):
                    iteration = 0
                    if bug.get("bug_info") and isinstance(bug["bug_info"], list) and len(bug["bug_info"]) > 0:
                        first_bug_info = bug["bug_info"][0]
                        iteration = first_bug_info.get("found_in_iteration", 0)
                    else:
                        iteration = bug.get("found_in_iteration", 0)
                    bug_type = bug.get("bug_type", "unknown")
                    is_real_bug = bug.get("is_real_bug", False)
                    
                    bug_details.append({
                        "iteration": iteration,
                        "method": test_method,
                        "type": bug_type,
                        "verified": True,
                        "is_real_bug": is_real_bug
                    })
                    processed_methods.add(test_method)
                    logger.info(f"Add bug: {test_method}, type: {bug_type}, is real bug: {is_real_bug}")
        
        # Calculate real bug and false positive counts for logging
        real_bugs = len([b for b in bug_details if b.get("is_real_bug", False)])
        false_positives = len([b for b in bug_details if not b.get("is_real_bug", False)])
        logger.info(f"Finally generate {len(bug_details)} verified bug details, {real_bugs} real bugs, {false_positives} false positives")
        return bug_details


    def save_test_summary(self):
        """
        Generate and save test summary to JSON file
        
        Returns:
        str: Test summary file path
        """
        try:
            # Ensure verified and collected bugs are counted correctly
            if hasattr(self, "potential_bugs") and self.potential_bugs:
                logger.info(f"Checking {len(self.potential_bugs)} potential bugs before generating test summary")
                for bug in self.potential_bugs:
                    method_name = bug.get("method_name", bug.get("test_method", "unknown"))
                    is_real_bug = bug.get("is_real_bug", False)
                    verified = bug.get("verified", False)
                    
                    # If it's a verified real bug, ensure it's added to verified_bug_methods
                    if verified and is_real_bug and method_name != "unknown":
                        if not hasattr(self, "verified_bug_methods"):
                            self.verified_bug_methods = []
                            
                        # Check if it's already in verified_bug_methods
                        if not any(b.get("method_name") == method_name for b in self.verified_bug_methods):
                            logger.info(f"Adding verified real bug to summary: {method_name}")
                            # Fix: ensure method_name field is correctly assigned
                            bug["method_name"] = method_name
                            self.verified_bug_methods.append(bug)
            
            # Generate test summary
            test_summary = self.generate_test_summary()
            
            # Ensure bug_trend and coverage_trend exist
            if "bug_trend" not in test_summary:
                logger.warning("Bug trend missing in test summary, adding empty list")
                test_summary["bug_trend"] = []
                
            if "coverage_trend" not in test_summary:
                logger.warning("Coverage trend missing in test summary, adding empty list")
                test_summary["coverage_trend"] = []
                
            # Calculate real bug count and log
            real_bugs_count = len([b for b in self.verified_bug_methods if b.get("is_real_bug", False) is True]) if hasattr(self, "verified_bug_methods") else 0
            logger.info(f"Found {real_bugs_count} real bugs in total")
                
            # Ensure logical_bugs_found reflects real bug count
            test_summary["logical_bugs_found"] = real_bugs_count
            
            # Determine output filename
            class_name = self.class_name
            summary_file = os.path.join(self.project_dir, f"{class_name}_test_summary.json")
            
            # Save the summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(test_summary, f, indent=2)
                
            logger.info(f"Enhanced test summary with {real_bugs_count} bugs saved to: {summary_file}")
            return summary_file
        except Exception as e:
            logger.error(f"Failed to save test summary: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try
            try:
                minimal_summary = {
                    "class_name": self.class_name,
                    "package_name": self.package_name,
                    "best_coverage": getattr(self, "current_coverage", 0.0),
                    "has_errors": True,
                    "iterations": len(self.history) if hasattr(self, "history") else 0,
                    "status": "Error",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error_message": str(e)
                }
                
                summary_file = os.path.join(self.project_dir, f"{self.class_name}_test_summary.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal_summary, f, indent=2)
                
                logger.info(f"Saved minimal test summary due to error: {summary_file}")
                return summary_file
            except Exception as e2:
                logger.error(f"Failed to save even minimal summary: {str(e2)}")
                return None
        
    
    def generate_integrated_test_code(self):
        """
        Generate integrated test code containing all verified bugs
        
        Returns:
        str: Integrated test code containing all valid bugs
        """
        logger.info(f"Generate integrated test code, merge {len(self.verified_bug_methods)} verified bug tests")
        
        # Find test case with highest coverage as base
        highest_coverage_test = self.best_test
        highest_coverage = self.current_coverage if hasattr(self, "current_coverage") else 0.0
        
        # Find test case with highest coverage from high coverage tests dictionary
        if hasattr(self, "high_coverage_tests") and self.high_coverage_tests:
            best_coverage_key = None
            best_coverage_value = 0.0
            
            for coverage_str, test_code in self.high_coverage_tests.items():
                try:
                    coverage_value = float(coverage_str.replace("_", "."))
                    if coverage_value > best_coverage_value:
                        best_coverage_value = coverage_value
                        best_coverage_key = coverage_str
                except:
                    continue
            
            if best_coverage_key and best_coverage_value > highest_coverage:
                highest_coverage = best_coverage_value
                highest_coverage_test = self.high_coverage_tests[best_coverage_key]
                logger.info(f"Select test code with coverage of {highest_coverage:.2f}% from high coverage tests dictionary")
        
        # Find test case with highest coverage from history
        if hasattr(self, "history") and self.history:
            for entry in self.history:
                if entry.get("coverage", 0.0) > highest_coverage and "test_code" in entry and entry["test_code"]:
                    highest_coverage = entry.get("coverage", 0.0)
                    highest_coverage_test = entry["test_code"]
                    logger.info(f"Select test code with coverage of {highest_coverage:.2f}% from history")
        
        logger.info(f"Select test code with coverage of {highest_coverage:.2f}% as integration base")
        
        # If no verified bugs, return test code with highest coverage
        if not self.verified_bug_methods:
            return highest_coverage_test
            
        # Use test code with highest coverage as base
        base_test_code = highest_coverage_test
        
        # Execute stricter bug deduplication and filtering
        real_bugs = [bug for bug in self.verified_bug_methods if bug.get("is_real_bug", False)]
        logger.info(f"After filtering, there are {len(real_bugs)} real bugs left")
        
        if len(real_bugs) == 0:
            logger.warning("After filtering, there are no real bugs to integrate, return test code with highest coverage")
            return highest_coverage_test
        
        # Extract test class name and package name
        import re
        class_pattern = r"public\s+class\s+(\w+)"
        package_pattern = r"package\s+([\w.]+);"
        
        class_match = re.search(class_pattern, base_test_code)
        package_match = re.search(package_pattern, base_test_code)
        
        test_class_name = class_match.group(1) if class_match else "IntegratedTest"
        package_name = package_match.group(1) if package_match else self.package_name
        
        # Find class end position for insertion
        class_end = base_test_code.rfind('}')
        if class_end == -1:
            class_end = len(base_test_code)
        
        # Collect all methods to add
        added_methods = []
        added_methods_names = set()
        
        # Save all source test codes to improve method extraction opportunities
        all_test_codes = set()
        all_test_codes.add(base_test_code)
        
        # Add all test codes from history
        if hasattr(self, "history") and self.history:
            for entry in self.history:
                if "test_code" in entry and entry["test_code"]:
                    all_test_codes.add(entry["test_code"])
        
        # Add high coverage tests
        if hasattr(self, "high_coverage_tests") and self.high_coverage_tests:
            for test_code in self.high_coverage_tests.values():
                all_test_codes.add(test_code)
        
        # Process each verified real bug test method
        for i, bug in enumerate(real_bugs):
            method_name = bug.get("method_name", "")
            if not method_name or method_name in added_methods_names:
                logger.warning(f"Skip method {method_name}: name is empty or duplicate")
                continue
                
            method_code = bug.get("code", "")
            
            # If no code found, try to extract from multiple sources
            if not method_code:
                # 1. Get from method_code field
                if bug.get("method_code"):
                    method_code = bug["method_code"]
                    logger.info(f"Found method code from method_code field: {method_name}")
                # 2. Extract from various test codes
                else:
                    for test_code in all_test_codes:
                        extracted_code = self._extract_method_from_test_code(test_code, method_name)
                        if extracted_code:
                            method_code = extracted_code
                            logger.info(f"Found method code from test codes: {method_name}")
                            break
                
                # 3. If still not found, check all states
                if not method_code and hasattr(self, "all_states"):
                    for state in self.all_states:
                        if hasattr(state, "test_code") and state.test_code:
                            extracted_code = self._extract_method_from_test_code(state.test_code, method_name)
                            if extracted_code:
                                method_code = extracted_code
                                logger.info(f"Found method code from states: {method_name}")
                                break
                
                # 4. Finally try to extract from final test code
                if not method_code:
                    method_code = self._extract_method_from_test_code(base_test_code, method_name)
                    if method_code:
                        logger.info(f"Found method code from base test code: {method_name}")
            
            # If still no method code found, create a placeholder test method
            if not method_code:
                logger.warning(f"Cannot find method {method_name} code, create placeholder")
                method_code = f"""
        @Test
        public void {method_name}() {{
            // TODO: This is a placeholder for a real bug found during testing
            // Bug type: {bug.get('bug_type', bug.get('logic_bug_type', 'unknown'))}
            // Please implement this test case
            fail("Test not implemented but a real bug was found here");
        }}
    """
                    
            # Ensure method name is not duplicate
            original_method_name = method_name
            counter = 1
            while method_name in added_methods_names:
                method_name = f"{original_method_name}_{counter}"
                counter += 1
                
            added_methods_names.add(method_name)
            
            # If needed, modify method name
            if original_method_name != method_name:
                method_code = method_code.replace(f"public void {original_method_name}", f"public void {method_name}")
                
            # Add method code and comment
            bug_type = bug.get("bug_type", bug.get("logic_bug_type", "unknown"))
            verification_confidence = bug.get("verification_confidence", 0.0)
            
            bug_method_with_comment = f"""
        /**
        * Bug test: {method_name}
        * Bug type: {bug_type}
        * Verification confidence: {verification_confidence:.2f}
        */
    {method_code}
    """
            added_methods.append(bug_method_with_comment)
            logger.info(f"Add method {method_name} to integrated test code")
        
        # If no valid methods, return original code
        if not added_methods:
            logger.warning("No valid bug test methods can be added, return test code with highest coverage")
            return highest_coverage_test
            
        # Check if base test code already contains some method names
        for method_name in list(added_methods_names):
            pattern = r"public\s+void\s+" + re.escape(method_name) + r"\s*\("
            if re.search(pattern, base_test_code):
                logger.info(f"Base test code already contains method {method_name}, skip")
                for i, method in enumerate(added_methods):
                    if f"public void {method_name}" in method:
                        added_methods.pop(i)
                        break
        
        # Insert all methods before class end
        integrated_code = (
            base_test_code[:class_end] + 
            "\n    // ===== 自动生成的Bug测试方法 ===== \n" +
            "".join(added_methods) +
            base_test_code[class_end:]
        )
        
        # Add necessary imports
        if "@Test" not in integrated_code:
            import_pos = integrated_code.find(";") + 1
            integrated_code = (
                integrated_code[:import_pos] + 
                "\n\nimport org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.*;" +
                integrated_code[import_pos:]
            )
            
        # Ensure Exception import
        if "throws Exception" in integrated_code and "import java.lang.Exception;" not in integrated_code:
            import_pos = integrated_code.find(";") + 1
            integrated_code = (
                integrated_code[:import_pos] + 
                "\nimport java.lang.Exception;" +
                integrated_code[import_pos:]
            )
        
        logger.info(f"Successfully generated integrated test code, added {len(added_methods)} bug test methods, based on test with coverage of {highest_coverage:.2f}%")
        
        # Check and fix compilation problems
        logger.info("Check if integrated test code has compilation problems...")
        is_valid, fixed_code = self.verify_integrated_test_compilation(integrated_code)
        
        if not is_valid:
            logger.warning("Integrated test code has compilation problems, try to fix")
            return fixed_code
            
        return integrated_code


    def verify_integrated_test_compilation(self, test_code):
        """
        Verify if integrated test code can be compiled, if not, try to fix
        
        Parameters:
        test_code (str): Integrated test code
        
        Returns:
        tuple: (is_valid, fixed_code) - whether valid, fixed code
        """
        logger.info("Verify integrated test code compilation...")
        
        # First add missing helper methods
        test_code = self._add_missing_helper_methods(test_code)
        
        # Save test code to temporary file
        test_file = save_test_code(
            test_code, 
            self.class_name, 
            self.package_name, 
            self.project_dir
        )
        
        # Try to compile test
        max_attempts = 3
        current_test = test_code
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Compilation attempt #{attempt}")
            
            # Save current version of test code
            save_test_code(
                current_test, 
                self.class_name, 
                self.package_name, 
                self.project_dir
            )
            
            # Run tests to check compilation errors
            _, _, _, compilation_errors = run_tests_with_jacoco(
                self.project_dir, 
                self.class_name, 
                self.package_name, 
                f"{self.package_name}.{self.class_name}Test"
            )
            
            # If no compilation errors, return success
            if not compilation_errors:
                logger.info("Integrated test code compiled successfully!")
                return True, current_test
                
            logger.warning(f"Integrated test code has compilation errors: {compilation_errors[:2]}")
            
            # Extract compilation error types
            duplicate_var_errors = sum(1 for err in compilation_errors if "variable" in err and "already defined" in err)
            symbol_errors = sum(1 for err in compilation_errors if "cannot find symbol" in err)
            
            # If mainly variable duplicate errors, try to enhance variable renaming
            if duplicate_var_errors > 0:
                logger.info(f"Detected {duplicate_var_errors} variable duplicate errors, try to enhance variable renaming")
                
                # Try to apply more aggressive variable renaming
                import re
                
                # Extract
                method_pattern = r'(public\s+void\s+test\w+\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{[^}]*\})'
                test_methods = re.finditer(method_pattern, current_test)
                
                # Store renamed methods
                renamed_methods = []
                used_vars = set()
                
                for i, method_match in enumerate(test_methods):
                    method_code = method_match.group(1)
                    renamed_code, new_vars = self._rename_variables(method_code, used_vars, i * 100 + attempt)
                    renamed_methods.append(renamed_code)
                    used_vars.update(new_vars)
                
                # If found methods and renamed
                if renamed_methods:
                    # Rebuild code, keep class declaration and member variables
                    class_start = current_test.find("public class")
                    class_body_start = current_test.find("{", class_start) + 1
                    
                    # Find start position of first test method
                    first_test_method = re.search(r'public\s+void\s+test\w+\s*\(', current_test)
                    if first_test_method:
                        first_method_start = first_test_method.start()
                        # Extract class header (including member variables)
                        class_header = current_test[class_start:first_method_start]
                        
                        # Rebuild code
                        fixed_code = (
                            current_test[:class_start] + 
                            class_header +
                            "\n    ".join(renamed_methods) +
                            "\n}" # Close class
                        )
                        
                        # Add missing helper methods
                        fixed_code = self._add_missing_helper_methods(fixed_code)
                        
                        current_test = fixed_code
                        continue
            
            # Use LLM to fix compilation errors
            fixed_code = self.fix_integrated_test_with_llm(current_test, compilation_errors)
            
            # If LLM cannot modify code, try next repair method
            if fixed_code == current_test:
                logger.warning("LLM cannot fix code, try next repair method")
                
                # If this is the last attempt, give up and return original best test code
                if attempt == max_attempts:
                    logger.error("Cannot fix integrated test code, give up and return original best test code")
                    return False, self.best_test
            else:
                # Use repaired code to continue
                current_test = fixed_code
                logger.info("Use repaired code to continue")
        
        # After maximum attempts, still not fully repaired
        logger.warning(f"After {max_attempts} attempts, still not fully repaired")
        
        # Return last repaired version
        return False, current_test

    def fix_integrated_test_with_llm(self, test_code, error_message=None):
        """
        Use LLM to fix compilation problems in integrated test code
        
        Parameters:
        test_code (str): Test code
        error_message (list): Error message list, if any
        
        Returns:
        str: Repaired test code
        """
        logger.info("Try to use LLM to fix compilation problems in integrated test code")
        
        # Create prompt - focus on clearly defining LLM's task and providing all necessary context
        prompt = f"""please help me fix the compilation issues in the following JUnit test code. your task is to identify issues such as undeclared variables, missing imports, method conflicts, and provide complete repaired code. i need the complete code, not just the repaired parts.

important note: i need the complete test class, including all original methods, not just the repaired parts.
your answer must include:
1. all package declarations
2. all import statements 
3. complete class definition
4. all existing test methods, not just the repaired ones
5. all fields and setup methods

very important: do not use placeholders or comments, such as "// all existing test methods remain unchanged..." or "// [previous test methods remain unchanged...]".
you must include the original code of all actual code, even if it is not changed. do not accept shortcuts, abbreviations, or comments indicating that code is omitted.
i need the complete code that can be saved to a file and compiled directly.

format your entire answer as a complete, compilable Java file that can be saved and run directly.

class information:
- class name: {self.class_name}
- package name: {self.package_name}

source code:
```java
{self.source_code} 
```

test code:
```java
{test_code}
```

compilation errors:
```
{error_message if error_message else "compilation failed, please check possible issues"}
```

please pay special attention to the following points:
1. check variable declarations and initializations - variables may need to be redeclared in different test methods
2. ensure that integrated test methods do not have variable name conflicts
3. ensure that all necessary imports exist
4. fix method signatures or parameter issues
5. ensure that variable scope is correct within methods

only fix necessary compilation issues, while preserving the original functionality of the test methods.
"""

        # Call LLM API
        try:
            api_response = call_anthropic_api(prompt)
            # api_response = call_deepseek_api(prompt)
            
            if not api_response or len(api_response) < 100:  # Ensure enough response
                logger.warning("LLM response insufficient, try alternative API")
                api_response = call_gpt_api(prompt)
                
            # Extract Java code
            from feedback import extract_java_code
            fixed_code = extract_java_code(api_response)
            
            if not fixed_code or len(fixed_code) < 100:
                logger.warning("Cannot extract valid Java code from LLM response")
                return test_code  # Return original code
                
            logger.info("LLM successfully fixed the integrated test code")
            return fixed_code
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return test_code  # Return original code
        

    def _extract_method_body(self, method_code):
        """
        Extract method body from complete method code
        
        Parameters:
        method_code (str): Complete method code
        
        Returns:
        str: Method body code
        """
        try:
            if not method_code:
                return ""
                
            # Find method body start position (after first left brace)
            body_start = method_code.find("{")
            if body_start == -1:
                logger.warning(f"Cannot find start brace in method code: {method_code[:50]}...")
                return ""
                
            # Find method body end position (matching right brace)
            depth = 1
            body_end = body_start + 1
            
            while depth > 0 and body_end < len(method_code):
                if method_code[body_end] == "{":
                    depth += 1
                elif method_code[body_end] == "}":
                    depth -= 1
                body_end += 1
                    
            if depth != 0:
                logger.warning(f"Method code braces do not match: {method_code[:50]}...")
                # Try to extract body with simpler method
                last_closing_brace = method_code.rfind("}")
                if last_closing_brace > body_start:
                    body_end = last_closing_brace + 1
                else:
                    return ""
                    
            # Extract method
            method_body = method_code[body_start+1:body_end-1]
            
            # Check if extracted method body is empty
            if not method_body.strip():
                logger.warning("Extracted method body is empty")
                return ""
                
            # Process indent (ensure correct indent)
            lines = method_body.split("\n")
            
            # Find minimum non-zero indent
            min_indent = float('inf')
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    if indent > 0:  # Only consider non-zero indent
                        min_indent = min(min_indent, indent)
            
            # If no valid indent found, use default indent of 4 spaces
            if min_indent == float('inf'):
                min_indent = 4
                
            # Standard indent of 8 spaces (standard indent in integrated methods)
            target_indent = 8
            
            # Reformat each line
            processed_lines = []
            for line in lines:
                if not line.strip():
                    processed_lines.append("")  # Keep empty lines
                    continue
                    
                # Calculate current line indent
                current_indent = len(line) - len(line.lstrip())
                
                # Handle special cases: some IDEs may use tabs instead of spaces
                if '\t' in line:
                    # Replace tabs with 4 spaces
                    line = line.replace('\t', '    ')
                    current_indent = len(line) - len(line.lstrip())
                
                if current_indent >= min_indent:
                    # Remove minimum common indent, then add target indent
                    indent_diff = current_indent - min_indent
                    processed_lines.append(" " * target_indent + " " * indent_diff + line.lstrip())
                else:
                    # Apply target indent directly
                    processed_lines.append(" " * target_indent + line.lstrip())
                        
            # Join into final result
            method_body = "\n".join(processed_lines)
                
            # Ensure final output is not empty
            if not method_body.strip():
                logger.warning("Method body after indent processing is empty")
                # Fall back to original extracted method body, apply basic indent
                method_body = "\n".join([" " * target_indent + line.lstrip() for line in method_code[body_start+1:body_end-1].split("\n") if line.strip()])
            
            return method_body
            
        except Exception as e:
            logger.error(f"Error extracting method body: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
