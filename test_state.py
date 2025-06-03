#!/usr/bin/env python3
"""
Logic-Aware Test State

This module extends the TestState class with logic awareness capabilities, allowing
it to track logical properties of tests, detect logical bugs, and maintain 
state information about logic-specific test properties.
"""

import os
import re
import json
import time
import logging
import traceback
from collections import defaultdict
import random

# Import from base TestState implementation
from enhanced_test_state import TestState
from enhanced_mcts_test_generator import TestState as EnhancedTestState
from feedback import save_test_code, run_tests_with_jacoco, get_coverage_percentage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logic_test_state")

class LogicAwareTestState(EnhancedTestState):
    """
    Extended TestState class with logic awareness capabilities for better
    detection of logical bugs and tracking of logical properties.
    """
    
    def __init__(self, test_code, class_name, package_name, project_dir, source_code=None, 
                 logic_model=None, logic_patterns=None):
        """
        Initialize Logic-Aware Test State
        
        Parameters:
        test_code (str): Test code
        class_name (str): Class name
        package_name (str): Package name
        project_dir (str): Project directory
        source_code (str): Source code
        logic_model (LogicModelExtractor): Logic model
        logic_patterns (list): Detected logic bug patterns
        """
        super().__init__(test_code, class_name, package_name, project_dir, source_code)
        
        # Logic-specific properties
        self.logic_model = logic_model
        self.logic_patterns = logic_patterns
        self.logical_bugs = []
        self.has_logical_bugs = False
        self.covered_logic_patterns = set()
        self.covered_branch_conditions = set()
        
        # Initialize additional properties
        self.logical_bug_methods = []
        self.has_boundary_tests = False
        self.has_boolean_logic_tests = False
        self.has_state_transition_tests = False
        self.has_exception_path_tests = False
        self.has_equivalence_class_tests = False
        
        # Risk metrics
        self.risk_score = 0.0
        self.high_risk_patterns_covered = 0
        self.critical_conditions_covered = 0
        
        # Compilation errors tracking
        self.compilation_errors = []
        self.previous_compilation_errors = []
        
        # Initialize test properties tracking
        self.has_assertions = False  # Track if tests have assertions
        
        # For tracking test patterns
        self.boolean_expressions_tested = []
        self.boundary_values_tested = []
        
        # Initialize assertion failures tracking
        self.assertion_failures = []
        
        # Additional metrics
        self.branch_coverage = 0.0
        self.logic_coverage = 0.0
        self.high_risk_pattern_coverage = 0.0
        self.method_complexity_coverage = 0.0
        
        # Ensure uncovered_lines is initialized
        if not hasattr(self, 'uncovered_lines'):
            self.uncovered_lines = []
        
        # Pre-analyze test properties on initialization
        self.analyze_test_logic_properties()
        
        # Log initialization
        logger.info(f"Initialized LogicAwareTestState with {len(self.logic_patterns) if self.logic_patterns else 0} logic patterns")
    
    def evaluate(self, validator=None, verify_bugs=False, current_iteration=None):
        """
        运行测试并使用增强的逻辑错误检测来衡量覆盖率
        
        Parameters:
        validator (TestValidator): 用于修复测试代码的可选验证器
        verify_bugs (bool): 是否立即验证错误
        current_iteration (int): 当前 MCTS 迭代编号
        """
        # 保存测试代码
        test_file = save_test_code(
            self.test_code, 
            self.class_name, 
            self.package_name, 
            self.project_dir
        )
        
        # 保存当前覆盖率，以防测试失败时使用
        previous_coverage = getattr(self, "coverage", 0.0)
        
        # 运行测试并获取 JaCoCo 覆盖率数据
        coverage_data, assertion_failures, execution_time, errors = run_tests_with_jacoco(
            self.project_dir, 
            self.class_name, 
            self.package_name, 
            f"{self.package_name}.{self.class_name}Test"
        )
        
        # Store compilation errors explicitly
        self.compilation_errors = errors if errors else []
        
        # If we have compilation errors, log them and don't proceed with other evaluations
        if self.compilation_errors:
            # logger.warning(f"Test has compilation errors: {self.compilation_errors[:2]}")
            logger.warning(f"Test has compilation errors!!!")
            self.executed = True
            
            # If we had previous coverage, maintain it
            if previous_coverage > 0:
                self.coverage = previous_coverage
            else:
                self.coverage = 0.0
                
            return
            
        # 处理新的覆盖率数据
        new_coverage = get_coverage_percentage(coverage_data)
        # new_coverage = coverage_data['class_summary'].get('INSTRUCTION', {}).get('coverage_percent', 0.0)
        
        # 确保覆盖率值被正确保存和更新
        if new_coverage > 0:
            self.coverage = new_coverage
            logger.debug(f"Updated coverage to {self.coverage}")
        elif previous_coverage > 0:
            # 如果新覆盖率为零但之前有有效覆盖率，保留旧值
            self.coverage = previous_coverage
            logger.debug(f"Maintained previous coverage {self.coverage}")
        else:
            # 默认值为 0.0
            self.coverage = 0.0
        
        self.executed = True
        
        # 明确检查测试结果中的断言失败
        if assertion_failures:
            for result in assertion_failures:
                method_match = re.search(r'Test\.(test\w+)', result)
                if method_match:
                    method_name = method_match.group(1)
                    
                    # 创建错误信息
                    bug_info = {
                        "type": "assertion_failure",
                        "description": result,
                        "test_method": method_name,
                        "error": "AssertionError",
                        "severity": "medium",
                        "verified": True,  # 将断言失败视为预先验证的
                        "is_real_bug": True,  # 断言失败通常是真正的错误
                        "bug_category": "logical",  # 默认标记为逻辑错误
                        "logic_bug_type": "incorrect_behavior"  # 默认类型
                    }
                    
                    # 如果错误还没有被添加，就添加到检测到的错误中
                    if not any(b.get("test_method") == method_name for b in self.detected_bugs):
                        self.detected_bugs.append(bug_info)
                        self.logical_bugs.append(bug_info)
                        self.has_logical_bugs = True
                        logger.info(f"Added assertion failure as logical bug from method: {method_name}")
                        
                        # 也添加到断言失败
                        self.assertion_failures.append({
                            "method": method_name,
                            "message": result
                        })
        
        # 如果执行成功，执行逻辑特定分析
        if self.executed:
            try:
                # 检查测试代码的逻辑特性
                self.analyze_test_logic_properties()
                
                # 识别检测到的错误中的逻辑错误
                self.classify_logical_bugs()
                
                # 跟踪逻辑模式覆盖率
                self.track_logic_pattern_coverage()
                
                # 跟踪分支条件覆盖率
                self.track_branch_condition_coverage()
                
                # 更新风险指标
                self.calculate_risk_metrics()
                
                logger.debug(f"Logic analysis complete: " +
                        f"logical bugs={len(self.logical_bugs)}, " +
                        f"covered patterns={len(self.covered_logic_patterns)}, " +
                        f"covered conditions={len(self.covered_branch_conditions)}")
                
            except Exception as e:
                logger.error(f"Error in logic analysis: {str(e)}")
                logger.error(traceback.format_exc())

    def analyze_test_logic_properties(self):
        """分析测试代码的逻辑特定属性"""
        # 检查布尔逻辑测试
        self.has_boolean_logic_tests = any(
            (("&&" in m.get("code", "") and "||" in m.get("code", "")) or
            ("assertTrue" in m.get("code", "") and "assertFalse" in m.get("code", "")))
            for m in self.test_methods if isinstance(m, dict)
        )
        
        # 检查边界值测试
        self.has_boundary_tests = any(
            (">=" in m.get("code", "") or "<=" in m.get("code", "") or 
            "==" in m.get("code", "") or "!=" in m.get("code", ""))
            for m in self.test_methods if isinstance(m, dict)
        )
        
        # 检查状态转换测试
        self.has_state_transition_tests = any(
            m.get("code", "").count(".") > 5  # 多个链式方法调用通常表示状态转换
            for m in self.test_methods if isinstance(m, dict)
        )
        
        # 检查运算符优先级测试
        self.has_operator_precedence_tests = any(
            ("(" in m.get("code", "") and ")" in m.get("code", "") and 
            ("&&" in m.get("code", "") or "||" in m.get("code", "")))
            for m in self.test_methods if isinstance(m, dict)
        )
        
        # 检查特殊输入测试
        self.has_exception_path_tests = any(
            ("assertThrows" in m.get("code", "") or 
            "try" in m.get("code", "") and "catch" in m.get("code", ""))
            for m in self.test_methods if isinstance(m, dict)
        )
        
        # 提取测试的布尔表达式
        for method in self.test_methods:
            if isinstance(method, dict) and "code" in method:
                method_code = method["code"]
                # 从断言中提取布尔表达式
                boolean_exprs = re.findall(r'assert(?:True|False|Equals)\s*\(\s*([^;]+?&&[^;]+|[^;]+?\|\|[^;]+?)\s*[,\)]', method_code)
                if boolean_exprs:
                    self.boolean_expressions_tested.extend(boolean_exprs)
                
                # 提取边界值测试
                boundary_tests = re.findall(r'assert(?:True|False|Equals)\s*\(\s*[^<>=!]+\s*([<>=!]+)\s*([^,\)]+)', method_code)
                for op, value in boundary_tests:
                    self.boundary_values_tested.append({"operator": op, "value": value.strip()})
        
        # 分析测试的逻辑覆盖深度
        self.logic_coverage_depth = 0
        
        # 检查边界值覆盖
        if len(self.boundary_values_tested) > 0:
            self.logic_coverage_depth += 1
        
        # 检查布尔表达式覆盖
        if len(self.boolean_expressions_tested) > 0:
            self.logic_coverage_depth += 1
        
        # 检查异常路径覆盖
        if self.has_exception_path_tests:
            self.logic_coverage_depth += 1
        
        # 检查复杂逻辑结构覆盖（嵌套条件、多个逻辑操作符等）
        complex_logic = any(
            ("&&" in m.get("code", "") and "||" in m.get("code", "") and "!" in m.get("code", ""))
            for m in self.test_methods if isinstance(m, dict)
        )
        if complex_logic:
            self.logic_coverage_depth += 1
        
        # 检查变异值测试（边界条件的微小变化）
        mutation_testing = any(
            ("+1" in m.get("code", "") and "-1" in m.get("code", "")) or
            ("MIN_VALUE" in m.get("code", "") or "MAX_VALUE" in m.get("code", ""))
            for m in self.test_methods if isinstance(m, dict)
        )
        if mutation_testing:
            self.logic_coverage_depth += 1
            
        # 更新测试质量指标 (0-1范围)
        self.logic_test_quality = min(1.0, self.logic_coverage_depth / 5.0)


    def classify_logical_bugs(self):
        """将检测到的bug分类为逻辑bug"""
        if not self.detected_bugs:
            return
            
        # 定义识别逻辑bug的模式
        logical_bug_patterns = [
            # 断言相关模式
            {"pattern": r'expected:.*?but was', "confidence": 0.7, "bug_type": "incorrect_value"},
            {"pattern": r'expected.*?true.*?but was.*?false|expected.*?false.*?but was.*?true', "confidence": 0.9, "bug_type": "incorrect_boolean"},
            {"pattern": r'expected.*?empty|expected.*?null', "confidence": 0.6, "bug_type": "empty_null_handling"},
            {"pattern": r'IndexOutOfBoundsException|ArrayIndexOutOfBoundsException', "confidence": 0.8, "bug_type": "index_error"},
            {"pattern": r'NullPointerException', "confidence": 0.6, "bug_type": "null_reference"},
            {"pattern": r'ClassCastException', "confidence": 0.7, "bug_type": "incorrect_type"},
            {"pattern": r'UnsupportedOperationException', "confidence": 0.8, "bug_type": "unsupported_operation"},
            {"pattern": r'IllegalArgumentException', "confidence": 0.7, "bug_type": "invalid_argument"},
            {"pattern": r'IllegalStateException', "confidence": 0.8, "bug_type": "invalid_state"},
            {"pattern": r'ConcurrentModificationException', "confidence": 0.9, "bug_type": "concurrency_issue"},
            {"pattern": r'NumberFormatException', "confidence": 0.7, "bug_type": "number_format"},
            
            # 逻辑特定模式
            {"pattern": r'overflow|underflow', "confidence": 0.8, "bug_type": "numeric_overflow"},
            {"pattern": r'boundary|fence.?post|off.by.one', "confidence": 0.9, "bug_type": "boundary_error"},
            {"pattern": r'operator.*?precedence|condition.*?logic', "confidence": 0.8, "bug_type": "operator_logic"},
            {"pattern": r'race.*?condition|deadlock|concurrent', "confidence": 0.9, "bug_type": "concurrency_issue"},
            {"pattern": r'boolean.*?condition|logic.*?error', "confidence": 0.8, "bug_type": "boolean_logic"},
            {"pattern": r'infinite.*?loop', "confidence": 0.9, "bug_type": "infinite_loop"},
            {"pattern": r'resource.*?leak|not.*?closed', "confidence": 0.8, "bug_type": "resource_leak"},
            {"pattern": r'state.*?corruption|invalid.*?state', "confidence": 0.8, "bug_type": "state_corruption"},
            {"pattern": r'assertion.*?fail.*?logic', "confidence": 0.7, "bug_type": "logical_assertion"}
        ]
        
        # 跟踪已添加的测试方法以避免重复
        added_methods = set()
        
        # 检查每个bug是否是逻辑bug
        for bug in self.detected_bugs:
            # 如果此方法已添加，跳过
            test_method = bug.get("test_method", "")
            if test_method in added_methods:
                continue
                
            is_logical = False
            highest_confidence = 0.0
            detected_bug_type = "unknown"
            
            # Bug消息 - 组合错误和描述字段
            bug_message = str(bug.get("error", "")) + " " + str(bug.get("description", ""))
            
            for pattern in logical_bug_patterns:
                if re.search(pattern["pattern"], bug_message, re.IGNORECASE):
                    is_logical = True
                    confidence = pattern["confidence"]
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        detected_bug_type = pattern["bug_type"]
            
            # 根据检测设置bug类别
            if is_logical:
                bug["bug_category"] = "logical"
                bug["logic_bug_type"] = detected_bug_type
                bug["logic_confidence"] = highest_confidence
                self.logical_bugs.append(bug)
                added_methods.add(test_method)
            else:
                bug["bug_category"] = "general"
        
        # 更新逻辑bug标志
        self.has_logical_bugs = len(self.logical_bugs) > 0
        
        if self.has_logical_bugs:
            logger.info(f"将 {len(self.logical_bugs)} 个bug分类为逻辑bug")


    def track_logic_pattern_coverage(self):
        """Track which logical patterns are covered by tests with improved confidence scoring"""
        if not self.logic_patterns:
            return
        
        logger.info(f"Tracking logic pattern coverage, found {len(self.logic_patterns)} patterns")
        
        # Initialize covered_logic_patterns as a dictionary with confidence scores
        # instead of a simple set to allow for confidence-based coverage
        if not hasattr(self, 'covered_logic_patterns_scores'):
            self.covered_logic_patterns_scores = {}
        
        # Initialize the set if not already done
        if not hasattr(self, 'covered_logic_patterns') or self.covered_logic_patterns is None:
            self.covered_logic_patterns = set()
        
        # Save the number of patterns already covered for reporting
        covered_before = len(self.covered_logic_patterns)
        
        # Convert code to lowercase for case-insensitive comparison
        all_test_code = self.test_code.lower() if self.test_code else ""
        
        # Define confidence thresholds based on risk level
        confidence_thresholds = {
            "high": 0.8,    # High-risk patterns need stronger evidence
            "medium": 0.6,  # Medium-risk patterns need moderate evidence
            "low": 0.5      # Low-risk patterns need basic evidence
        }
        
        # Track patterns with updated confidence in this run
        updated_patterns = set()
        
        for pattern in self.logic_patterns:
            pattern_id = f"{pattern['type']}_{pattern['location']}"
            pattern_type = pattern.get("type", "unknown")
            pattern_location = pattern.get("location", 0)
            pattern_risk = pattern.get("risk_level", "medium")
            
            # Get current confidence score or initialize to 0
            current_confidence = self.covered_logic_patterns_scores.get(pattern_id, 0.0)
            
            # Reset confidence slightly over time if not reinforced
            # This allows patterns to be "uncovered" if evidence weakens
            if pattern_id not in updated_patterns and current_confidence > 0:
                # Decay confidence by 5% each time
                new_confidence = current_confidence * 0.95
                self.covered_logic_patterns_scores[pattern_id] = new_confidence
                
                # If confidence drops below threshold, remove from covered set
                if new_confidence < confidence_thresholds.get(pattern_risk, 0.6):
                    if pattern_id in self.covered_logic_patterns:
                        self.covered_logic_patterns.remove(pattern_id)
                        logger.debug(f"Pattern confidence decayed: {pattern_id} removed from covered set")
            
            # 1. Direct line number match - strongest evidence
            if f"line {pattern_location}" in all_test_code or f"行 {pattern_location}" in all_test_code:
                confidence_boost = 0.7
                logger.debug(f"Direct line number match for pattern: {pattern_id}")
            else:
                confidence_boost = 0
            
            # 2. Pattern type keyword matching - moderate evidence
            pattern_keywords = {
                "null_handling": ["null", "nullpointer", "nullpointerexception", "assertnull", "nullcheck"],
                "array_index_bounds": ["index", "bounds", "outofbounds", "array", "arrayindexoutofbounds"],
                "off_by_one": ["boundar", "off by one", "off-by-one", "boundary"],
                "string_comparison": ["string", "equals", "compare", "assertion"],
                "boolean_logic": ["boolean", "logic", "boolean expression", "logical"],
                "boundary_condition": ["boundary", "edge case", "边界条件"],
                "resource_leak": ["resource", "leak", "close"],
                "operator_precedence": ["operator", "precedence"],
                "copy_paste": ["duplicate", "copy", "paste"],
                "integer_overflow": ["overflow", "integer"],
                "bitwise_logical_confusion": ["bitwise", "logical"],
                # Add more pattern types as needed
            }
            
            # Use more specific keyword matching
            keywords = pattern_keywords.get(pattern_type, [pattern_type, "bug", "test", "error"])
            
            # Count how many keywords match rather than just checking if any match
            keyword_matches = sum(1 for keyword in keywords if keyword in all_test_code)
            keyword_confidence = min(0.5, 0.1 * keyword_matches)  # Cap at 0.5
            confidence_boost += keyword_confidence
            
            # 3. Bug detection evidence - stronger for matching bug types
            if hasattr(self, 'logical_bugs') and self.logical_bugs:
                for bug in self.logical_bugs:
                    bug_description = bug.get("description", "").lower()
                    bug_error = bug.get("error", "").lower()
                    bug_type = bug.get("logic_bug_type", "unknown").lower()
                    
                    # More specific matching criteria
                    pattern_in_bug = pattern_type in bug_description or pattern_type in bug_error
                    pattern_related_to_bug_type = pattern_type.replace("_", "") in bug_type
                    
                    # Check for more specific keyword matches in bug details
                    keyword_in_bug = any(keyword in bug_description or keyword in bug_error 
                                for keyword in pattern_keywords.get(pattern_type, []))
                    
                    if pattern_in_bug or pattern_related_to_bug_type or keyword_in_bug:
                        confidence_boost += 0.4
                        logger.debug(f"Bug evidence for pattern {pattern_id}, bug type: {bug_type}")
                        break
            
            # 4. Test method name evidence - weaker
            method_confidence = 0
            for method in self.test_methods:
                if isinstance(method, dict) and "name" in method:
                    method_name = method["name"].lower()
                    # More specific method name matching
                    if pattern_type in method_name or pattern_type.replace("_", "") in method_name:
                        method_confidence = 0.3
                        logger.debug(f"Method name evidence for pattern {pattern_id}: {method_name}")
                        break
            confidence_boost += method_confidence
            
            # REMOVED: No more probabilistic coverage for high test coverage
            # REMOVED: No more probabilistic coverage for multiple bugs
            
            # Update confidence score with evidence from this run
            new_confidence = min(1.0, current_confidence + confidence_boost)
            
            # Only consider the pattern covered if confidence exceeds the risk-based threshold
            threshold = confidence_thresholds.get(pattern_risk, 0.6)
            was_covered = pattern_id in self.covered_logic_patterns
            should_be_covered = new_confidence >= threshold
            
            # Update confidence score
            self.covered_logic_patterns_scores[pattern_id] = new_confidence
            updated_patterns.add(pattern_id)
            
            # Update covered set based on threshold
            if should_be_covered and not was_covered:
                self.covered_logic_patterns.add(pattern_id)
                logger.info(f"Pattern newly covered: {pattern_id} with confidence {new_confidence:.2f}")
            elif was_covered and not should_be_covered:
                self.covered_logic_patterns.remove(pattern_id)
                logger.info(f"Pattern no longer covered: {pattern_id} with confidence {new_confidence:.2f}")
        
        # Record newly covered patterns count
        newly_covered = len(self.covered_logic_patterns) - covered_before
        if newly_covered > 0:
            logger.info(f"Newly covered {newly_covered} logic patterns")
        elif newly_covered < 0:
            logger.info(f"Uncovered {abs(newly_covered)} previously covered patterns due to confidence decay")
        
        # Log coverage statistics
        total_patterns = len(self.logic_patterns)
        covered_count = len(self.covered_logic_patterns)
        logger.info(f"Pattern coverage: {covered_count}/{total_patterns} " +
                f"({covered_count/total_patterns*100:.1f}%)")
        
        # Record pattern coverage levels for monitoring
        high_risk_patterns = [p for p in self.logic_patterns if p.get("risk_level") == "high"]
        high_risk_covered = sum(1 for p in high_risk_patterns 
                            if f"{p['type']}_{p['location']}" in self.covered_logic_patterns)
        if high_risk_patterns:
            logger.info(f"High-risk pattern coverage: {high_risk_covered}/{len(high_risk_patterns)} " +
                    f"({high_risk_covered/len(high_risk_patterns)*100:.1f}%)")

    def track_branch_condition_coverage(self):
        """Track which branch conditions from the logic model are covered by tests"""
        if not self.logic_model or not hasattr(self.logic_model, 'boundary_conditions'):
            logger.warning("No logic model or boundary conditions available for branch coverage tracking")
            return
        
        # 初始化已覆盖分支条件（如果尚未完成）
        if not hasattr(self, 'covered_branch_conditions') or self.covered_branch_conditions is None:
            self.covered_branch_conditions = set()
        
        # 获取边界条件数量
        boundary_conditions = getattr(self.logic_model, 'boundary_conditions', [])
        logger.info(f"Tracking branch condition coverage, found {len(boundary_conditions)} conditions")
        
        # 对于逻辑模型中的每个边界条件，检查测试是否可能覆盖它
        conditions_covered_this_run = 0
        for condition in boundary_conditions:
            condition_id = f"{condition['method']}_{condition['line']}"
            condition_line = condition.get("line", 0)
            condition_type = condition.get("type", "unknown")
            condition_method = condition.get("method", "unknown")
            
            # 如果条件已经被标记为覆盖，跳过重复分析
            if condition_id in self.covered_branch_conditions:
                continue
            
            # 初始化覆盖标志
            condition_covered = False
            
            # 检查测试方法名称和内容
            for test_method in self.test_methods:
                method_name = test_method.get("name", "").lower()
                test_content = test_method.get("code", "").lower()
                
                # 如果测试名称或内容表明它正在测试相关方法或条件
                target_method = condition_method.lower()
                if (
                    target_method in method_name or 
                    f"line {condition_line}" in test_content or
                    (condition_type == "if_condition" and "condition" in method_name) or
                    (condition_type in ["while_loop", "for_loop"] and "loop" in method_name)
                ):
                    condition_covered = True
                    logger.debug(f"Condition {condition_id} covered by test method {method_name}")
                    break
            
            # 根据测试具有的正确特性为条件检查
            if not condition_covered:
                if condition_type == "if_condition" and self.has_boolean_logic_tests:
                    condition_covered = True
                    logger.debug(f"Condition {condition_id} (if) covered by boolean logic tests")
                elif (condition_type == "while_loop" or condition_type == "for_loop") and self.has_boundary_tests:
                    condition_covered = True
                    logger.debug(f"Condition {condition_id} (loop) covered by boundary tests")
            
            # 如果找到了逻辑错误，认为相关的分支条件被覆盖
            # 但要限制这种假设，只覆盖与逻辑错误类型相关的条件
            if not condition_covered and self.logical_bugs:
                for bug in self.logical_bugs:
                    bug_type = bug.get("logic_bug_type", "unknown")
                    if (condition_type == "if_condition" and bug_type in ["boolean_logic", "operator_logic"]) or \
                    (condition_type in ["while_loop", "for_loop"] and bug_type in ["boundary_error", "index_error", "infinite_loop"]):
                        condition_covered = True
                        logger.debug(f"Condition {condition_id} assumed covered due to related logical bug {bug_type}")
                        break
            
            # 如果条件被覆盖，添加到已覆盖分支条件
            if condition_covered:
                self.covered_branch_conditions.add(condition_id)
                conditions_covered_this_run += 1
        
        # 如果我们仍然没有覆盖条件，但我们有好的测试和断言，
        # 至少假设一些基本条件被覆盖以避免零值
        # 但只有在测试质量好的情况下才这样做
        if len(self.covered_branch_conditions) == 0 and self.has_assertions and len(self.test_methods) > 2:
            min_covered = min(len(boundary_conditions), 2)  # 至少覆盖 2 个条件或全部（如果少于 2 个）
            logger.debug(f"No conditions covered but good tests found, assuming {min_covered} basic conditions are covered")
            for i in range(min_covered):
                if i < len(boundary_conditions):
                    condition = boundary_conditions[i]
                    condition_id = f"{condition['method']}_{condition['line']}"
                    self.covered_branch_conditions.add(condition_id)
        
        logger.info(f"After tracking, covered {len(self.covered_branch_conditions)} out of {len(boundary_conditions)} branch conditions")


    def calculate_risk_metrics(self):
        """Calculate risk metrics related to logic coverage"""
        # Calculate high-risk pattern coverage
        if self.logic_patterns:
            high_risk_patterns = [p for p in self.logic_patterns if p.get("risk_level") == "high"]
            if high_risk_patterns:
                covered_high_risk = 0
                for pattern in high_risk_patterns:
                    pattern_id = f"{pattern['type']}_{pattern['location']}"
                    if pattern_id in self.covered_logic_patterns:
                        covered_high_risk += 1
                self.high_risk_pattern_coverage = (covered_high_risk / len(high_risk_patterns)) * 100
        
        # Calculate method complexity coverage
        if self.logic_model and hasattr(self.logic_model, 'get_high_complexity_methods'):
            complex_methods = self.logic_model.get_high_complexity_methods(threshold=8)
            if complex_methods:
                # Check how many complex methods have good coverage
                covered_complex_methods = 0
                for method in complex_methods:
                    method_name = method.get("name", "")
                    # Look for methods with branch conditions covered
                    method_conditions = [cond_id for cond_id in self.covered_branch_conditions 
                                       if cond_id.startswith(f"{method_name}_")]
                    if method_conditions:
                        covered_complex_methods += 1
                
                if complex_methods:
                    self.method_complexity_coverage = (covered_complex_methods / len(complex_methods)) * 100
    
    def count_logical_bugs(self):
        """Count the number of logical bugs detected"""
        return len(self.logical_bugs)
    
    def get_logical_bug_finding_methods(self):
        """Return test methods that find logical bugs"""
        logical_bug_methods = []
        
        for bug in self.logical_bugs:
            method_name = bug.get("test_method")
            if method_name:
                method_code = self.extract_test_method_by_name(method_name)
                if method_code:
                    logical_bug_methods.append({
                        "code": method_code,
                        "triggers_bug": True,
                        "bug_type": bug.get("logic_bug_type", "unknown"),
                        "severity": bug.get("severity", "medium"),
                        "confidence": bug.get("logic_confidence", 0.5),
                        "verified": bug.get("verified", False),
                        "is_real_bug": bug.get("is_real_bug", None),
                        "bug_category": "logical",
                        "method_name": method_name
                    })
        
        return logical_bug_methods
    
    def get_complex_methods_coverage(self):
        """Get coverage information for complex methods"""
        if not self.logic_model or not hasattr(self.logic_model, 'method_complexity'):
            return {}
            
        complex_methods_coverage = {}
        
        for method_name, complexity in self.logic_model.method_complexity.items():
            # Only consider methods with high complexity
            if complexity.get("cyclomatic", 0) > 5:
                # Count related branch conditions
                covered_branches = len([cond_id for cond_id in self.covered_branch_conditions 
                                      if cond_id.startswith(f"{method_name}_")])
                
                # Count total branches for this method
                total_branches = len([cond for cond in self.logic_model.boundary_conditions
                                   if cond.get("method") == method_name])
                
                if total_branches > 0:
                    branch_coverage = (covered_branches / total_branches) * 100
                else:
                    branch_coverage = 0
                    
                complex_methods_coverage[method_name] = {
                    "complexity": complexity.get("cyclomatic", 0),
                    "covered_branches": covered_branches,
                    "total_branches": total_branches,
                    "branch_coverage": branch_coverage
                }
        
        return complex_methods_coverage
