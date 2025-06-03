#!/usr/bin/env python3
"""
LAMBDA Framework: Logic-Aware Monte carlo Bug Detection Architecture

This module serves as the main entry point for the LAMBDA framework, which enhances
MCTS-based test generation with logic-aware capabilities to improve detection of
logical bugs in Java code.

The framework integrates static analysis, logic model extraction, and enhanced MCTS
to generate tests that specifically target logical vulnerabilities.
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from collections import defaultdict

# Import core components
from model_extractor import LogicModelExtractor
from failure_scenario import LogicBugPatternDetector
from fa_mcts import LogicAwareMCTS
from bug_verifier import LogicBugVerifier
from test_generation_strategies import LogicTestStrategySelector
from test_state import LogicAwareTestState

# Import from enhanced_mcts_test_generator for compatibility
from enhanced_mcts_test_generator import TestValidator, TestMethodExtractor
from verify_bug_with_llm import merge_verified_bug_tests
from feedback import (
    generate_initial_test, save_test_code, generate_test_summary,
    read_source_code, find_source_code, strip_java_comments,
    run_tests_with_jacoco, get_coverage_percentage, 
    check_pom_for_jacoco, add_jacoco_to_pom,
    read_test_prompt_file, reset_llm_metrics, get_llm_metrics_summary,
    call_anthropic_api, call_gpt_api, extract_java_code
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("lambda_framework.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lambda_framework")

class LAMBDAFramework:
    """
    Main LAMBDA framework class that orchestrates the components for
    logic-aware test generation.
    """
    
    def __init__(self, project_dir, prompt_dir, analysis_dir=None, 
                 max_iterations=20, target_coverage=100.0,
                 verify_mode="batch", prioritize_bugs=True,
                 logic_weight=2.0, logical_bugs_threshold=15, verbose=False):
        """
        Initialize the LAMBDA framework
        
        Parameters:
        project_dir (str): Java project directory
        prompt_dir (str): Directory with test prompts
        analysis_dir (str): Directory for static analysis results (optional)
        max_iterations (int): Maximum MCTS iterations
        target_coverage (float): Target coverage percentage
        verify_mode (str): Bug verification mode (immediate/batch/none)
        prioritize_bugs (bool): Whether to prioritize bug finding over coverage
        logic_weight (float): Weight for logic-related rewards (higher = more focus on logic)
        logical_bugs_threshold (int): Number of logical bugs to find before terminating search
        verbose (bool): Enable verbose logging
        """
        self.project_dir = project_dir
        self.prompt_dir = prompt_dir
        self.analysis_dir = analysis_dir or os.path.join(project_dir, "lambda_analysis")
        self.max_iterations = max_iterations
        self.target_coverage = target_coverage
        self.verify_mode = verify_mode
        self.prioritize_bugs = prioritize_bugs
        self.logic_weight = logic_weight
        self.logical_bugs_threshold = logical_bugs_threshold
        self.verbose = verbose
        
        # Create analysis directory if it doesn't exist
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "classes_processed": 0,
            "bugs_found": 0,
            "logical_bugs_found": 0,
            "avg_coverage": 0.0,
            "execution_time": 0,
            "bug_patterns": defaultdict(int)
        }
        
        # Set logging level
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Initialized LAMBDA framework with logic_weight={logic_weight}, logical_bugs_threshold={logical_bugs_threshold}")
    
    def process_class(self, class_name, package_name):
        """
        使用逻辑感知测试生成处理单个类
        
        Parameters:
        class_name (str): 要处理的类名
        package_name (str): 包名
        
        Returns:
        tuple: (success, coverage, bug_count, logical_bug_count, test_code)
        """
        logger.info(f"处理类: {package_name}.{class_name}")
        start_time = time.time()
        
        # 1. 查找源代码
        source_file = find_source_code(self.project_dir, class_name, package_name)
        if not source_file:
            logger.error(f"未找到源代码文件: {package_name}.{class_name}")
            # 创建一个空的逻辑模型，以便可以继续尝试生成测试
            logger.warning("由于缺少源文件创建空逻辑模型")
            logic_model = LogicModelExtractor(
                source_code="", 
                class_name=class_name, 
                package_name=package_name
            )
            return False, 0.0, 0, 0, ""
        
        # 2. 读取源代码
        source_code = read_source_code(source_file)
        if not source_code or not source_code.strip():
            logger.error(f"无法读取源代码或文件为空: {package_name}.{class_name}")
            # 创建一个空的逻辑模型
            logger.warning("由于源代码为空创建空逻辑模型")
            logic_model = LogicModelExtractor(
                source_code="", 
                class_name=class_name, 
                package_name=package_name
            )
            return False, 0.0, 0, 0, ""
        
        # 3. 提取逻辑模型
        logger.info("从源代码提取逻辑模型")
        try:
            if source_code is None or not source_code.strip():
                logger.warning("源代码为空或为None，创建空逻辑模型")
                logic_model = LogicModelExtractor(
                    source_code="", 
                    class_name=class_name, 
                    package_name=package_name
                )
            else:
                logic_model = LogicModelExtractor(
                    source_code=source_code, 
                    class_name=class_name, 
                    package_name=package_name
                )
                
            if not logic_model.boundary_conditions and not logic_model.logical_operations:
                logger.warning("逻辑模型提取产生空模型。这可能影响分析质量。")
        except Exception as e:
            logger.error(f"创建逻辑模型时出错: {str(e)}")
            # 创建空逻辑模型以避免None引用
            logic_model = LogicModelExtractor(
                source_code="", 
                class_name=class_name, 
                package_name=package_name
            )
        
        # 4. 检测逻辑bug模式
        logger.info("检测逻辑bug模式")
        try:
            pattern_detector = LogicBugPatternDetector(
                source_code=source_code, 
                class_name=class_name, 
                package_name=package_name, 
                logic_model=logic_model
            )
            logic_patterns = pattern_detector.detect_patterns()
            
            # 记录检测到的模式
            if logic_patterns:
                logger.info(f"detected {len(logic_patterns)} potential logical bug patterns:")
                for pattern in logic_patterns:
                    logger.info(f"  - {pattern['type']} (risk: {pattern['risk_level']}) in line {pattern['location']}")
                    self.metrics["bug_patterns"][pattern['type']] += 1
            
            if not logic_patterns:
                logger.warning("No logical patterns detected. This may indicate simple code or limited detection capabilities.")
                logic_patterns = []  # Ensure empty list rather than None
                
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            logger.error(traceback.format_exc())
            # Create empty pattern list to continue execution
            logic_patterns = []
        
        # 5. Read test prompt
        logger.info("Read test prompt")
        test_prompt_file = os.path.join(self.prompt_dir, f"{class_name}_test_prompt.txt")
        if not os.path.exists(test_prompt_file):
            test_prompt_file = os.path.join(self.prompt_dir, f"{class_name}.txt")
            if not os.path.exists(test_prompt_file):
                logger.error(f"Test prompt file not found: {class_name}_test_prompt.txt or {class_name}.txt")
                return False, 0.0, 0, 0, ""
        
        # Read test prompt content
        test_prompt_content = read_test_prompt_file(self.prompt_dir, class_name)
        if not test_prompt_content:
            logger.error(f"Failed to read test prompt from {test_prompt_file}")
            return False, 0.0, 0, 0, ""
        
        # 5. Generate initial test
        logger.info("Generate initial test")
        initial_test = generate_initial_test(test_prompt_file, source_code)
        
        if not initial_test:
            logger.error("Initial test generation failed")
            return False, 0.0, 0, 0, ""
        
        # 6. Create strategy selector based on detected patterns
        strategy_selector = LogicTestStrategySelector(logic_patterns, logic_model)
        
        # 7. Run logic-aware MCTS for test generation
        logger.info("Start executing logic-aware MCTS for test generation")
        mcts = LogicAwareMCTS(
            project_dir=self.project_dir,
            prompt_dir=self.prompt_dir,
            class_name=class_name,
            package_name=package_name,
            initial_test_code=initial_test,
            source_code=source_code,
            test_prompt=test_prompt_content,
            logic_model=logic_model,
            logic_patterns=logic_patterns,
            strategy_selector=strategy_selector,
            max_iterations=self.max_iterations,
            exploration_weight=1.0,
            verify_bugs_mode=self.verify_mode,
            focus_on_bugs=self.prioritize_bugs,
            logic_weight=self.logic_weight,
            logical_bugs_threshold=self.logical_bugs_threshold
        )
        
        # Run MCTS search
        final_test, coverage = mcts.run_search()
        
        # Get verified bug list
        verified_bugs = mcts.verified_bug_methods
        
        # Calculate logical bugs
        real_logical_bugs = [bug for bug in verified_bugs if 
                            bug.get("is_real_bug", False) and 
                            (bug.get("bug_category", "") == "logical" or 
                            bug.get("bug_type", "").startswith("logical_"))]
        
        # Update metrics
        self.metrics["classes_processed"] += 1
        self.metrics["bugs_found"] += len(verified_bugs)
        self.metrics["logical_bugs_found"] += len(real_logical_bugs)
        self.metrics["avg_coverage"] = ((self.metrics["avg_coverage"] * (self.metrics["classes_processed"] - 1)) + coverage) / self.metrics["classes_processed"]
        
        # Generate and save summary
        execution_time = time.time() - start_time
        self.metrics["execution_time"] += execution_time
        
        # Create result summary for this class
        result_summary = {
            "class_name": class_name,
            "package_name": package_name,
            "coverage": coverage,
            "total_bugs": len(verified_bugs),
            "logical_bugs": len(real_logical_bugs),
            "logical_patterns_detected": len(logic_patterns),
            "execution_time": execution_time,
            "logic_patterns": [pattern['type'] for pattern in logic_patterns]
        }
        
        # Save result summary
        result_file = os.path.join(self.analysis_dir, f"{class_name}_lambda_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_summary, f, indent=2)
        logger.info(f"Result summary saved to: {result_file}")
        
        logger.info(f"{package_name}.{class_name} processed")
        # logger.info(f"Coverage: {coverage:.2f}%, Total bugs: {len(verified_bugs)}, Logical bugs: {len(real_logical_bugs)}")
        
        return True, coverage, len(verified_bugs), len(real_logical_bugs), final_test


    def batch_process(self, output_file=None):
        """
        Batch process all classes in the prompt directory
        
        Parameters:
        output_file (str): Path to save batch results
        
        Returns:
        list: Processing results
        """
        import glob
        import re
        
        # Find all test prompt files
        prompt_files = glob.glob(os.path.join(self.prompt_dir, "*_test_prompt.txt"))
        prompt_files.extend(glob.glob(os.path.join(self.prompt_dir, "*.txt")))
        
        # Filter valid prompt files
        # valid_files = [f for f in prompt_files if not any(x in f for x in 
        #               ["_improved", "_history", "_summary", "_best", "_mcts", 
        #                "_bug", "_critical", "_analysis", "_lambda"])]
        
        if not prompt_files:
            logger.error(f"No test prompt files found in {self.prompt_dir}")
            return []
        
        logger.info(f"Found {len(prompt_files)} test prompt files, starting batch processing")
        
        results = []
        logical_bug_count = 0
        total_bug_count = 0
        
        for file_path in prompt_files:
            # Extract class and package name
            class_name = os.path.basename(file_path).replace("_test_prompt.txt", "").replace(".txt", "")
            
            # Extract package from file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                package_match = re.search(r'Package:\s*([\w.]+)', content)
                package_name = package_match.group(1) if package_match else None
            
            if not package_name:
                logger.warning(f"Could not extract package name from {file_path}, skipping")
                continue
            
            try:
                # Process class
                success, coverage, bug_count, logical_count, test_code = self.process_class(
                    class_name, package_name)
                
                # Update counts
                total_bug_count += bug_count
                logical_bug_count += logical_count
                
                # Record result
                result = {
                    "class_name": class_name,
                    "package_name": package_name,
                    "coverage": coverage,
                    "bug_count": bug_count,
                    "logical_bug_count": logical_count,
                    "success": success,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                
                # Save intermediate results
                if output_file:
                    try:
                        # Check if output_file is a directory
                        intermediate_output = output_file
                        if os.path.isdir(output_file):
                            intermediate_output = os.path.join(output_file, "lambda_batch_results.json")
                        
                        with open(intermediate_output, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save intermediate results: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error processing {class_name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                results.append({
                    "class_name": class_name,
                    "package_name": package_name,
                    "error": str(e),
                    "success": False,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Generate final summary
        logger.info("Batch processing completed:")
        logger.info(f"Total classes processed: {len(results)}")
        logger.info(f"Total bugs found: {total_bug_count}")
        logger.info(f"Logical bugs found: {logical_bug_count}")
        logger.info(f"Logic bug patterns distribution: {dict(self.metrics['bug_patterns'])}")
        
        # Save final metrics
        metrics_file = os.path.join(self.analysis_dir, "lambda_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Final metrics saved to: {metrics_file}")
        
        # Save final results
        if output_file:
            # Check if output_file is a directory
            if os.path.isdir(output_file):
                output_file = os.path.join(output_file, "lambda_batch_results.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Final results saved to: {output_file}")
        
        return results

def main():
    """
    Command line entry point for the LAMBDA framework
    """
    parser = argparse.ArgumentParser(description="LAMBDA: Logic-Aware Monte carlo Bug Detection Architecture")
    parser.add_argument("--project", required=True, help="Java project root directory")
    parser.add_argument("--prompt", required=True, help="Directory containing test prompts")
    parser.add_argument("--analysis", help="Directory for static analysis results")
    parser.add_argument("--class", dest="class_name", help="Specific class name to test")
    parser.add_argument("--package", help="Package name of the class")
    parser.add_argument("--output", help="Output result file path")
    parser.add_argument("--batch", action="store_true", help="Batch process all classes")
    parser.add_argument("--max-iterations", type=int, default=30, help="Maximum MCTS iterations")
    parser.add_argument("--target-coverage", type=float, default=100.0, help="Target coverage percentage")
    parser.add_argument("--verify-mode", choices=["immediate", "batch", "none"], default="batch",
                        help="When to verify bugs: during MCTS (immediate), after (batch), or not at all (none)")
    parser.add_argument("--failure-weight", type=float, default=2.0, 
                        help="Weight for failure-related rewards (higher = more focus on failure)")
    parser.add_argument("--bugs-threshold", type=int, default=1000, 
                        help="Number of bugs to find before terminating search")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Initialize the framework
    framework = LAMBDAFramework(
        project_dir=args.project,
        prompt_dir=args.prompt,
        analysis_dir=args.analysis,
        max_iterations=args.max_iterations,
        target_coverage=args.target_coverage,
        verify_mode=args.verify_mode,
        failure_weight=args.failure_weight,
        bugs_threshold=args.bugs_threshold,
        verbose=args.verbose
    )
    
    # Check Jacoco configuration
    if check_pom_for_jacoco(args.project):
        logger.info("JaCoCo plugin is configured in the project")
    else:
        logger.warning("JaCoCo plugin not found, attempting to add it")
        add_jacoco_to_pom(args.project)
    
    # Reset LLM metrics
    reset_llm_metrics()
    
    # Process classes
    if args.batch:
        framework.batch_process(args.output)
    elif args.class_name and args.package:
        framework.process_class(args.class_name, args.package)
    else:
        parser.error("Must specify --batch or both --class and --package")
    
    # Output LLM usage metrics
    metrics = get_llm_metrics_summary()
    logger.info("LLM Usage Metrics:")
    logger.info(f"Total requests: {metrics['total_requests']}")
    logger.info(f"Avg. token size: {metrics['avg_token_size']:.1f}")
    logger.info(f"Total time: {metrics['total_time_minutes']:.2f} minutes")

if __name__ == "__main__":
    main()
