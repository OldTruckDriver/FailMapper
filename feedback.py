import os
import re
import glob
import time
import json
import argparse
import subprocess
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
import random
import anthropic
import statistics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_generator")

# API settings
API_KEY = "your key"
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "your model name") 
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

ANTHROPIC_API_KEY = "your key"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1/"
ANTHROPIC_DEFAULT_MODEL = "your model name"
# ANTHROPIC_DEFAULT_MODEL = "claude-3-7-sonnet-latest"
ANTHROPIC_DEFAULT_MAX_TOKENS = 8192
ANTHROPIC_DEFAULT_TEMPERATURE = 0.7

# DeepSeek API settings
DEEPSEEK_API_KEY = "your key"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_DEFAULT_MODEL = "your model name"
DEEPSEEK_DEFAULT_MAX_TOKENS = 8192
DEEPSEEK_DEFAULT_TEMPERATURE = 0.7

# Global metrics tracking variables
llm_metrics = {
    "request_count": 0,
    "token_sizes": [],
    "start_time": None,
    "end_time": None,
    "request_times": [],
}

def reset_llm_metrics():
    """Reset all LLM metrics to initial values."""
    global llm_metrics
    llm_metrics = {
        "request_count": 0,
        "token_sizes": [],
        "start_time": time.time(),
        "end_time": None,
        "request_times": [],
    }

def get_llm_metrics_summary():
    """Get a summary of LLM metrics from the current run."""
    global llm_metrics
    
    # Update end time
    llm_metrics["end_time"] = time.time()
    
    # Calculate statistics
    total_time = llm_metrics["end_time"] - llm_metrics["start_time"]
    token_sizes = llm_metrics["token_sizes"]
    
    summary = {
        "total_requests": llm_metrics["request_count"],
        "max_token_size": max(token_sizes) if token_sizes else 0,
        "min_token_size": min(token_sizes) if token_sizes else 0,
        "avg_token_size": statistics.mean(token_sizes) if token_sizes else 0,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "avg_request_time": statistics.mean(llm_metrics["request_times"]) if llm_metrics["request_times"] else 0
    }
    
    return summary

def _estimate_token_size(text):
    """Estimate token size based on a simple heuristic."""
    # A rough approximation: 1 token is about 4 characters for English text
    return len(text) // 4

# ChatGPT API Call
def create_session():
    """Create a session object for API requests"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def call_gpt_api(prompt, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    """
    Call the GPT API with the provided prompt.
    
    Parameters:
    prompt (str): The prompt to send to the API
    model (str): Model to use, defaults to DEFAULT_MODEL
    max_tokens (int): Maximum tokens in response, defaults to DEFAULT_MAX_TOKENS
    temperature (float): Temperature for response generation, defaults to DEFAULT_TEMPERATURE
    
    Returns:
    str: The model's response
    """
    global llm_metrics
    
    # Initialize metrics if this is the first call
    if llm_metrics["start_time"] is None:
        reset_llm_metrics()
    
    # Track number of requests
    llm_metrics["request_count"] += 1
    
    # Estimate token size of the prompt
    estimated_token_size = _estimate_token_size(prompt)
    llm_metrics["token_sizes"].append(estimated_token_size)
    
    # Track time for this request
    request_start = time.time()
    
    try:
        if not API_KEY:
            raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
        
        session = create_session()
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        logger.info(f"Calling API: model={model}, max_tokens={max_tokens}")
        
        # Add retry and delay mechanism
        retries = 3
        backoff = 5  # Initial wait time in seconds
        
        for attempt in range(retries):
            try:
                response = session.post(
                    f"{API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120  # Increase timeout, API may need more time to process
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Choices not found in API response: {response_json}")
                    return ""
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    wait_time = backoff * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API rate limit exceeded, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call error: {str(e)}")
                    continue
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        logger.error("API call retry limit exceeded")
        return ""
    except Exception as e:
        # Track request time even for exceptions
        request_time = time.time() - request_start
        llm_metrics["request_times"].append(request_time)
        
        logging.error(f"Exception in call_gpt_api: {str(e)}")
        return f"ERROR: {str(e)}"



# Claude API Call
def create_anthropic_session():
    """Create a session object for API requests"""
    session = requests.Session()
    return session

def call_anthropic_api(prompt, model=ANTHROPIC_DEFAULT_MODEL, max_tokens=ANTHROPIC_DEFAULT_MAX_TOKENS, temperature=ANTHROPIC_DEFAULT_TEMPERATURE):
    """
    Call the Anthropic API with the provided prompt.
    
    Parameters:
    prompt (str): The prompt to send to the API
    model (str): Model to use, defaults to ANTHROPIC_DEFAULT_MODEL
    max_tokens (int): Maximum tokens in response, defaults to ANTHROPIC_DEFAULT_MAX_TOKENS
    temperature (float): Temperature for response generation, defaults to ANTHROPIC_DEFAULT_TEMPERATURE
    
    Returns:
    str: The model's response
    """
    global llm_metrics
    
    # Initialize metrics if this is the first call
    if llm_metrics["start_time"] is None:
        reset_llm_metrics()
    
    # Track number of requests
    llm_metrics["request_count"] += 1
    
    # Estimate token size of the prompt
    estimated_token_size = _estimate_token_size(prompt)
    llm_metrics["token_sizes"].append(estimated_token_size)
    
    # Track time for this request
    request_start = time.time()
    
    try:
        if not API_KEY:
            raise ValueError("Anthropic API key not set. Please set API_KEY variable or environment variable.")
        
        session = create_anthropic_session()
        
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        logger.info(f"Calling API: model={model}, max_tokens={max_tokens}")
        
        retries = 3
        backoff = 5  # Initial wait time in seconds
        
        for attempt in range(retries):
            try:
                response = session.post(
                    f"{ANTHROPIC_API_BASE}messages",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                if "content" in response_json and len(response_json["content"]) > 0:
                    return response_json["content"][0]["text"]
                else:
                    logger.error(f"Content not found in API response: {response_json}")
                    return ""
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(f"API rate limit exceeded, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call error: {str(e)}")
                    break
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        logger.error("API call retry limit exceeded")
        return ""
    except Exception as e:
        # Track request time even for exceptions
        request_time = time.time() - request_start
        llm_metrics["request_times"].append(request_time)
        
        logging.error(f"Exception in call_anthropic_api: {str(e)}")
        return f"ERROR: {str(e)}"

def call_deepseek_api(prompt, model=DEEPSEEK_DEFAULT_MODEL, max_tokens=DEEPSEEK_DEFAULT_MAX_TOKENS, temperature=DEEPSEEK_DEFAULT_TEMPERATURE):
    """
    Call the DeepSeek API with the provided prompt.
    
    Parameters:
    prompt (str): The prompt to send to the API
    model (str): Model to use, defaults to DEEPSEEK_DEFAULT_MODEL
    max_tokens (int): Maximum tokens in response, defaults to DEEPSEEK_DEFAULT_MAX_TOKENS
    temperature (float): Temperature for response generation, defaults to DEEPSEEK_DEFAULT_TEMPERATURE
    
    Returns:
    str: The model's response
    """
    global llm_metrics
    
    # Initialize metrics if this is the first call
    if llm_metrics["start_time"] is None:
        reset_llm_metrics()
    
    # Track number of requests
    llm_metrics["request_count"] += 1
    
    # Estimate token size of the prompt
    estimated_token_size = _estimate_token_size(prompt)
    llm_metrics["token_sizes"].append(estimated_token_size)
    
    # Track time for this request
    request_start = time.time()
    
    try:
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key not set. Please set DEEPSEEK_API_KEY variable or environment variable.")
        
        session = create_session()
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        logger.info(f"Calling DeepSeek API: model={model}, max_tokens={max_tokens}")
        
        retries = 3
        backoff = 5  # Initial wait time in seconds
        
        for attempt in range(retries):
            try:
                response = session.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Choices not found in API response: {response_json}")
                    return ""
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(f"API rate limit exceeded, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call error: {str(e)}")
                    break
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        logger.error("API call retry limit exceeded")
        return ""
    except Exception as e:
        # Track request time even for exceptions
        request_time = time.time() - request_start
        llm_metrics["request_times"].append(request_time)
        
        logging.error(f"Exception in call_deepseek_api: {str(e)}")
        return f"ERROR: {str(e)}"

def extract_java_code(text):
    """提取更可靠的Java代码提取"""
    # first try to match a single complete Java code block (the entire class)
    class_pattern = re.compile(r'```java\s*((?:public\s+)?(?:class|interface|enum)\s+\w+[\s\S]*?)\s*```', re.DOTALL)
    class_match = class_pattern.search(text)
    
    if class_match:
        extracted_code = class_match.group(1)
        # ensure the code is complete, including the full class definition, not just a fragment
        if "class " in extracted_code and "{" in extracted_code and extracted_code.strip().endswith("}"):
            return extracted_code
    
    # if no complete class is found, collect all Java code blocks and concatenate them
    java_pattern = re.compile(r'```java\s*(.*?)\s*```', re.DOTALL)
    matches = java_pattern.findall(text)
    
    if matches:
        # check if there is a code block that contains a complete class definition
        for match in matches:
            if "class " in match and "{" in match and match.strip().endswith("}"):
                return match
                
        # if there is no complete class, but there are code blocks, use the longest one
        if len(matches) == 1:
            return matches[0]
        else:
            # if there are multiple code blocks, try to intelligently merge them
            combined_code = "\n\n".join(matches)
            # check if the merged code is a complete class
            if "class " in combined_code and "{" in combined_code and combined_code.strip().endswith("}"):
                return combined_code
            else:
                # if the merged code is not complete, return the longest code block
                return max(matches, key=len)
    
    # fallback to any code block
    code_pattern = re.compile(r'```\s*(.*?)\s*```', re.DOTALL)
    matches = code_pattern.findall(text)
    
    if matches:
        # try to find a code block that contains a complete class definition
        for match in matches:
            if "class " in match and "{" in match and match.strip().endswith("}"):
                return match
        
        # otherwise return the longest code block
        return max(matches, key=len)
    
    # final attempt: extract any content between quotes, if it looks like Java code
    if "public class" in text or "import " in text:
        # try to extract the content from the class declaration to the last brace
        class_start = text.find("public class")
        if class_start == -1:
            class_start = text.find("class ")
        
        if class_start != -1:
            # after finding the class start position, try to extract the content until the end
            open_braces = 0
            in_class = False
            class_content = []
            
            for line in text[class_start:].split('\n'):
                class_content.append(line)
                
                if '{' in line:
                    in_class = True
                    open_braces += line.count('{')
                    
                if '}' in line:
                    open_braces -= line.count('}')
                    
                if in_class and open_braces == 0:
                    break
                    
            if class_content:
                return '\n'.join(class_content)
        
        # if still cannot extract, return the entire text, it may contain Java code
        return text
        
    # if all methods fail, return the original text
    return text

def generate_initial_test(test_prompt_file, source_code):
    """
    Generate initial unit test
    
    Parameters:
    test_prompt_file (str): Path to test prompt file
    source_code (str): Source code
    
    Returns:
    str: Generated test code
    """
    try:
        with open(test_prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read prompt file: {str(e)}")
        return ""
    
    prompt = f"""
{prompt_content}



Please provide the complete test class code, including all necessary imports and annotations. Ensure that your tests are thorough, covering all aspects of the class behavior while considering the provided structure, data flow, and dependencies.

Important notes:
1. Remember to import all necessary classes as listed in the Imports section.
2. In your test class, explicitly verify that the class implements all listed interfaces and extends the superclass (if any).
3. When testing overridden methods, add comments indicating which interface or superclass they are inherited from.
4. DO NOT use @Nested annotations or nested test classes, as they cause coverage tracking issues.
5. Always provide a complete, well-structured test class that will compile without any modifications.
6. Use straightforward test methods without nesting to ensure proper coverage tracking.


Please generate a complete JUnit test class, ensuring coverage of all main functionality.
Use JUnit 5 (Jupiter) annotations and assertions. Please follow all testing requirements in the prompt.

IMPORTANT: YOUR RESPONSE MUST CONTAIN THE COMPLETE TEST CLASS CODE. DO NOT OMIT ANY PARTS OF THE CODE OR USE PLACEHOLDERS.
"""
    
    logger.info(f"Generating initial test, prompt length: {len(prompt)}")
    # response = call_gpt_api(prompt)
    response = call_anthropic_api(prompt)
    # response = call_deepseek_api(prompt)
    if not response:
        logger.error("API returned empty response")
        return ""
    
    # Extract Java code
    test_code = extract_java_code(response)
    logger.info(f"Extracted test code length: {len(test_code)}")
    
    return test_code

# Define Apache License text
APACHE_LICENSE = """/*
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
 */
"""

def save_test_code(test_code, class_name, package_name, project_dir):
    """
    Save test code to file
    
    Parameters:
    test_code (str): Test code
    class_name (str): Name of the class being tested
    package_name (str): Package name
    project_dir (str): Project directory
    
    Returns:
    str: Path to the saved file
    """
    # 添加空值检查
    if test_code is None:
        logger.error("Cannot save test code: test_code is None")
        return ""
        
    # Define possible test directory paths
    test_dirs = [
        os.path.join(project_dir, "src", "test", "java", package_name.replace(".", os.sep)),
        os.path.join(project_dir, "src", "test", "java", "test", package_name.replace(".", os.sep)),
        os.path.join(project_dir, "test", "java", package_name.replace(".", os.sep))
    ]
    
    # Choose the first existing directory, create the first one if none exist
    test_dir = next((d for d in test_dirs if os.path.exists(d)), test_dirs[0])
    
    # Ensure directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Determine test class name
    test_class_name = f"{class_name}Test"
    
    # Build complete file path
    file_path = os.path.join(test_dir, f"{test_class_name}.java")
    
    # Check if code already includes license header
    if not test_code.strip().startswith("/*"):
        # If test code doesn't contain package declaration, add it
        if not re.search(r'package\s+[\w.]+;', test_code):
            test_code = f"{APACHE_LICENSE}\npackage {package_name};\n\n{test_code}"
        else:
            # Find package declaration
            package_match = re.search(r'(package\s+[\w.]+;)', test_code)
            if package_match:
                # Insert license before package declaration
                package_stmt = package_match.group(1)
                test_code = test_code.replace(package_stmt, f"{APACHE_LICENSE}\n{package_stmt}")
            else:
                # If package declaration not found but not added, add license at the beginning
                test_code = f"{APACHE_LICENSE}\n{test_code}"
    
    # Save file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(test_code)
        logger.info(f"Test code saved to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save test code: {str(e)}")
        return ""

def run_maven_command(command, project_dir='.'):
    """
    Run Maven command and return output and error information
    
    Parameters:
    command (str): Maven command
    project_dir (str): Project directory
    
    Returns:
    tuple: (success, stdout, stderr)
    """
    full_command = f"mvn {command}"
    
    try:
        process = subprocess.Popen(
            full_command,
            shell=True,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        success = process.returncode == 0
        
        return success, stdout, stderr
    
    except Exception as e:
        logger.error(f"Failed to run Maven command: {str(e)}")
        return False, "", str(e)

def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_maven_errors(output):
    """
    Parse error information from Maven output with improved test failure detection.
    Extract each individual test failure as a separate error.
    
    Parameters:
    output (str): Maven command output
    
    Returns:
    tuple: (compilation_errors, assertion_failures) - Lists of different error types
    """
    if not output:
        return [], []
        
    # Clean ANSI codes
    output = remove_ansi_escape_sequences(output)
    
    compilation_errors = []
    assertion_failures = []
    
    # Check for OutOfMemory errors first (highest priority)
    memory_errors = re.findall(r'(java\.lang\.OutOfMemoryError:.*?)(?:\n|\r\n|\r)', output)
    if memory_errors:
        compilation_errors.append(f"Critical memory error: {memory_errors[0].strip()}")
    
    # Check for VM limit errors (another form of OOM)
    vm_limit_errors = re.findall(r'(Requested array size exceeds VM limit.*?)(?:\n|\r\n|\r)', output)
    if vm_limit_errors:
        compilation_errors.append(f"Critical memory error: {vm_limit_errors[0].strip()}")
    
    # Extract individual test failures - this is the key improvement
    # Look for lines like: [ERROR]   SoundexTest.testEmptyInput:79 expected: <0000> but was: <>
    test_failure_pattern = r'\[ERROR\]\s+([A-Za-z0-9_.]+\.[A-Za-z0-9_]+)(?::(\d+))?\s+(.*)'
    test_failures = re.findall(test_failure_pattern, output)
    
    for test_class_method, line_number, error_msg in test_failures:
        # Create a formatted error message
        if line_number:
            formatted_error = f"{test_class_method}:{line_number} {error_msg.strip()}"
        else:
            formatted_error = f"{test_class_method} {error_msg.strip()}"
        
        assertion_failures.append(formatted_error)
    
    # If no specific test failures found, look for general test failure blocks
    if not assertion_failures:
        test_failure_blocks = re.findall(r'\[ERROR\] Failures:\s*(.*?)(?=\[INFO\]|\[ERROR\]|$)', output, re.DOTALL)
        if test_failure_blocks:
            for block in test_failure_blocks:
                assertion_failures.append(f"Test failures: {block.strip()}")
    
    # Check for general compilation errors
    compile_errors = re.findall(r'\[ERROR\] (?!Failures:)(.*?\.java:\d+:.*?)(?=\[|\n\[|\Z)', output, re.DOTALL)
    for error in compile_errors:
        compilation_errors.append(error.strip())
    
    # Check for build failure
    if "BUILD FAILURE" in output and not (compilation_errors or assertion_failures):
        compilation_errors.append(output)
    
    return compilation_errors, assertion_failures


def find_jacoco_report(project_dir):
    """
    Find Jacoco-generated XML report file
    
    Parameters:
    project_dir (str): Project directory
    
    Returns:
    str: Report file path
    """
    # Common Jacoco report paths
    patterns = [
        os.path.join(project_dir, 'target', 'site', 'jacoco', 'jacoco.xml'),
        os.path.join(project_dir, 'target', 'site', 'jacoco-ut', 'jacoco.xml'),
        os.path.join(project_dir, 'target', 'site', 'jacoco-aggregate', 'jacoco.xml'),
        os.path.join(project_dir, 'build', 'reports', 'jacoco', 'test', 'jacocoTestReport.xml'),
        os.path.join(project_dir, 'target', 'jacoco.xml'),
        os.path.join(project_dir, 'target', 'jacoco', 'jacoco.xml')
    ]
    
    # Log search process
    logger.info("Searching for Jacoco report file...")
    for pattern in patterns:
        logger.debug(f"Checking path: {pattern}")
        if os.path.exists(pattern):
            logger.info(f"Found Jacoco report: {pattern}")
            return pattern
    
    # Use glob for wider search
    logger.info("Using glob to search for Jacoco report...")
    xml_files = glob.glob(os.path.join(project_dir, 'target', 'site', '**', 'jacoco*.xml'), recursive=True)
    if xml_files:
        logger.info(f"Found possible Jacoco report: {xml_files[0]}")
        return xml_files[0]
    
    # If still not found, try running specific Jacoco command
    # logger.info("Trying to directly run Jacoco to generate report...")
    # _, _, _ = run_maven_command('clean test', project_dir)
    # success, _, _ = run_maven_command('jacoco:report', project_dir)
    # if success:
    #     # Search again for the report
    #     for pattern in patterns:
    #         if os.path.exists(pattern):
    #             logger.info(f"Generated and found Jacoco report: {pattern}")
    #             return pattern
    
    logger.error("No Jacoco report file found")
    return None


def extract_jacoco_coverage(xml_path, target_class=None, target_package=None):
    """
    Extract coverage data from Jacoco XML report
    
    Parameters:
    xml_path (str): Jacoco XML report path
    target_class (str): Target class name, if specified, extract only information for this class
    target_package (str): Target package name, if specified, extract only information for this package
    
    Returns:
    dict: Coverage data
    """
    if not xml_path:
        logger.error("No Jacoco XML report path provided")
        return None
        
    if not os.path.exists(xml_path):
        logger.error(f"Report file does not exist: {xml_path}")
        return None
    
    coverage_data = {
        'summary': {},
        'uncovered_lines': [],
        'uncovered_branches': [],
        'class_coverage': defaultdict(lambda: {'lines': [], 'branches': []})
    }
    
    # 标记是否为类特定模式
    class_specific_mode = target_class is not None
    target_found = False
    
    # Normalize the target class name - remove Test suffix if present
    normalized_target_class = target_class
    if target_class and target_class.endswith("Test"):
        normalized_target_class = target_class[:-4]  # Remove "Test" suffix
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Parse overall coverage summary
        for counter in root.findall(".//counter"):
            type_attr = counter.get('type')
            covered = int(counter.get('covered', 0))
            missed = int(counter.get('missed', 0))
            total = covered + missed
            coverage = (covered / total * 100) if total > 0 else 0
            
            coverage_data['summary'][type_attr] = {
                'covered': covered,
                'missed': missed,
                'total': total,
                'coverage_percent': round(coverage, 2)
            }
        
        # Parse information for each package
        for package in root.findall(".//package"):
            package_name = package.get('name', '')
            
            # Normalize package name for comparison
            normalized_package_name = package_name.replace('/', '.')
            
            # If target package name specified, skip non-matching packages
            if target_package and normalized_package_name != target_package:
                continue
            
            # Parse classes in the package
            for class_elem in package.findall(".//class"):
                class_name = class_elem.get('name', '')
                full_class_name = class_name.replace('/', '.')
                short_class_name = class_name.split('/')[-1] if '/' in class_name else class_name
                
                # Try various ways to match the class name
                is_match = False
                
                # Check if class names match exactly
                if normalized_target_class and (short_class_name == normalized_target_class or 
                                            short_class_name == target_class or 
                                            class_name.endswith(f"/{normalized_target_class}") or
                                            class_name.endswith(f"/{target_class}")):
                    is_match = True
                    logger.info(f"Found exact class match: {full_class_name}")
                
                # If not an exact match, check for partial match
                if normalized_target_class and not is_match:
                    if normalized_target_class in short_class_name or normalized_target_class in full_class_name:
                        is_match = True
                        logger.info(f"Found partial class match: {full_class_name}")
                
                # If target class specified and no match, skip
                if normalized_target_class and not is_match:
                    continue
                
                # Record that we found the target class
                if normalized_target_class and is_match:
                    target_found = True
                    
                    # Create separate summary information for the target class
                    class_summary = {}
                    for counter in class_elem.findall(".//counter"):
                        type_attr = counter.get('type')
                        covered = int(counter.get('covered', 0))
                        missed = int(counter.get('missed', 0))
                        total = covered + missed
                        coverage = (covered / total * 100) if total > 0 else 0
                        
                        class_summary[type_attr] = {
                            'covered': covered,
                            'missed': missed,
                            'total': total,
                            'coverage_percent': round(coverage, 2)
                        }
                    
                    coverage_data['class_summary'] = class_summary
                
                # Look for source file corresponding to the class
                sourcefile = None
                for sf in package.findall(".//sourcefile"):
                    # Try to match source file to class name
                    class_file_name = short_class_name + ".java"
                    normalized_class_file = short_class_name
                    if normalized_class_file.endswith("Test"):
                        normalized_class_file = normalized_class_file[:-4] + ".java"
                    else:
                        normalized_class_file = normalized_class_file + ".java"
                    
                    if (sf.get('name') == class_file_name or 
                        sf.get('name') == normalized_class_file or
                        normalized_target_class and sf.get('name') == normalized_target_class + ".java"):
                        sourcefile = sf
                        break
                
                if not sourcefile:
                    logger.debug(f"No matching source file found for class: {short_class_name}")
                    continue
                
                # Parse line information in the source file
                for line in sourcefile.findall(".//line"):
                    line_number = int(line.get('nr', 0))
                    line_covered = True
                    branch_fully_covered = True
                    
                    # Check line coverage
                    mi = int(line.get('mi', 0))  # Missed instructions
                    ci = int(line.get('ci', 0))  # Covered instructions
                    
                    if mi > 0:
                        line_covered = False
                    
                    # Check branch coverage
                    mb = int(line.get('mb', 0) if 'mb' in line.attrib else 0)  # Missed branches
                    cb = int(line.get('cb', 0) if 'cb' in line.attrib else 0)  # Covered branches
                    
                    if mb > 0:
                        branch_fully_covered = False
                    
                    full_class_name = class_name.replace('/', '.')
                    
                    # Record uncovered lines
                    if not line_covered:
                        line_info = {
                            'package': package_name.replace('/', '.'),
                            'class': full_class_name,
                            'source_file': sourcefile.get('name'),
                            'line': line_number,
                            'ci': ci,
                            'mi': mi,
                            'coverage': f"{ci}/{ci+mi}"
                        }
                        coverage_data['uncovered_lines'].append(line_info)
                        coverage_data['class_coverage'][full_class_name]['lines'].append(line_info)
                    
                    # Record uncovered branches
                    if not branch_fully_covered:
                        branch_info = {
                            'package': package_name.replace('/', '.'),
                            'class': full_class_name,
                            'source_file': sourcefile.get('name'),
                            'line': line_number,
                            'cb': cb,
                            'mb': mb,
                            'coverage': f"{cb}/{cb+mb}"
                        }
                        coverage_data['uncovered_branches'].append(branch_info)
                        coverage_data['class_coverage'][full_class_name]['branches'].append(branch_info)
        
        # 如果是类特定模式但没有找到目标类，记录警告并返回整体覆盖率
        if class_specific_mode and not target_found:
            logger.warning(f"Warning: Target class {target_class} not found in Jacoco report")
            # Use overall coverage instead
            if 'summary' in coverage_data and 'INSTRUCTION' in coverage_data['summary']:
                overall_summary = {
                    'INSTRUCTION': coverage_data['summary']['INSTRUCTION'],
                    'LINE': coverage_data['summary']['LINE'] if 'LINE' in coverage_data['summary'] else None,
                    'BRANCH': coverage_data['summary']['BRANCH'] if 'BRANCH' in coverage_data['summary'] else None,
                    'METHOD': coverage_data['summary']['METHOD'] if 'METHOD' in coverage_data['summary'] else None,
                    'CLASS': coverage_data['summary']['CLASS'] if 'CLASS' in coverage_data['summary'] else None
                }
                coverage_data['class_summary'] = {k: v for k, v in overall_summary.items() if v is not None}
                logger.info(f"Using overall coverage instead: {coverage_data['class_summary'].get('INSTRUCTION', {}).get('coverage_percent', 0)}%")
        
        # 如果是类特定模式，只保留目标类的信息
        if class_specific_mode and target_found:
            # 保留找到的目标类的信息，不需要额外过滤
            pass
        
        return coverage_data
    
    except Exception as e:
        logger.error(f"Error parsing Jacoco XML report: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def get_class_uncovered_details(coverage_data, class_name, package_name=None):
    """
    Get uncovered details for a specific class
    
    Parameters:
    coverage_data (dict): Coverage data
    class_name (str): Class name
    package_name (str): Package name
    
    Returns:
    dict: Uncovered lines and branches information
    """
    if not coverage_data or 'class_coverage' not in coverage_data:
        return None
    
    full_name = f"{package_name}.{class_name}" if package_name else class_name
    class_data = None
    
    # Direct lookup for exact class name match
    if full_name in coverage_data['class_coverage']:
        class_data = coverage_data['class_coverage'][full_name]
    else:
        # If exact match not found, look for keys containing the class name
        for key, data in coverage_data['class_coverage'].items():
            class_part = key.split('.')[-1]
            if class_part == class_name:
                if package_name is None or package_name in key:
                    class_data = data
                    break
    
    if not class_data:
        return None
    
    return {
        'lines': sorted(class_data.get('lines', []), key=lambda x: x['line']),
        'branches': sorted(class_data.get('branches', []), key=lambda x: x['line'])
    }

def get_coverage_percentage(coverage_data):
    """
    Get coverage percentage from coverage data
    
    Parameters:
    coverage_data (dict): Coverage data
    
    Returns:
    float: Coverage percentage
    """
    if not coverage_data:
        return 0.0
    
    # 首先尝试从class_summary获取覆盖率
    if 'class_summary' in coverage_data:
        # 优先使用指令覆盖率，其次使用行覆盖率，最后使用分支覆盖率
        if 'INSTRUCTION' in coverage_data['class_summary']:
            return coverage_data['class_summary']['INSTRUCTION']['coverage_percent']
        elif 'LINE' in coverage_data['class_summary']:
            return coverage_data['class_summary']['LINE']['coverage_percent']
        elif 'BRANCH' in coverage_data['class_summary']:
            return coverage_data['class_summary']['BRANCH']['coverage_percent']
    
    # 如果没有类级别摘要，尝试从整体摘要获取覆盖率
    if 'summary' in coverage_data:
        if 'INSTRUCTION' in coverage_data['summary']:
            return coverage_data['summary']['INSTRUCTION']['coverage_percent']
        elif 'LINE' in coverage_data['summary']:
            return coverage_data['summary']['LINE']['coverage_percent']
        elif 'BRANCH' in coverage_data['summary']:
            return coverage_data['summary']['BRANCH']['coverage_percent']
    
    # 如果找不到任何覆盖率数据，记录警告
    logger.warning("No coverage percentage data found in coverage_data")
    return 0.0

def read_test_prompt_file(prompt_dir, class_name):
    """
    Read test prompt file content
    
    Parameters:
    prompt_dir (str): Prompt file directory
    class_name (str): Class name
    
    Returns:
    str: Prompt file content, empty string if not found
    """
    possible_files = [
        os.path.join(prompt_dir, f"{class_name}_test_prompt.txt"),
        os.path.join(prompt_dir, f"{class_name}.txt")
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read prompt file: {str(e)}")
                break
    
    return ""

def format_feedback_for_prompt(coverage_data, errors, class_name, package_name, prompt_content, maven_output=""):
    """
    Generate comprehensive feedback information for prompts
    
    Parameters:
    coverage_data (dict): Coverage data
    errors (list): Error list
    class_name (str): Class name
    package_name (str): Package name
    prompt_content (str): Original prompt content
    maven_output (str): Original Maven output (for showing coverage tool errors)
    
    Returns:
    str: Formatted feedback information
    """
    feedback = []
    
    # Check for assertion failures
    assertion_failures = [err for err in errors if "assertion failed" in err.lower() or "expected:" in err.lower() or "AssertionFailedError" in err]
    has_assertion_failures = len(assertion_failures) > 0
    
    # 1. Add test results
    if errors:
        if has_assertion_failures:
            feedback.append("Assertion failures found in tests, please fix these issues first:")
            for i, error in enumerate(assertion_failures, 1):
                feedback.append(f"{i}. {error}")
            
            # Add other non-assertion errors
            other_errors = [err for err in errors if err not in assertion_failures]
            if other_errors:
                feedback.append("\nOther test errors:")
                for i, error in enumerate(other_errors, 1):
                    feedback.append(f"{i}. {error}")
        else:
            feedback.append("Tests failed, found the following errors:")
            for i, error in enumerate(errors, 1):
                feedback.append(f"{i}. {error}")
    else:
        feedback.append("All tests passed, no errors.")
    
    # 2. Add coverage information
    if coverage_data:
        # 添加覆盖率摘要，明确指出是针对特定类的
        feedback.append(f"\nCoverage Summary for {package_name}.{class_name}:")
        
        # 如果有类特定的摘要，优先使用它
        if 'class_summary' in coverage_data:
            summary = coverage_data.get('class_summary', {})
            feedback.append("Class-specific coverage metrics:")
        else:
            summary = coverage_data.get('summary', {})
            feedback.append("Note: Using overall coverage metrics (class-specific data not found)")
        
        # Check if all types of coverage reach 100%
        all_covered = True
        for type_name, data in summary.items():
            coverage_percent = data['coverage_percent']
            feedback.append(f"- {type_name}: {data['covered']}/{data['total']} ({coverage_percent}%)")
            if coverage_percent < 100.0:
                all_covered = False
        
        # Get uncovered details for specific class
        details = get_class_uncovered_details(coverage_data, class_name, package_name)
        
        if not details:
            # When details for specific class not found
            if all_covered:
                # If summary shows 100% coverage, provide positive feedback
                feedback.append(f"\nCongratulations! {package_name}.{class_name} has reached 100% code coverage.\nAll code lines and branches are covered by tests.")
            else:
                # If summary doesn't show 100% coverage, but details not found
                feedback.append(f"\nCould not find detailed coverage information for {package_name}.{class_name}. Please refer to the coverage summary above for areas not fully covered.")
        else:
            # Format uncovered lines information
            uncovered_lines = details.get('lines', [])
            if uncovered_lines:
                feedback.append(f"\nUncovered lines in {class_name} (total: {len(uncovered_lines)}):")
                for line in uncovered_lines[:20]:  # Show at most 20 lines
                    feedback.append(f"- Line {line['line']}: Coverage {line['coverage']}")
                if len(uncovered_lines) > 20:
                    feedback.append(f"... etc., total {len(uncovered_lines)} uncovered lines")
            else:
                feedback.append(f"\nAll lines in {class_name} are covered.")
            
            # Format uncovered branches information
            uncovered_branches = details.get('branches', [])
            if uncovered_branches:
                feedback.append(f"\nUncovered branches in {class_name} (total: {len(uncovered_branches)}):")
                for branch in uncovered_branches[:20]:  # Show at most 20 branches
                    feedback.append(f"- Line {branch['line']}: Branch coverage {branch['coverage']}")
                if len(uncovered_branches) > 20:
                    feedback.append(f"... etc., total {len(uncovered_branches)} uncovered branches")
            else:
                feedback.append(f"\nAll branches in {class_name} are covered.")
    else:
        # If no coverage data
        feedback.append(f"\nCould not get coverage information for {package_name}.{class_name}. Please ensure the project is properly configured with Jacoco plugin and can generate coverage reports.")
        
        # Add original Maven error information, if available
        if maven_output:
            # Extract Jacoco-related errors
            jacoco_errors = re.findall(r'(?:ERROR|WARNING).*?jacoco.*?(?:\n|$)', maven_output, re.IGNORECASE)
            if jacoco_errors:
                feedback.append("\nJacoco-related errors:")
                for error in jacoco_errors:
                    feedback.append(f"- {error.strip()}")
            
            # Extract plugin execution errors
            plugin_errors = re.findall(r'(?:ERROR).*?plugin.*?(?:\n|$)', maven_output, re.IGNORECASE)
            if plugin_errors:
                feedback.append("\nMaven plugin execution errors:")
                for error in plugin_errors:
                    feedback.append(f"- {error.strip()}")
    
    # 3. Add test improvement suggestions, emphasizing assertion failures
    feedback.append("\nTest Improvement Suggestions:")
    
    if has_assertion_failures:
        feedback.append("Priority actions:")
        feedback.append("1. Carefully check test cases with assertion failures, understand the difference between expected and actual values")
        feedback.append("2. Confirm that test data is correct, or if the expected behavior of the tested method has changed")
        feedback.append("3. Fix assertions to match the current implementation's correct behavior")
        feedback.append("4. Don't blindly modify assertions to pass tests, ensure changes reflect correct expected behavior")
        feedback.append("\nOther improvement areas:")
    
    feedback.append(f"1. Ensure tests cover all possible code paths in {class_name}, including boundary conditions and exception scenarios")
    feedback.append("2. Test methods with different parameter combinations")
    feedback.append("3. Add test cases for specific functionality to ensure correctness")
    feedback.append("4. Consider using parameterized tests for similar but different input scenarios")
    feedback.append("5. Write dedicated test cases for uncovered branches")
    
    return "\n".join(feedback)


def run_tests_with_jacoco(project_dir, class_name=None, package_name=None, test_class=None, skip_coverage=False):
    """
    Run Maven tests and get Jacoco coverage report and test error information
    with improved error parsing
    
    Parameters:
    project_dir (str): Project directory
    class_name (str): Class name to analyze
    package_name (str): Package name of the class
    test_class (str): Specified test class to run
    skip_coverage (bool): Skip coverage generation for faster test execution
    
    Returns:
    tuple: (coverage_data, assertion_failures, execution_time, compilation_errors)
    """
    # Build Maven command
    maven_command = 'clean test'
    
    # Only add jacoco:report if we're not skipping coverage
    if not skip_coverage:
        maven_command += ' jacoco:report'
    
    # If specified test class, only run that test class
    if test_class:
        maven_command = f'clean test -Dtest={test_class}'
        # if not skip_coverage:
        #     maven_command += ' jacoco:report'
    
    # Run Maven tests and generate Jacoco report
    logger.info(f"Running tests{' and generating Jacoco report' if not skip_coverage else ''}...")
    success, stdout, stderr = run_maven_command(maven_command, project_dir)
    run_maven_command('jacoco:report', project_dir)
    
    
    # Combine standard output and error output
    combined_output = stdout + "\n" + stderr
    
    # Parse Maven errors with improved function
    compilation_errors, assertion_failures = parse_maven_errors(combined_output)
    
    # Log errors appropriately
    if compilation_errors:
        logger.warning("Test run has compilation errors:")
        for i, error in enumerate(compilation_errors[:5], 1):
            logger.warning(f"{i}. {error[:200]}...")
        if len(compilation_errors) > 5:
            logger.warning(f"...and {len(compilation_errors) - 5} more compilation errors")
            
    if assertion_failures:
        logger.warning("Test run has assertion failures:")
        for i, failure in enumerate(assertion_failures[:5], 1):
            logger.warning(f"{i}. {failure[:200]}...")
        if len(assertion_failures) > 5:
            logger.warning(f"...and {len(assertion_failures) - 5} more assertion failures")
    
    xml_report = find_jacoco_report(project_dir)
    
    if not xml_report:
        logger.error("No Jacoco XML report found, please confirm that the project has the Jacoco plugin configured")
        # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
        return None, assertion_failures, 0.0, compilation_errors

    # If we're skipping coverage, return early
    if skip_coverage:
        return None, assertion_failures, 0.0, compilation_errors
    
    # Find Jacoco report
    # xml_report = find_jacoco_report(project_dir)
    # if not xml_report:
    #     # Try to force generate Jacoco report
    #     logger.warning("Trying to force generate Jacoco report...")
    #     run_maven_command('jacoco:report', project_dir)
    #     xml_report = find_jacoco_report(project_dir)
        
    #     if not xml_report:
    #         logger.error("No Jacoco XML report found, please confirm that the project has the Jacoco plugin configured")
    #         # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
    #         return None, assertion_failures, 0.0, compilation_errors
    #     # logger.error("No Jacoco XML report found, please confirm that the project has the Jacoco plugin configured")
    #     # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
    #     return None, assertion_failures, 0.0, compilation_errors
    
    # logger.info(f"Found Jacoco XML report: {xml_report}")
    
    # Normalize the class name (remove Test suffix if present)
    target_class = class_name
    if target_class and target_class.endswith("Test"):
        target_class = target_class[:-4]  # Remove "Test" suffix
        logger.info(f"Normalizing class name from {class_name} to {target_class} for coverage analysis")
    
    # Set timeout handling
    try:
        import threading
        
        result = [None]
        error_flag = [False]
        
        def extract_with_timeout():
            try:
                logger.info("Starting to parse Jacoco report...")
                result[0] = extract_jacoco_coverage(xml_report, target_class, package_name)
                if result[0] and 'class_summary' in result[0]:
                    coverage_pct = result[0]['class_summary'].get('INSTRUCTION', {}).get('coverage_percent', 0.0)
                    logger.info(f"Found coverage data: {coverage_pct:.2f}% for {package_name}.{target_class}")
                else:
                    logger.warning(f"No class-specific coverage found for {package_name}.{target_class}")
                logger.info("Finished parsing Jacoco report successfully")
            except Exception as e:
                error_flag[0] = True
                logger.error(f"Error processing Jacoco report: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Create and start thread
        extract_thread = threading.Thread(target=extract_with_timeout)
        extract_thread.daemon = True
        extract_thread.start()
        
        # Wait for thread to complete with timeout
        extract_thread.join(timeout=10)
        
        if extract_thread.is_alive():
            logger.error("Jacoco report parsing timed out after 10 seconds, possibly stuck in infinite loop")
            # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
            return None, assertion_failures, 0.0, compilation_errors
        
        if error_flag[0]:
            logger.error("Exception occurred during Jacoco report parsing")
            # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
            return None, assertion_failures, 0.0, compilation_errors
        
        coverage_data = result[0]
        # coverage_data = result[0]['class_summary'].get('INSTRUCTION', {}).get('coverage_percent', 0.0)
        # Log coverage summary
        if coverage_data and 'class_summary' in coverage_data:
            for type_name, data in coverage_data['class_summary'].items():
                logger.info(f"Coverage metric {type_name}: {data['coverage_percent']}% ({data['covered']}/{data['total']})")
                
        # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
        return coverage_data, assertion_failures, 0.0, compilation_errors
        
    except Exception as e:
        logger.error(f"Unexpected error in run_tests_with_jacoco: {str(e)}")
        logger.error(traceback.format_exc())
        # Return tuples that match expected return values: (coverage_data, assertion_failures, execution_time, compilation_errors)
        return None, assertion_failures, 0.0, compilation_errors

def find_source_code(project_dir, class_name, package_name):
    """
    Find source code file
    
    Parameters:
    project_dir (str): Project directory
    class_name (str): Class name
    package_name (str): Package name
    
    Returns:
    str: Source code file path, or None if not found
    """
    if not project_dir or not class_name:
        logger.error("Project directory or class name not provided")
        return None
        
    if package_name:
        package_path = package_name.replace('.', os.sep)
        
        # Common locations to look for Java source files
        potential_paths = [
            os.path.join(project_dir, 'src', 'main', 'java', package_path, f"{class_name}.java"),
            os.path.join(project_dir, 'src', 'java', package_path, f"{class_name}.java"),
            os.path.join(project_dir, 'src', package_path, f"{class_name}.java"),
            os.path.join(project_dir, 'java', package_path, f"{class_name}.java")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Found source file at: {path}")
                return path
                
    # If not found, try using glob
    try:
        possible_files = glob.glob(os.path.join(project_dir, 'src', '**', f"{class_name}.java"), recursive=True)
        if not possible_files:
            possible_files = glob.glob(os.path.join(project_dir, '**', f"{class_name}.java"), recursive=True)
            
        if possible_files:
            logger.info(f"Found source file via glob: {possible_files[0]}")
            return possible_files[0]
    except Exception as e:
        logger.error(f"Error searching for source file: {str(e)}")
    
    logger.error(f"Could not find source file for {package_name}.{class_name}")
    return None

def read_source_code(file_path):
    """
    Read source code file
    
    Parameters:
    file_path (str): File path
    
    Returns:
    str: Source code content, empty string if file doesn't exist or is empty
    """
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Source code file does not exist: {file_path}")
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content or not content.strip():
                logger.warning(f"Source code file is empty: {file_path}")
            return content or ""  # 确保返回空字符串而不是None
    except Exception as e:
        logger.error(f"Failed to read source code file: {str(e)}")
        return ""

def strip_java_comments(source_code):
    """
    Remove comments and license headers from Java source code
    
    Args:
        source_code (str): Original Java source code
        
    Returns:
        str: Java source code with comments removed
    """
    import re
    
    # Remove block comments (including license headers)
    # This handles both /* */ and /** */ style comments
    source_without_block_comments = re.sub(r'/\*[\s\S]*?\*/', '', source_code)
    
    # Remove single line comments
    source_without_comments = re.sub(r'//.*$', '', source_without_block_comments, flags=re.MULTILINE)
    
    # Remove empty lines that may have been created by comment removal
    cleaned_source = re.sub(r'\n\s*\n+', '\n\n', source_without_comments)
    
    return cleaned_source.strip()


def apply_rule_based_repairs(test_code, errors, class_name, package_name, project_dir):
    """
    Apply rule-based repair strategies to fix errors in the test code,
    attempting to resolve common issues before invoking a large model.
    
    Parameters:
    test_code (str): The test code
    errors (list): List of error messages
    class_name (str): The class name
    package_name (str): The package name
    project_dir (str): The project directory
    
    Returns:
    tuple: (repaired code, whether it was fixed, remaining errors)
    """
    logger.info("Applying rule-based repair strategies...")
    
    if not test_code or not errors:
        return test_code, False, errors
    
    original_test_code = test_code
    fixed = False
    remaining_errors = errors.copy()
    
    # Rule 1: Fix truncated test code (add missing braces, etc.)
    if any("Parse error" in err for err in errors) or any("Found <EOF>" in err for err in errors):
        logger.info("Attempting to fix code truncation issues")
        # Count the number of braces to detect missing closing braces
        open_braces = test_code.count('{')
        close_braces = test_code.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            test_code = test_code + "}" * (open_braces - close_braces)
            fixed = True
            logger.info(f"Added {open_braces - close_braces} missing closing braces")
        
        # If there are still Parse errors, try to retain methods with @Test annotation
        if not fixed or any("Parse error" in err for err in remaining_errors):
            test_methods = []
            current_pos = 0
            
            # Find all methods with @Test annotation
            while current_pos < len(test_code):
                test_annotation_pos = test_code.find("@Test", current_pos)
                if test_annotation_pos == -1:
                    break
                
                # Find the start and end of the method
                method_start = test_code.find("{", test_annotation_pos)
                if method_start == -1:
                    break
                
                # Record the method start position
                test_methods.append(test_annotation_pos)
                current_pos = method_start + 1
            
            if test_methods:
                # If test methods are found, retain the most complete one
                last_test_pos = test_methods[-1]
                # Try to find the class definition for this method
                class_pos = test_code.rfind("class", 0, last_test_pos)
                
                if class_pos != -1:
                    # Build a new test class containing only the last test method
                    class_def_end = test_code.find("{", class_pos)
                    class_def = test_code[class_pos:class_def_end+1]
                    
                    last_method_code = test_code[last_test_pos:]
                    # Ensure the method code is properly closed
                    open_method_braces = last_method_code.count('{')
                    close_method_braces = last_method_code.count('}')
                    if open_method_braces > close_method_braces:
                        last_method_code += "}" * (open_method_braces - close_method_braces)
                    
                    # Use string concatenation to avoid f-string brace escaping confusion
                    new_test_code = f"{class_def}\n    {last_method_code}\n" + "}"
                    test_code = new_test_code
                    fixed = True
                    logger.info("Refactored test class to retain the last test method")
    
    # Rule 2: Fix "cannot find symbol" errors (usually missing import statements)
    if any("cannot find symbol" in err for err in errors):
        logger.info("Attempting to fix missing import statements")
        
        # Find the source code file to get its import statements
        source_file = find_source_code(project_dir, class_name, package_name)
        if source_file:
            source_code = read_source_code(source_file)
            
            # Extract import statements from the source code
            import_statements = []
            lines = source_code.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("static import ") or "import " in line:
                    import_statements.append(line)
            
            # Check if the test code already contains a package declaration
            if "package " in test_code:
                # Add import statements after the package declaration
                package_end = test_code.find(';', test_code.find("package ")) + 1
                imports_block = '\n'.join(import_statements)
                test_code = test_code[:package_end] + '\n\n' + imports_block + test_code[package_end:]
            else:
                # Add package declaration and import statements at the beginning of the code
                imports_block = "package " + package_name + ";\n\n" + '\n'.join(import_statements)
                test_code = imports_block + '\n\n' + test_code
            
            fixed = True
            logger.info(f"Added {len(import_statements)} import statements from the source code")
    
    # Rule 3: Fix common JUnit-related issues
    if any("org.junit" in err for err in errors):
        logger.info("Attempting to fix JUnit-related issues")
        
        # Check if JUnit imports are missing
        if "import org.junit" not in test_code:
            junit5_imports = [
                "import org.junit.jupiter.api.Test;",
                "import org.junit.jupiter.api.Assertions;",
                "import static org.junit.jupiter.api.Assertions.*;",
                "import org.junit.jupiter.api.BeforeEach;",
                "import org.junit.jupiter.api.AfterEach;"
            ]
            
            # Check if the test code already contains a package declaration
            if "package " in test_code:
                # Add import statements after the package declaration
                package_end = test_code.find(';', test_code.find("package ")) + 1
                junit_imports = '\n'.join(junit5_imports)
                test_code = test_code[:package_end] + '\n\n' + junit_imports + test_code[package_end:]
            else:
                # Add package declaration and import statements at the beginning of the code
                junit_imports = "package " + package_name + ";\n\n" + '\n'.join(junit5_imports)
                test_code = junit_imports + '\n\n' + test_code
            
            fixed = True
            logger.info("Added JUnit 5 import statements")
    
    # Rule 4: Fix Mockito-related issues
    if any("org.mockito" in err for err in errors) or any("mock" in err.lower() for err in errors):
        logger.info("Attempting to fix Mockito-related issues")
        
        # Add Mockito imports
        if "import org.mockito" not in test_code:
            mockito_imports = [
                "import org.mockito.Mockito;",
                "import static org.mockito.Mockito.*;",
                "import org.mockito.Mock;",
                "import org.mockito.InjectMocks;",
                "import org.mockito.junit.jupiter.MockitoExtension;",
                "import org.junit.jupiter.api.extension.ExtendWith;"
            ]
            
            # Check if the test code already contains a package declaration
            if "package " in test_code:
                # Add import statements after the package declaration
                package_end = test_code.find(';', test_code.find("package ")) + 1
                mockito_imports_text = '\n'.join(mockito_imports)
                test_code = test_code[:package_end] + '\n\n' + mockito_imports_text + test_code[package_end:]
            else:
                # Add package declaration and import statements at the beginning of the code
                mockito_imports_text = "package " + package_name + ";\n\n" + '\n'.join(mockito_imports)
                test_code = mockito_imports_text + '\n\n' + test_code
            
            # Check if the test class has already added MockitoExtension
            if "@ExtendWith" not in test_code and "class " in test_code:
                class_start = test_code.find("class ")
                class_end = test_code.find("{", class_start)
                
                if class_start != -1 and class_end != -1:
                    extension_annotation = "@ExtendWith(MockitoExtension.class)\n"
                    test_code = test_code[:class_start] + extension_annotation + test_code[class_start:]
                    fixed = True
                    logger.info("Added Mockito extension annotation")
            
            fixed = True
            logger.info("Added Mockito import statements")
    
    # Check if the repair was successful
    if fixed and test_code != original_test_code:
        # Save the repaired code and validate it
        test_file_path = save_test_code(test_code, class_name, package_name, project_dir)
        if test_file_path:
            # Build the test class name
            test_class_name = f"{class_name}Test"
            test_class = f"{package_name}.{test_class_name}"
            
            # Run the tests and check for errors
            coverage_data, new_errors, execution_time, _ = run_tests_with_jacoco(project_dir, class_name, package_name, test_class)
            
            if not new_errors or len(new_errors) < len(errors):
                logger.info("Rule-based repair successfully reduced the number of errors")
                return test_code, True, new_errors
    
    # If no repair was made or the repair was unsuccessful, return the original code and errors
    return test_code, fixed, remaining_errors


def improve_test_coverage(project_dir, prompt_dir, test_prompt_file, class_name, package_name, test_code, source_code, max_attempts=20, target_coverage=95.0):
    """
    Iteratively optimize test coverage while tracking the best results.
    """
    attempt = 0
    current_coverage = 0.0
    has_errors = True

    # Retrieve the original prompt content
    prompt_content = read_test_prompt_file(prompt_dir, class_name)

    # Store test result history
    history = []

    # Track the best test code
    best_test_code = test_code
    best_coverage = 0.0
    best_has_errors = True

    # Remove comments from the source code to reduce tokens
    cleaned_source_code = strip_java_comments(source_code)

    while attempt < max_attempts:
        try:
            logger.info(f"Attempt #{attempt+1}: Saving and running tests")

            # Save the current test code
            test_file_path = save_test_code(test_code, class_name, package_name, project_dir)
            if not test_file_path:
                logger.error("Failed to save test code, terminating optimization")
                break

            # Construct the test class name
            test_class_name = f"{class_name}Test"
            test_class = f"{package_name}.{test_class_name}"

            # Run tests and obtain coverage data
            coverage_data, assertion_failures, execution_time, errors = run_tests_with_jacoco(project_dir, class_name, package_name, test_class)

            # Check for errors
            has_errors = bool(errors)

            # Get the current coverage percentage
            current_coverage = get_coverage_percentage(coverage_data)

            # Record the current attempt results
            history.append({
                "attempt": attempt + 1,
                "coverage": current_coverage,
                "has_errors": has_errors,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            logger.info(f"Attempt #{attempt+1} Result: Coverage={current_coverage:.2f}%")

            # If there are errors, try applying rule-based fixes first
            if has_errors:
                logger.info("Test has errors, attempting rule-based fixes")
                repaired_code, fixed, remaining_errors = apply_rule_based_repairs(
                    test_code, errors, class_name, package_name, project_dir
                )

                if fixed:
                    logger.info("Rule-based fixes applied successfully")
                    # If fixes are successful, rerun tests with the repaired code
                    test_code = repaired_code

                    # Save the repaired code
                    test_file_path = save_test_code(test_code, class_name, package_name, project_dir)
                    if test_file_path:
                        # Rerun tests
                        coverage_data, assertion_failures, execution_time, errors = run_tests_with_jacoco(project_dir, class_name, package_name, test_class)

                        # Update status
                        has_errors = bool(errors)
                        current_coverage = get_coverage_percentage(coverage_data)

                        # Update history record
                        history.append({
                            "attempt": attempt + 1.1,  # Use decimal to indicate rule-based repair results
                            "coverage": current_coverage,
                            "has_errors": has_errors,
                            "repair_method": "rule-based",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        logger.info(f"Post rule-based fix results: Coverage={current_coverage:.2f}%, Errors={has_errors}")

            # Update the best test code if:
            # 1. Current coverage is higher than the best coverage, or
            # 2. Current coverage is equal to the best coverage, but the current test has no errors while the best one does
            if (current_coverage > best_coverage) or (current_coverage == best_coverage and not has_errors and best_has_errors):
                best_test_code = test_code
                best_coverage = current_coverage
                best_has_errors = has_errors
                logger.info(f"New best test found: Coverage={best_coverage:.2f}%, Errors={best_has_errors}")

                # Save the best test code with a special suffix
                # best_file_path = save_test_code(best_test_code, f"{class_name}_Best", package_name, project_dir)
                # logger.info(f"Best test code saved at: {best_file_path}")

            # Check termination conditions
            if (not has_errors and current_coverage >= target_coverage) or attempt >= max_attempts - 1:
                logger.info(f"Termination conditions met: Coverage={current_coverage:.2f}% (Target={target_coverage}%), Errors={has_errors}, Attempts={attempt+1}/{max_attempts}")
                break

            # Prepare comprehensive feedback
            feedback = format_feedback_for_prompt(coverage_data, errors, class_name, package_name, prompt_content)

            # Check for assertion failures
            has_assertion_failures = any("assertion failed" in err.lower() or "expected:" in err.lower() or "AssertionFailedError" in err for err in errors)

            # Construct an optimization prompt using the cleaned source code
            try:
                with open(test_prompt_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read prompt file: {str(e)}")
                return ""

            improve_prompt = f"""
            CRITICAL: I need the ENTIRE test class including ALL original methods, not just the fixed parts.
Your response must contain:
1. All package declarations
2. All import statements 
3. The complete class definition
4. ALL existing test methods, not just the fixed ones
5. All fields and setup methods

Format your entire response as a SINGLE complete Java file that I can save and run directly.
===============================
JAVA CLASS UNIT TEST GENERATION WITH FEEDBACK
===============================
-----------------
SOURCE CODE
-----------------
```java
{source_code}
```

-----------------
CURRENT TEST CODE
-----------------
```java
{test_code}
```

-----------------
TEST FEEDBACK
-----------------
{feedback} {"Please prioritize fixing assertion failures. Carefully analyze the differences between expected and actual values, and adjust test assertions according to the actual behavior of the source code." if has_assertion_failures else ""} Based on the above feedback, improve the unit tests for this class. Focus on: {"1. Fixing assertion failures" if has_assertion_failures else "1. Increasing code coverage"} 2. Adding tests for uncovered lines and branches 3. Resolving all test errors and failures 4. Adhering to all test requirements from the original prompt

Please provide a complete JUnit test class that addresses these issues. Do not include explanations, only return the improved Java test code. """
            logger.info(f"Calling API to improve test code (Attempt #{attempt+1})") 
            api_response = call_anthropic_api(improve_prompt) 
            # api_response = call_gpt_api(improve_prompt)
            # api_response = call_deepseek_api(improve_prompt)
            # Extract Java code
            improved_test_code = extract_java_code(api_response)

            if not improved_test_code:
                logger.error("Failed to extract test code from API response, terminating optimization")
                break

            # Update test code
            test_code = improved_test_code
            attempt += 1
        except Exception as e:
            logger.error(f"Error during improvement attempt: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    # Save optimization history
    try:
        history_file = os.path.join(project_dir, f"{class_name}_test_history.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Test optimization history saved at: {history_file}")
    except Exception as e:
        logger.error(f"Failed to save optimization history: {str(e)}")

    # Save the best test code as the final result (overwrite last attempt)
    final_file_path = save_test_code(best_test_code, class_name, package_name, project_dir)
    logger.info(f"Best test code saved as final result: {final_file_path}")
    logger.info(f"Best achieved coverage: {best_coverage:.2f}%, Errors={best_has_errors}")

    # Return the best result instead of the last attempt
    return best_test_code, best_coverage, best_has_errors, attempt


def generate_test_summary(project_dir, class_name, package_name, coverage, 
                       has_errors, iterations, status, history=None):
    """
    Generate a comprehensive test summary file
    
    Parameters:
    project_dir (str): Project directory
    class_name (str): Class name
    package_name (str): Package name
    coverage (float): Best coverage percentage
    has_errors (bool): Whether final test has errors
    iterations (int): Number of iterations performed
    status (str): Test status
    history (list): Optional history entries from MCTS execution
    
    Returns:
    str: Path to summary file
    """
    summary_file = os.path.join(project_dir, f"{class_name}_test_summary.json")
    
    try:
        # If history not provided, try to load from file
        if history is None:
            history_file = os.path.join(project_dir, f"{class_name}_test_history.json")
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
        
        # 加载逻辑指标，以获取bug相关信息
        logic_metrics_file = os.path.join(project_dir, f"{class_name}_logic_metrics.json")
        logic_metrics = {}
        if os.path.exists(logic_metrics_file):
            try:
                with open(logic_metrics_file, 'r', encoding='utf-8') as f:
                    logic_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load logic metrics: {str(e)}")
        
        # 从逻辑指标中获取bug信息
        logical_bugs_found = logic_metrics.get("total_logical_bug_tests", 0)
        logical_bug_types = logic_metrics.get("logical_bug_types_found", [])
        bugs_found_iteration = logic_metrics.get("iterations_to_first_logical_bug")
        
        # Create summary data
        summary = {
            "class_name": class_name,
            "package_name": package_name,
            "best_coverage": coverage,
            "has_errors": has_errors,
            "iterations": iterations,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "logical_bugs_found": logical_bugs_found,
            "logical_bug_types": logical_bug_types,
            "bugs_found_iteration": bugs_found_iteration,
            "history": history
        }
        
        # Generate coverage trend if history available
        if history:
            # Extract coverage data per iteration
            coverage_trend = []
            
            for i, entry in enumerate(history):
                trend_entry = {
                    "iteration": entry.get("iteration", i+1), 
                    "coverage": entry.get("coverage", 0.0),
                    "best_coverage": entry.get("current_best_coverage", 0.0),
                    "reward": entry.get("reward", 0.0)
                }
                
                # 添加bug信息
                if "detected_bugs" in entry:
                    trend_entry["detected_bugs"] = len(entry.get("detected_bugs", []))
                elif "bugs_found" in entry:
                    trend_entry["detected_bugs"] = entry.get("bugs_found", 0)
                else:
                    trend_entry["detected_bugs"] = 0
                    
                coverage_trend.append(trend_entry)
            
            summary["coverage_trend"] = coverage_trend
            
            # Add bug detection trend - 优先使用entry中的bug信息，如果没有则从bug_details中提取
            bug_trend = []
            
            for i, entry in enumerate(history):
                bug_entry = {
                    "iteration": entry.get("iteration", i+1)
                }
                
                # 获取detected_bugs
                if "detected_bugs" in entry and isinstance(entry["detected_bugs"], list):
                    bug_entry["detected_bugs"] = len(entry["detected_bugs"])
                elif "bugs_found" in entry:
                    bug_entry["detected_bugs"] = entry.get("bugs_found", 0)
                else:
                    bug_entry["detected_bugs"] = 0
                
                # 获取verified_bugs
                if "verified_bugs" in entry and isinstance(entry["verified_bugs"], list):
                    bug_entry["verified_bugs"] = len(entry["verified_bugs"])
                elif "bug_details" in entry:
                    # 从bug详情中提取已验证的bug
                    verified_count = sum(1 for bug in entry["bug_details"] 
                                        if bug.get("verified", False) and bug.get("is_real_bug", False))
                    bug_entry["verified_bugs"] = verified_count
                else:
                    bug_entry["verified_bugs"] = 0
                
                bug_trend.append(bug_entry)
            
            summary["bug_trend"] = bug_trend
            
            # 添加bug详情
            bug_details = []
            for i, entry in enumerate(history):
                if "bug_details" in entry and entry["bug_details"]:
                    for bug in entry["bug_details"]:
                        bug_info = {
                            "iteration": entry.get("iteration", i+1),
                            "method": bug.get("method", "unknown"),
                            "type": bug.get("type", "unknown"),
                            "verified": bug.get("verified", False),
                            "is_real_bug": bug.get("is_real_bug", False)
                        }
                        bug_details.append(bug_info)
            
            if bug_details:
                summary["bug_details"] = bug_details
            
            # Add performance stats
            performance_stats = {
                "avg_execution_time": sum(entry.get("execution_time", 0) for entry in history) / len(history),
                "max_execution_time": max(entry.get("execution_time", 0) for entry in history),
                "coverage_improvement_rate": (coverage - history[0].get("coverage", 0)) / max(1, len(history)),
                "final_iterations": len(history)
            }
            
            summary["performance_stats"] = performance_stats
        
        # Save summary to file
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Test summary saved to: {summary_file}")
        return summary_file
    
    except Exception as e:
        logger.error(f"Failed to generate test summary: {str(e)}")
        logger.error(traceback.format_exc())
        return None

        
def create_consolidated_report(project_dir, results):
    """
    Create a consolidated report for batch processing
    
    Parameters:
    project_dir (str): Project directory
    results (list): Batch processing results
    
    Returns:
    str: Path to consolidated report
    """
    report_file = os.path.join(project_dir, "test_generation_report.json")
    
    try:
        # Sort results by coverage (highest first)
        sorted_results = sorted(results, key=lambda x: x.get("coverage", 0.0), reverse=True)
        
        # Calculate summary statistics
        total_classes = len(results)
        successful_classes = sum(1 for r in results if r.get("success", False))
        average_coverage = sum(r.get("coverage", 0.0) for r in results) / total_classes if total_classes > 0 else 0.0
        
        # Create report data
        report = {
            "summary": {
                "total_classes": total_classes,
                "successful_classes": successful_classes,
                "failed_classes": total_classes - successful_classes,
                "success_rate": (successful_classes / total_classes * 100) if total_classes > 0 else 0.0,
                "average_coverage": average_coverage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "class_results": sorted_results
        }
        
        # Save report to file
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Consolidated test generation report saved to: {report_file}")
        return report_file
    
    except Exception as e:
        logger.error(f"Failed to create consolidated report: {str(e)}")
        return None

def process_class(project_dir, prompt_dir, class_name, package_name, max_attempts=20, target_coverage=100.0):
    """
    Process single class test generation and optimization
    
    Parameters:
    project_dir (str): Project directory
    prompt_dir (str): Prompt directory
    class_name (str): Class name
    package_name (str): Package name
    max_attempts (int): Maximum attempt count
    target_coverage (float): Target coverage percentage
    
    Returns:
    tuple: (success, coverage, has_errors, test_code)
    """
    logger.info(f"Starting to process class: {package_name}.{class_name}")
    
    # 1. Read test prompt file
    test_prompt_file = os.path.join(prompt_dir, f"{class_name}_test_prompt.txt")
    if not os.path.exists(test_prompt_file):
        test_prompt_file = os.path.join(prompt_dir, f"{class_name}.txt")
        if not os.path.exists(test_prompt_file):
            logger.error(f"Test prompt file not found: {class_name}_test_prompt.txt or {class_name}.txt")
            return False, 0.0, True, ""
    
    # 2. Find and read source code
    source_file = find_source_code(project_dir, class_name, package_name)
    if not source_file:
        logger.error(f"Source code file not found: {class_name}.java")
        return False, 0.0, True, ""
    
    source_code = read_source_code(source_file)
    if not source_code:
        logger.error(f"Failed to read source code")
        return False, 0.0, True, ""
    
    # 3. Generate initial tests
    logger.info("Generating initial test code")
    initial_test = generate_initial_test(test_prompt_file, source_code)
    
    if not initial_test:
        logger.error("Initial test generation failed")
        return False, 0.0, True, ""
    
    # 4. Iteratively optimize test coverage
    logger.info("Starting iterative test code optimization")
    best_test, best_coverage, has_errors, attempts = improve_test_coverage(
        project_dir, prompt_dir, test_prompt_file, class_name, package_name, initial_test, source_code, max_attempts, target_coverage)
    
    # 5. Final test save (already done in improve_test_coverage)
    
    # 6. Output result summary
    status = "Success" if not has_errors and best_coverage >= target_coverage else "Not meeting standards"
    logger.info(f"Class {package_name}.{class_name} processing completed")
    logger.info(f"Best coverage: {best_coverage:.2f}%")
    logger.info(f"Iteration count: {attempts + 1}")
    logger.info(f"Final status: {status}")
    
    
    # 7. Generate comprehensive summary 
    generate_test_summary(project_dir, class_name, package_name, best_coverage, has_errors, attempts + 1, status)
    
    return True, best_coverage, has_errors, best_test

def batch_process_classes(project_dir, prompt_dir, output_file=None, max_attempts=20, target_coverage=100.0):
    """
    Batch process all classes in directory, tracking best tests
    
    Parameters:
    project_dir (str): Project directory
    prompt_dir (str): Prompt directory
    output_file (str): Output result file
    max_attempts (int): Maximum attempt count
    target_coverage (float): Target coverage percentage
    
    Returns:
    list: Processing result list
    """
    # Find all test prompt files
    prompt_files = glob.glob(os.path.join(prompt_dir, "*_test_prompt.txt"))
    prompt_files.extend(glob.glob(os.path.join(prompt_dir, "*.txt")))
    
    # Filter valid prompt files (exclude _improved.txt etc.)
    valid_files = [f for f in prompt_files if "_improved" not in f and "_history" not in f and "_summary" not in f and "_best" not in f]
    
    if not valid_files:
        logger.error(f"No test prompt files found in {prompt_dir}")
        return []
    
    logger.info(f"Found {len(valid_files)} test prompt files, starting batch processing")
    
    results = []
    success_count = 0
    # best_results_dir = "/Users/ruiqidong/Desktop/unittest/dataset/cli_best_tests/src/test/java/org/apache/commons/cli"
    # os.makedirs(best_results_dir, exist_ok=True)
    
    for file_path in valid_files:
        class_name = os.path.basename(file_path).replace("_test_prompt.txt", "").replace(".txt", "")
        package_name = extract_package_from_file(file_path)
        
        if not package_name:
            logger.warning(f"Could not extract package name from {file_path}, skipping")
            continue
        
        logger.info(f"Starting processing: {package_name}.{class_name}")
        
        try:
            success, coverage, has_errors, test_code = process_class(
                project_dir, prompt_dir, class_name, package_name, max_attempts, target_coverage)
            
            if success and not has_errors and coverage >= target_coverage:
                success_count += 1
            
            # Record result
            result = {
                "class_name": class_name,
                "package_name": package_name,
                "coverage": coverage,
                "has_errors": has_errors,
                "success": success and not has_errors and coverage >= target_coverage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)
            
            # Copy the best test to the best_tests directory
            # if test_code:
            #     best_test_file = os.path.join(best_results_dir, f"{class_name}Test.java")
            #     try:
            #         with open(best_test_file, 'w', encoding='utf-8') as f:
            #             f.write(test_code)
            #         result["best_test_path"] = best_test_file
            #         logger.info(f"Best test for {class_name} copied to: {best_test_file}")
            #     except Exception as e:
            #         logger.error(f"Failed to copy best test: {str(e)}")
            
            # Save intermediate results
            # if output_file:
            #     try:
            #         with open(output_file, 'w', encoding='utf-8') as f:
            #             json.dump(results, f, indent=2)
            #     except Exception as e:
            #         logger.error(f"Failed to save intermediate results: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error occurred while processing {class_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            results.append({
                "class_name": class_name,
                "package_name": package_name,
                "coverage": 0.0,
                "has_errors": True,
                "success": False,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Create consolidated report
    create_consolidated_report(project_dir, results)
    
    # Output summary
    logger.info("Batch processing completed")
    logger.info(f"Total: {len(results)} classes")
    logger.info(f"Success: {success_count} classes")
    logger.info(f"Failed: {len(results) - success_count} classes")
    logger.info(f"Best tests directory: {best_results_dir}")
    
    # Save final results
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    return results

def extract_package_from_file(file_path):
    """
    Extract package name from test prompt file
    
    Parameters:
    file_path (str): File path
    
    Returns:
    str: Package name
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            package_match = re.search(r'Package:\s*([\w.]+)', content)
            if package_match:
                return package_match.group(1)
    except:
        pass
    return None

def check_pom_for_jacoco(project_dir):
    """
    Check if project's pom.xml file includes Jacoco plugin
    
    Parameters:
    project_dir (str): Project directory
    
    Returns:
    bool: Whether Jacoco plugin is found
    """
    pom_file = os.path.join(project_dir, "pom.xml")
    if not os.path.exists(pom_file):
        logger.warning("pom.xml file not found, cannot check Jacoco configuration")
        return False
    
    try:
        with open(pom_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if includes jacoco plugin
        if 'jacoco-maven-plugin' in content:
            logger.info("Found Jacoco plugin configuration in pom.xml")
            return True
        else:
            logger.warning("Jacoco plugin configuration not found in pom.xml")
            return False
    except Exception as e:
        logger.error(f"Failed to read pom.xml file: {str(e)}")
        return False

def add_jacoco_to_pom(project_dir):
    """
    Add Jacoco plugin configuration to pom.xml file
    
    Parameters:
    project_dir (str): Project directory
    
    Returns:
    bool: Whether successfully added
    """
    pom_file = os.path.join(project_dir, "pom.xml")
    if not os.path.exists(pom_file):
        logger.error("pom.xml file not found, cannot add Jacoco configuration")
        return False
    
    try:
        with open(pom_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If already includes jacoco plugin, skip
        if 'jacoco-maven-plugin' in content:
            logger.info("pom.xml already includes Jacoco plugin configuration, no need to add")
            return True
        
        # Find build tag
        build_end_match = re.search(r'</build>', content)
        if not build_end_match:
            logger.error("Could not find </build> tag in pom.xml, cannot add Jacoco configuration")
            return False
        
        # Build Jacoco plugin configuration
        jacoco_config = """
        <plugin>
            <groupId>org.jacoco</groupId>
            <artifactId>jacoco-maven-plugin</artifactId>
            <version>0.8.8</version>
            <executions>
                <execution>
                    <goals>
                        <goal>prepare-agent</goal>
                    </goals>
                </execution>
                <execution>
                    <id>report</id>
                    <phase>test</phase>
                    <goals>
                        <goal>report</goal>
                    </goals>
                </execution>
            </executions>
        </plugin>
        """
        
        # Add Jacoco configuration before build tag ends
        pos = build_end_match.start()
        new_content = content[:pos] + jacoco_config + content[pos:]
        
        # Save modified pom.xml
        with open(pom_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info("Successfully added Jacoco plugin configuration to pom.xml")
        return True
    
    except Exception as e:
        logger.error(f"Failed to modify pom.xml file: {str(e)}")
        return False

def log_detailed_metrics(output_file=None):
    """
    Log detailed metrics about LLM usage with comprehensive statistics.
    
    Parameters:
    output_file (str): Optional file to save the detailed metrics to
    
    Returns:
    dict: Detailed metrics
    """
    global llm_metrics
    
    # Ensure end time is set
    if llm_metrics["end_time"] is None:
        llm_metrics["end_time"] = time.time()
    
    # Calculate basic statistics
    total_time = llm_metrics["end_time"] - llm_metrics["start_time"]
    token_sizes = llm_metrics["token_sizes"]
    request_times = llm_metrics["request_times"]
    
    # Skip if no data
    if not token_sizes:
        return {"error": "No metrics data available"}
    
    # Calculate detailed statistics
    detailed_metrics = {
        "total_requests": llm_metrics["request_count"],
        "token_sizes": {
            "max": max(token_sizes),
            "min": min(token_sizes),
            "mean": statistics.mean(token_sizes),
            "median": statistics.median(token_sizes),
            "p90": sorted(token_sizes)[int(0.9 * len(token_sizes))],
            "p95": sorted(token_sizes)[int(0.95 * len(token_sizes))],
            "p99": sorted(token_sizes)[int(0.99 * len(token_sizes))] if len(token_sizes) >= 100 else max(token_sizes),
            "total_tokens": sum(token_sizes)
        },
        "time": {
            "total_seconds": total_time,
            "total_minutes": total_time / 60,
            "request_times": {
                "max": max(request_times) if request_times else 0,
                "min": min(request_times) if request_times else 0,
                "mean": statistics.mean(request_times) if request_times else 0,
                "median": statistics.median(request_times) if request_times else 0,
                "p90": sorted(request_times)[int(0.9 * len(request_times))] if request_times else 0,
                "total": sum(request_times) if request_times else 0
            }
        },
        "distribution": {
            "tokens_histogram": _create_histogram(token_sizes, 10),
            "time_histogram": _create_histogram(request_times, 10)
        },
        "timestamp": {
            "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(llm_metrics["start_time"])),
            "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(llm_metrics["end_time"]))
        }
    }
    
    # Calculate requests per minute 
    duration_minutes = total_time / 60
    if duration_minutes > 0:
        detailed_metrics["requests_per_minute"] = llm_metrics["request_count"] / duration_minutes
    else:
        detailed_metrics["requests_per_minute"] = llm_metrics["request_count"]
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_metrics, f, indent=2)
            logging.info(f"Detailed metrics saved to: {output_file}")
        except Exception as e:
            logging.error(f"Failed to save detailed metrics: {str(e)}")
    
    return detailed_metrics

def _create_histogram(data, num_bins):
    """Create a simple histogram of data."""
    if not data:
        return []
    
    min_val = min(data)
    max_val = max(data)
    bin_size = (max_val - min_val) / num_bins if max_val > min_val else 1
    
    # Initialize bins
    bins = [0] * num_bins
    
    # Count values in each bin
    for value in data:
        bin_index = min(num_bins - 1, int((value - min_val) / bin_size))
        bins[bin_index] += 1
    
    # Create histogram data
    histogram = []
    for i in range(num_bins):
        lower_bound = min_val + i * bin_size
        upper_bound = lower_bound + bin_size if i < num_bins - 1 else max_val
        histogram.append({
            "range": f"{lower_bound:.2f} - {upper_bound:.2f}",
            "count": bins[i],
            "percentage": (bins[i] / len(data)) * 100
        })
    
    return histogram

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GPT auto-generate and optimize Java unit tests')
    parser.add_argument('--project', required=True, help='Java project root directory')
    parser.add_argument('--prompt', required=True, help='Directory containing test prompts')
    parser.add_argument('--class', dest='class_name', help='Class name to test')
    parser.add_argument('--package', help='Package name of the class')
    parser.add_argument('--output', help='Output result file path')
    parser.add_argument('--batch', action='store_true', help='Batch process all classes')
    parser.add_argument('--max-attempts', type=int, default=10, help='Maximum attempt count')
    parser.add_argument('--target-coverage', type=float, default=100.0, help='Target coverage percentage')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--api-base', help='OpenAI API base URL')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Model to use')
    parser.add_argument('--check-jacoco', action='store_true', help='Check and add Jacoco configuration')
    # parser.add_argument('--best-dir', help='Directory to save best test results', default='best_tests')
    
    args = parser.parse_args()
    
    # Set API key and URL
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    if args.api_base:
        global API_BASE
        API_BASE = args.api_base
    
    if args.model:
        globals()["DEFAULT_MODEL"] = args.model
    
    # Check API key
    if not API_KEY:
        parser.error("Please provide OpenAI API key, can be set via --api-key parameter or OPENAI_API_KEY environment variable")
    
    # Check if project directory exists
    if not os.path.exists(args.project):
        parser.error(f"Project directory does not exist: {args.project}")
    
    # Check if prompt directory exists
    if not os.path.exists(args.prompt):
        parser.error(f"Prompt directory does not exist: {args.prompt}")
    
    # Check and add Jacoco configuration
    if args.check_jacoco:
        if not check_pom_for_jacoco(args.project):
            logger.info("Trying to add Jacoco plugin to pom.xml...")
            add_jacoco_to_pom(args.project)
    
    # Create directory for best tests
    # best_dir = args.best_dir
    # if not os.path.exists(os.path.join(args.project, best_dir)):
    #     os.makedirs(os.path.join(args.project, best_dir), exist_ok=True)
    #     logger.info(f"Created directory for best tests: {os.path.join(args.project, best_dir)}")
    
    # Batch process or single class process
    if args.batch:
        results = batch_process_classes(
            args.project, 
            args.prompt, 
            args.output, 
            args.max_attempts, 
            args.target_coverage
        )
        
        # Generate consolidated report
        report_path = create_consolidated_report(args.project, results)
        if report_path:
            logger.info(f"Generated consolidated report at: {report_path}")
        
    # elif args.class_name and args.package:
    #     success, coverage, has_errors, test_code = process_class(
    #         args.project, 
    #         args.prompt, 
    #         args.class_name, 
    #         args.package, 
    #         args.max_attempts, 
    #         args.target_coverage
    #     )
        
    #     if success:
    #         status = "Success" if not has_errors and coverage >= args.target_coverage else "Not meeting standards"
    #         logger.info(f"Class {args.package}.{args.class_name} processed with {status}")
    #         logger.info(f"Coverage: {coverage:.2f}%")
    #         logger.info(f"Has errors: {has_errors}")
            
    #         # Save the best test to the best tests directory
    #         # if test_code:
    #         #     best_test_file = os.path.join(args.project, best_dir, f"{args.class_name}Test.java")
    #         #     try:
    #         #         with open(best_test_file, 'w', encoding='utf-8') as f:
    #         #             f.write(test_code)
    #         #         logger.info(f"Best test for {args.class_name} saved to: {best_test_file}")
    #         #     except Exception as e:
    #         #         logger.error(f"Failed to save best test: {str(e)}")
    #     else:
    #         logger.error(f"Failed to process class {args.package}.{args.class_name}")
    else:
        parser.error("Must specify --batch or both --class and --package")

if __name__ == "__main__":
    main()