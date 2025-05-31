#!/usr/bin/env python3
import os
import subprocess
import argparse

def extract_project_name(project_path):
    """Extract the project name from the project path"""
    return os.path.basename(project_path)

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run analysis commands in sequence')
    parser.add_argument('project_path', help='Path to the project being tested')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--class_name', required=True, help='Name of the class being tested')
    parser.add_argument('--package', required=True, help='Package of the class being tested')
    # parser.add_argument('--joda_subdir', default='JodaTime', 
    #                     help='Subdirectory for the project (default: JodaTime)')
    
    args = parser.parse_args()
    
    # Extract project name from path
    project_name = extract_project_name(args.project_path)
    
    # Command 1: Run main.py
    cmd1 = ["python", "main.py", args.project_path, "--output_dir", args.output_dir]
    print(f"Running command 1: {' '.join(cmd1)}")
    subprocess.run(cmd1, check=True)
    
    # Command 2: Run prompt_generator.py
    json_path = os.path.join(args.output_dir, project_name, f"{project_name}_combined_analysis.json")
    prompts_dir = os.path.join(args.output_dir, project_name, "prompts")
    cmd2 = ["python", "prompt_generator.py", json_path, "--output_dir", prompts_dir]
    print(f"Running command 2: {' '.join(cmd2)}")
    subprocess.run(cmd2, check=True)
    
    # Command 3: Run mcts_integrated_feedback.py
    # joda_path = os.path.join(args.project_path, args.joda_subdir)
    # cmd3 = ["python", "mcts_integrated_feedback.py", 
    #         "--project", args.project_path, 
    #         "--prompt", prompts_dir, 
    #         "--class", args.class_name, 
    #         "--package", args.package]
    
    cmd3 = ["python", "lambda_framework.py", 
            "--project", args.project_path, 
            "--prompt", prompts_dir, 
            "--class", args.class_name, 
            "--package", args.package]

    # cmd3 = ["python", "lambda_framework.py", 
    #         "--project", args.project_path, 
    #         "--prompt", prompts_dir, 
    #         "--batch"]
    
    print(f"Running command 3: {' '.join(cmd3)}")
    subprocess.run(cmd3, check=True)
    
    print("All commands executed successfully!")

if __name__ == "__main__":
    main()