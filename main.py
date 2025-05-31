import os
import json
import argparse
from file_analyzer import analyze_java_file
from dependency_analyzer import analyze_java_project
from indirect_dependency_analyzer import EnhancedJavaDependencyAnalyzer
from boundary_exception_analyzer import analyze_boundary_and_exception

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def analyze_project(project_path: str) -> dict:
    project_info = {}
    total_files = 0
    successful_files = 0
    failed_files = 0
    
    print(f"Starting project analysis: {project_path}")
    
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if 'Test' in file:
                continue
            # Skip module-info.java files as they use Java 9+ module syntax which javalang can't parse
            if file == 'module-info.java':
                continue
            if file.endswith('.java'):
                total_files += 1
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)
                
                print(f"Analyzing: {relative_path}")
                
                try:
                    result = analyze_java_file(file_path)
                    
                    # Check if fallback method was used
                    if "parsing_method" in result:
                        if result["parsing_method"] == "regex_fallback":
                            print(f"  -> Using regex fallback parsing")
                        elif result["parsing_method"] == "preprocessed_javalang":
                            print(f"  -> Using preprocessed javalang parsing")
                        else:
                            print(f"  -> Using {result['parsing_method']} parsing")
                    
                    project_info[relative_path] = result
                    successful_files += 1
                    
                except Exception as e:
                    failed_files += 1
                    print(f"  -> Parsing failed: {str(e)}")
                    
                    # Create minimal error record
                    project_info[relative_path] = {
                        "error": str(e),
                        "classes": [],
                        "interfaces": [],
                        "enums": [],
                        "parsing_method": "failed"
                    }
    
    print(f"\nAnalysis completed:")
    print(f"  Total files: {total_files}")
    print(f"  Successfully parsed: {successful_files}")
    print(f"  Parsing failed: {failed_files}")
    print(f"  Success rate: {(successful_files/total_files*100):.1f}%" if total_files > 0 else "  Success rate: 0%")
    
    return project_info

def save_json(data: dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=SetEncoder)

# def main():
#     project_name = "spring-boot-package"
#     # project_name = "Tutorial_Stack"
#     project_path = f"../samples/{project_name}"
#     output_dir = f"../results/static_analysis/{project_name}"
    
#     # File analysis (including data flow graph)
#     output_file_dfg = os.path.join(output_dir, f"{project_name}_dfg.json")
#     project_info_dfg = analyze_project(project_path)
#     save_json(project_info_dfg, output_file_dfg)
#     print(f"Data flow graph analysis completed, results saved to {output_file_dfg}")
    
#     # Dependency analysis
#     output_file_dep = os.path.join(output_dir, f"{project_name}_dependency.json")
#     project_info_dep = analyze_java_project(project_path)
#     save_json(project_info_dep, output_file_dep)
#     print(f"Dependency analysis completed, results saved to {output_file_dep}")
    
#     # Indirect dependency analysis
#     output_file_idc = os.path.join(output_dir, f"{project_name}_IDC.json")
#     analyzer = EnhancedJavaDependencyAnalyzer(project_path)
#     analyzer.analyze()
#     analyzer.save_to_json(output_file_idc)
#     print(f"Indirect dependency analysis completed, results saved to {output_file_idc}")

    # Boundary condition and exception handling analysis
    # output_file_bea = os.path.join(output_dir, f"{project_name}_boundary_exception.json")
    # boundary_exception_info = analyze_boundary_and_exception(project_path)
    # save_json(boundary_exception_info, output_file_bea)
    # print(f"Boundary condition and exception handling analysis completed, results saved to {output_file_bea}")

    # Merge all analysis results
    # combined_results = {
    #     "data_flow_graph": project_info_dfg,
    #     "dependencies": project_info_dep,
    #     "indirect_dependencies": {k: list(v) for k, v in analyzer.dependencies.items()},
    #     "boundary_and_exception": boundary_exception_info
    # }

def main():
    parser = argparse.ArgumentParser(description="Analyze Java project and generate static analysis results.")
    parser.add_argument("project_path", help="Path to the Java project")
    parser.add_argument("--output_dir", default="../results/static_analysis", help="Directory to save output files")
    args = parser.parse_args()

    project_name = os.path.basename(args.project_path)
    output_dir = os.path.join(args.output_dir, project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # File analysis (including data flow graph)
    output_file_dfg = os.path.join(output_dir, f"{project_name}_dfg.json")
    project_info_dfg = analyze_project(args.project_path)
    save_json(project_info_dfg, output_file_dfg)
    print(f"Data flow graph analysis completed, results saved to {output_file_dfg}")
    
    # Dependency analysis
    output_file_dep = os.path.join(output_dir, f"{project_name}_dependency.json")
    project_info_dep = analyze_java_project(args.project_path)
    save_json(project_info_dep, output_file_dep)
    print(f"Dependency analysis completed, results saved to {output_file_dep}")
    
    # Indirect dependency analysis
    output_file_idc = os.path.join(output_dir, f"{project_name}_IDC.json")
    analyzer = EnhancedJavaDependencyAnalyzer(args.project_path)
    analyzer.analyze()
    analyzer.save_to_json(output_file_idc)
    print(f"Indirect dependency analysis completed, results saved to {output_file_idc}")

    # Merge all analysis results
    combined_results = {
        "data_flow_graph": project_info_dfg,
        "dependencies": project_info_dep,
        "indirect_dependencies": {k: list(v) for k, v in analyzer.dependencies.items()}
    }
    output_file_combined = os.path.join(output_dir, f"{project_name}_combined_analysis.json")
    save_json(combined_results, output_file_combined)
    print(f"All analysis results have been merged and saved to {output_file_combined}")

if __name__ == "__main__":
    main()