import json
import os
import re
import argparse

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def simplify_type(type_str):
    if isinstance(type_str, str):
        if "BasicType" in type_str:
            return re.search(r"name=([\w]+)", type_str).group(1)
        elif "ReferenceType" in type_str:
            return re.search(r"name=([\w]+)", type_str).group(1)
    return type_str

def parse_complex_type(type_str):
    if isinstance(type_str, str):
        return type_str
    elif isinstance(type_str, dict):
        return type_str.get('name', str(type_str))
    elif hasattr(type_str, 'name'):
        if type_str.name == 'List' and hasattr(type_str, 'arguments') and type_str.arguments:
            inner_type = parse_complex_type(type_str.arguments[0].type)
            return f"List<{inner_type}>"
        elif hasattr(type_str, 'arguments') and type_str.arguments:
            args = ', '.join(parse_complex_type(arg.type) for arg in type_str.arguments)
            return f"{type_str.name}<{args}>"
        else:
            return type_str.name
    else:
        return str(type_str)

def get_methods(class_info):
    return set(method['name'] for method in class_info.get('methods', []))

def get_imports_from_methods(methods):
    imports = set()
    for method in methods:
        imports.add(parse_complex_type(method['return_type']))
        for param in method.get('parameters', []):
            imports.add(parse_complex_type(param))
    return imports

def get_imports_from_fields(fields):
    return set(parse_complex_type(field['type']) for field in fields)

def summarize_data_flow(class_info):
    summary = {
        "field_initializations": [],
        "method_flows": {},
        "overall_flow": [],
        "boundary_conditions": [],
        "exception_handling": []
    }

    for field in class_info.get("fields", []):
        if "initializer" in field:
            summary["field_initializations"].append(f"{field['name']} = {field['initializer']}")

    data_flow_graph = class_info.get("data_flow_graph", {})
    for method, flow in data_flow_graph.items():
        method_summary = []
        for step in flow:
            if step["type"] == "condition":
                condition = step.get('to', '')
                method_summary.append(f"Check: {condition}")
                summary["boundary_conditions"].append(f"{method}: {condition}")
            elif step["type"] == "assignment":
                method_summary.append(f"Assign: {step.get('details', '')}")
            elif step["type"] == "throw":
                exception = step.get('from', '')
                method_summary.append(f"Throw: {exception}")
                summary["exception_handling"].append(f"{method}: Throws {exception}")
            elif step["type"] == "return":
                method_summary.append(f"Return: {step.get('details', '')}")
        summary["method_flows"][method] = method_summary

    for field_init in summary["field_initializations"]:
        summary["overall_flow"].append(f"Initialize: {field_init}")
    for method, flow in summary["method_flows"].items():
        summary["overall_flow"].append(f"Method: {method}")
        summary["overall_flow"].extend(f"  {step}" for step in flow)

    return summary

def generate_prompt(class_info, package, dependencies_info, indirect_dependencies_info):
    class_name = class_info['name']
    
    superclass = class_info.get('extends')
    implemented_interfaces = class_info.get('implements', [])
    
    fields = [
        {
            "name": field["name"],
            "type": parse_complex_type(field["type"]),
            "modifiers": field["modifiers"],
            "initializer": field.get("initializer", "None"),
            "visibility": field.get("visibility", "package-private"),
            "is_final": "final" in field["modifiers"]
        }
        for field in class_info.get('fields', [])
    ]
    
    constructors = [
        {
            "name": constructor["name"],
            "parameters": [parse_complex_type(param) for param in constructor.get("parameters", [])],
            "modifiers": constructor["modifiers"],
            "throws": [parse_complex_type(exception) for exception in constructor.get("throws", [])],
            "body": constructor.get("body", "Not available")
        }
        for constructor in class_info.get('constructors', [])
    ]
    
    inherited_methods = set()
    if superclass:
        superclass_info = dependencies_info['testable_units'].get(superclass, {})
        inherited_methods.update(get_methods(superclass_info))
    for interface in implemented_interfaces:
        interface_info = dependencies_info['testable_units'].get(interface, {})
        inherited_methods.update(get_methods(interface_info))
    
    methods = [
        {
            "name": method["name"],
            "return_type": parse_complex_type(method["return_type"]),
            "parameters": [parse_complex_type(param) for param in method.get("parameters", [])],
            "modifiers": method["modifiers"],
            "throws": [parse_complex_type(exception) for exception in method.get("throws", [])],
            "body": method.get("body", "Not available"),
            "is_override": method["name"] in inherited_methods
        }
        for method in class_info.get('methods', [])
    ]

    data_flow_summary = summarize_data_flow(class_info)
    
    direct_deps = dependencies_info.get('dependencies', [])
    indirect_deps = indirect_dependencies_info.get(f"{package}.{class_name}", [])

    imports = set()
    imports.add(f"{package}.{class_name}")
    imports.update(implemented_interfaces)
    if superclass:
        imports.add(superclass)
    imports.update(get_imports_from_methods(methods))
    imports.update(get_imports_from_fields(fields))
    imports.update(class_info.get('imports', []))

    # Remove basic types from imports
    basic_types = {"void", "boolean", "int", "long", "float", "double", "char", "byte", "short"}
    imports = {imp for imp in imports if imp not in basic_types}

    is_generic = '<' in class_name or any('<' in str(field['type']) for field in fields) or any('<' in str(method['return_type']) for method in methods)

    prompt = f"""
===============================
JAVA CLASS UNIT TEST GENERATION
===============================

Class: {class_name}
Package: {package}

-----------
1. STRUCTURE
-----------
Superclass: {superclass if superclass else 'None'}
Implemented Interfaces: {', '.join(implemented_interfaces) if implemented_interfaces else 'None'}

Fields:
{json.dumps(fields, indent=4)}

Constructors:
{json.dumps(constructors, indent=4)}

Methods:
{json.dumps(methods, indent=4)}

--------------------
2. DATA FLOW SUMMARY
--------------------
Field Initializations:
{json.dumps(data_flow_summary['field_initializations'], indent=4)}

Method Flows:
{json.dumps(data_flow_summary['method_flows'], indent=4)}

Overall Flow:
{json.dumps(data_flow_summary['overall_flow'], indent=4)}

Boundary Conditions:
{json.dumps(data_flow_summary['boundary_conditions'], indent=4)}

Exception Handling:
{json.dumps(data_flow_summary['exception_handling'], indent=4)}

-------------
3. DEPENDENCIES
-------------
Direct Dependencies:
{json.dumps(direct_deps, indent=4)}

Indirect Dependencies:
{json.dumps(indirect_deps, indent=4)}

Imports:
{json.dumps(list(imports), indent=4)}


"""
    
    return prompt


def process_project(json_file, output_dir):
    data = load_json(json_file)
    
    dfg_info = data['data_flow_graph']
    dependencies_info = data['dependencies']
    indirect_dependencies_info = data['indirect_dependencies']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path, file_info in dfg_info.items():
        for class_info in file_info.get('classes', []):
            class_name = class_info['name']
            package = next((info['package'] for info in dependencies_info['testable_units'].values() if info['class_name'] == class_name), "")

            prompt = generate_prompt(class_info, package, dependencies_info, indirect_dependencies_info)
            
            output_file = os.path.join(output_dir, f"{class_name}_test_prompt.txt")
            with open(output_file, 'w') as f:
                f.write(prompt)
            
            print(f"Generated test prompt for {package}.{class_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate test prompts from static analysis results.")
    parser.add_argument("json_file", help="Path to the combined analysis JSON file")
    parser.add_argument("--output_dir", default="test_prompts", help="Directory to save generated prompts")
    args = parser.parse_args()

    process_project(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()