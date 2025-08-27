# Java Project Static Analysis and Unit Test Generation

This project provides a comprehensive suite of tools for static analysis of Java projects and automated unit test generation. It consists of several Python scripts that work together to analyze Java source code, extract relevant information, and generate unit test prompts for large language models (LLMs).

## Features

- Java source code parsing and analysis
- Data flow graph generation
- Dependency analysis (direct and indirect)
- Boundary condition and exception handling detection
- Unit test prompt generation for LLMs
- Automated unit test generation using LLMs

## Main Components

1. `main.py`: The entry point of the application. It orchestrates the entire analysis process.
2. `file_analyzer.py`: Analyzes individual Java files and extracts class information.
3. `dependency_analyzer.py`: Analyzes project dependencies and structure.
4. `indirect_dependency_analyzer.py`: Identifies indirect dependencies in the project.
5. `data_flow_analyzer.py`: Generates data flow graphs for Java methods.
6. `prompt_generator.py`: Creates prompts for LLMs based on the analysis results.
7. `generate_unit_test.py`: Uses LLMs to generate unit tests based on the prompts.

## Usage

1. Place your Java project in the `samples` directory.
2. Update the `project_name` variable in `main.py` to match your project's name.
3. Run `main.py` to perform the static analysis:

   ```
   python main.py path/to/project --output_dir ../results/
   ```
   example usage:
   ```
   python main.py /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/commons-cli-30 --output_dir ../results/buggy_version/commons-cli-30
   ```
4. After the analysis is complete, run `prompt_generator.py` to create prompts for the LLM:

   ```
   python prompt_generator.py path/to/_combined_analysis.json --output_dir ../results/project/prompts
   ```
   example usage:
   ```
   python prompt_generator.py /Users/ruiqidong/Desktop/unittest/results/buggy_version/commons-cli-30/commons-cli-30/commons-cli-30_combined_analysis.json --output_dir ../results/buggy_version/commons-cli-30/prompts
   ```
5. After the analysis is complete, run `prompt_generator.py` to create prompts for the LLM:

   ```
   python feedback.py --project path/to/project --prompt path/to/prompt_dir --class class_name --output . --package package_name --api-key gpt_api_key
   ``` 

example usage:
```
python feedback.py --project /Users/ruiqidong/Desktop/unittest/dataset/commons-cli --prompt ../results/commons-cli/prompts --class Parser --output . --package org.apache.commons.cli --api-key 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
```

```
python mcts_integrated_feedback.py --project /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/codec/commons-codec-16 --prompt /Users/ruiqidong/Desktop/unittest/results/buggy_version/codec/commons-codec-16/prompts --class Base32 --package org.apache.commons.codec.binary
 ```

python lambda_framework.py --project /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/cli/commons-cli-40 --prompt /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/cli/commons-cli-40/commons-cli-40/prompts --class TypeHandler --package org.apache.commons.cli --output /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/cli/commons-cli-40


python main.py /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/cli/commons-cli-35 --output_dir ../results/buggy_version/cli

python prompt_generator.py /Users/ruiqidong/Desktop/unittest/results/buggy_version/codec/commons-codec-18/commons-codec-18_combined_analysis.json --output_dir /Users/ruiqidong/Desktop/unittest/results/buggy_version/codec/commons-codec-18/prompts

python mcts_integrated_feedback.py --project /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/codec/commons-codec-12 --prompt /Users/ruiqidong/Desktop/unittest/results/buggy_version/codec/commons-codec-12/prompts --class Base64InputStream --package org.apache.commons.codec.binary


python run.py /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/lambda/cli/commons-cli-38 --output_dir /Users/ruiqidong/Desktop/unittest/dataset/buggy_version/lambda/cli/commons-cli-38 --class_name DefaultParser --package org.apache.commons.cli









python run.py /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-cli --output_dir /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-cli --class_name CSVParser --package org.apache.commons.cli


python main.py /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv --output_dir /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv

python prompt_generator.py /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv/commons-csv/commons-csv_combined_analysis.json --output_dir /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv/commons-csv/prompts

python lambda_framework.py --project /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-math/commons-math-transform --prompt /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-math/commons-math-transform/prompts --output /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-math/commons-math-transform/ --batch


python lambda_framework.py --project /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv --prompt /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv/commons-csv/prompts --output /Users/ruiqidong/Desktop/unittest/dataset/new_bugs/commons-csv --class CSVRecord --package org.apache.commons.csv










python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-2 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-2 \
   --class_name TestMapDeserialization --package com.fasterxml.jackson.databind.deser.TestMapDeserialization



python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-246 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-246 \
   --class_name StartResumeUsersPlaybackRequest --package com.wrapper.spotify.requests.data.player



python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-245 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-245 \
   --class_name ParameterSpec --package com.squareup.javapoet

python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-238 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-238 \
   --class_name JsonIgnoreFields --package com.json.ignore


python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-235 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-235 \
   --class_name AccountManager --package org.cash.count.service.impl

python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-234 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-234 \
   --class_name TransferService --package org.cash.count.service.impl

python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-232 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-232 \
   --class_name ByteArrays --package com.paritytrading.foundation




python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-202 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-202 \
   --class_name ClassGraph --package io.github.classgraph

python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-198 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-198 \
   --class_name DecryptionMaterialsRequest --package com.amazonaws.encryptionsdk.model



python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-194 \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-194 \
   --class_name PMDRunner --package com.gdssecurity.pmd



python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-192/src/server \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-192/src/server \
   --class_name RepairRunStatus --package io.cassandrareaper.resources.view


python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-191/src/server \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bears-benchmark/workspace/Bears-191/src/server \
   --class_name NodesStatus --package io.cassandrareaper.resources.view


python run.py \
   /Users/ruiqidong/Desktop/unittest/lambda/bugs-dot-jar/accumulo/core \
   --output_dir /Users/ruiqidong/Desktop/unittest/lambda/bugs-dot-jar/accumulo/core \
   --class_name Authorizations --package org.apache.accumulo.core.security



# Gitbug-Java
python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/assertj-vavr --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/assertj-vavr --class_name AbstractSeqAssert --package org.assertj.vavr.api --project_type maven


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/assertj-vavr --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/assertj-vavr --class_name AbstractTraversableAssert --package org.assertj.vavr.api --project_type maven


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/adr-j --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/adr-j --class_name CommandNew --package org.doble.commands --project_type gradle


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/ari-proxy --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/ari-proxy --class_name AriCommandResponseProcessing --package io.retel.ariproxy.boundary.commandsandresponses --project_type maven


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/aws-secretsmanager-jdbc --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/aws-secretsmanager-jdbc --class_name AWSSecretsManagerPostgreSQLDriver --package com.amazonaws.secretsmanager.sql --project_type maven



python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/beanshell --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/beanshell --class_name This --package bsh --project_type maven


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/cloudsimplus --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/cloudsimplus --class_name HostAbstract --package org.cloudsimplus.hosts --project_type maven



python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/ConfigMe --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/ConfigMe --class_name PropertyListBuilder --package ch.jalu.configme.configurationdata --project_type maven


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/crawler-commons --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/crawler-commons --class_name SimpleRobotRulesParser --package crawlercommons.robots --project_type maven

python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/database-engine --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/database-engine --class_name DBApp --package app


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/dataframe-ec --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/dataframe-ec --class_name StringValue --package io.github.vmzakharov.ecdataframe.dsl.value


python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/dotenv-java --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/dotenv-java --class_name DotenvParser --package io.github.cdimascio.dotenv.internal



python run.py /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/JSONata4Java --output_dir /Users/ruiqidong/Desktop/unittest/gitbug-java/projects/JSONata4Java --class_name OrderByOperator --package com.api.jsonata4java.expressions


6. You can skip steps from 3-5, this `auto_generate_tests.py` is the new entry point to run all above steps in a scale way without mannually input projects, run `auto_generate_tests.py` to generate unit tests (as labels)using the Claude with your key:

   ```
   python auto_generate_tests.py ../samples --output_dir ../output
   ```

## Output

The analysis results and generated artifacts will be saved in the following directory structure:

```
output/
├── static_analysis/
│   └── [project_name]/
│       ├── [project_name]_dfg.json
│       ├── [project_name]_dependency.json
│       ├── [project_name]_IDC.json
│       └── [project_name]_combined_analysis.json
├── test_prompts/
│   └── [project_name]/
│       └── [class_name]_test_prompt.txt
└── generated_tests/
    └── [project_name]/
        └── [class_name]_generated_test.java
```

## Requirements

- Python 3.7+
- javalang
- torch
- transformers

Install the required packages using:

```
pip install javalang torch transformers
```

## Notes

- This project uses the HuggingFace Transformers library to interact with LLMs for test generation.
- Make sure you have sufficient GPU resources when running the LLM for test generation.
- The quality of generated tests depends on the completeness of the static analysis and the capabilities of the chosen LLM.

## Future Improvements

- Add support for more Java frameworks and libraries
- Improve data flow analysis for more complex scenarios
- Enhance test generation prompts for better LLM output
- Implement result validation and test execution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
