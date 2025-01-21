<!--
 * @Date: 2024-11-30 13:20:04
 * @LastEditors: muzhancun muzhancun@126.com
 * @LastEditTime: 2025-01-21 21:00:05
 * @FilePath: /CausalEnvGeneration/README.md
-->

# Integrating Causal Knowledge for Automated RL Environment Generation And Evaluation in Open World

This is final project for the 2024 Fall course "Knowledge Representation and Learning" at PKU. The project is to integrate causal knowledge for automated RL environment generation and evaluation in open world. The project is based on the [MineStudio](https://github.com/CraftJarvis/MineStudio)

## Generating Environments Using Causal Knowledge

This section describes our approach for synthesizing diverse Minecraft environments tailored for multi-task reinforcement learning and evaluation. The method integrates causal graphs with large language models to produce task-specific environment configurations. By leveraging causal dependencies and structured reasoning, we ensure the generated environments are both coherent and sufficiently varied.

The generation process employs GPT-4o, which synthesizes environment configurations in JSON format based on the causal structure and past configurations. The LLM is prompted to ensure adherence to logical constraints while introducing controlled variability. A detailed prompt can be found in `minestudio/simulator/callbacks/custom/generate_configs.py`.

If you want to generate your configs tailed for a specific task, you can modify the prompt in this file and run the following command:
```bash
python minestudio/simulator/callbacks/custom/generate_configs.py
```

We have provided a set of generated configurations for mining blocks in `minestudio/simulator/callbacks/custom/configs`.

You can try to load the configurations and run the simulation using the following command:
```bash
python minestudio/simulator/callbacks/custom/mine.py
```
This file also includes the process of loading SAM-2 and Molmo models.
Please comment out the corresponding code if you do not have the models to see a vanilla visualization.