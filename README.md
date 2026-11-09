# Welcome to our README

The following programs are present:
- 0_train_models.py: trains the (decision tree) classification models of the basic diseases + Parkinson.
- 1_basic_mcp.py: shows a basic planner that plans the 4 diseases (diabetes, heart, breast_cancer, parkinson).
- 2_llm_mcp_basic.py: the basic triage experiment using an LLM as planner with only the (diabetes, heart, breast_cancer) and with reliable answer parsing, uncertainty thresholds and abstention logic. 
- 2_llm_mcp.py: the experiment with an extra disease (diabetes, heart, breast_cancer, parkinson) and with reliable answer parsing, uncertainty thresholds and abstention logic. 
- 2_llm_mcp_random.py: the baseline that randomly select each disease with a probabilty of 50%.

The confusion matrices for 2_llm_mcp.py are saved in the results folder. The console output is in normal_rollout.out. For 2_llm_mcp_random.py, they are in the results_random_rollout folder and in random_roullout.out. For 2_llm_mcp_basic.py, they exist on the basic branch on github.

The files can be run using: uv run "filename"


To run 2_llm_mcp.py on netherite, submit the job: "slurmjob.sh" using the command: sbatch slurmjob.sh
