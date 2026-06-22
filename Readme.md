### Dataset
We place all processed datasets used in this project under the `Dataset/` folder.
The original chart images are derived from **Chart-MRAG[1]** and **ChartQA-Pro[2]**

To run this project, you should follow the VAGEN README to install the required packages,
```
# Create a new conda environment
conda create -n chartwalker python=3.10 -y
conda activate chartwalker

# verl
git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

git clone https://github.com/downing777/MMGraph.git
cd VAGEN
bash scripts/install.sh
```

Then follow the requirements.txt to install the KG required packages.

To run the agent training scripts,

```
cd VAGEN
bash scripts/examples/masked_grpo/kg_nav/run_tmux.sh
bash scripts/examples/masked_grpo/kg_chartmrag/run_chartmrag_tmux.sh
```

run other scripts,

```
PYTHONPATH=. python XXX.py 
```
### Reference
[1] Bench-marking multimodal rag through a chart-based document question-answering generation framework. arXiv preprint arXiv:2502.14864, 2025

[2] Chartqapro: A more diverse and challenging benchmark for chart question answering. arXiv preprint arXiv:2504.05506, 2025