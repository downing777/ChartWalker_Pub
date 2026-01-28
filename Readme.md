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