# Structural Causal Models
This repository implements basic capabilities tied to Structural Causal Models.


## Repository Structure

```plaintext
.
├── cli/
│   ├── __init__.py               
│   ├── sample_cli.py
│   ├── scm_cli.py
├── config/
├── ├── global_variables.json     
├── examples/                
├── graphs/                   
│   ├── __init__.py
│   ├── generator.py
│   ├── utils.py   
├── scm/
│   ├── __init__.py
│   ├── base.py
│   ├── counterfactuals.py
│   ├── functions.py
│   ├── noises.py
│   ├── sampler.py
├── scripts/
│   ├── generate_scm.py
│   ├── sample_data.py
├── tests/  
├── utils/
│   ├── __init__.py
│   ├── io.py
│   ├── plot.py       
└── README.md               

```

## Prerequisites
Python ≥ 3.8

## Usage

### 1. SCM Generation
Currently supports only fully observed SCMs with three modes of generation:

#### a) For a fixed graph topology:
Example: Chain graph with 4 intervenable nodes, (standard) Gaussian noises, and linear structural equations.
```plaintext
python scripts/generate_scm.py  --graph_type chain --n 4 --funct_type linear --noise_types "N(0,1)" --save --plot
```
* **Inputs**:
  * `--graph_type`: `chain`, `parallel`, `random`
  * `--n`: number of nodes excluding Y
  * `--func_type`: `linear`, `polynomial`
  * `--noise_types`: `N(0,1)`, `Exp(1.0)`, `Ber(0.5)`
* **Outputs**:
  * JSON file under: `outputs/scms/`
  * Plot under: `outputs/plots`       

#### b) From JSON files:
```plaintext
python scripts/sample_data.py --file_name "SCM_n4_chain_graph_linear_functions_N(0,1).json" --mode l1 --n_samples 1000 --save --plot  
```
* Files must be stored under `outputs/scms/`

#### c) From a set of structural equations:
Example:
```plaintext
from scm.base import SCM

functions = {
    "X1": "lambda _: 0",
    "X2": "lambda X1: 2 * X1"
}
noise = {
    "X1": ("gaussian", [0, 1]),
    "X2": ("gaussian", [0, 1])
}
scm = SCM.from_functions(functions, noise)
scm.visualize()
```
### 2. Sampling
#### a) Observational (L1)
```plaintext
python scripts/sample_data.py 
  --file_name "SCM_n4_chain_graph_linear_functions_N(0,1).json" 
  --mode l1 
  --n_samples 1000 
  --save --plot
```
* Output CSV: `outputs/data/SCM_n4_chain_graph_linear_functions_N(0,1).csv`
* Output plot (histogram): `outputs/plots/SCM_n4_chain_graph_linear_functions_N(0,1).png`
#### b) Interventional (L2)
##### (i) Atomic interventions:
From the console:
```plaintext
python scripts/sample_data.py 
  --file_name SCM_n4_chain_graph_linear_functions_N(0,1).json 
  --mode l2 
  --n_samples 1000 
  --do "(X1,5)" 
  --save --plot
```
From a JSON file:
```plaintext
python scripts/sample_data.py \
  --file_name SCM_n4_chain_graph_linear_functions_N(0,1).json 
  --mode l2 
  --n_samples 1000 
  --interventions_json examples/atomic_intervention.json 
  --save --plot
```
with `atomic_intervention.json` containing
```plaintext
{
  "X1": 5
}
```
for the intervention `do(X1=5)`.
##### (ii) Simultaneous and mixed-type interventions on multiple nodes:
From a JSON file:
```plaintext
python scripts/sample_data.py \
  --file_name SCM_n4_chain_graph_linear_functions_N(0,1).json 
  --mode l2 
  --n_samples 1000 
  --interventions_json examples/mixed_interventions.json 
  --save --plot
```
with `mixed_interventions.json` containing, for example:
```plaintext
{
  "X1": 5,
  "X2": {
    "function": "lambda X1: X1 + 3",
    "noise": ["gaussian", [0, 0.1]]
  }
}
```
