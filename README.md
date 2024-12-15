# Rethinking Chain-of-Thought from the Perspective of Self-Training

This is the official implementation of `Rethinking Chain-of-Thought from the Perspective of Self-Training`.

[//]: # (The paper is available at [arXiv]&#40;https://arxiv.org/abs/2205.11916&#41;.)
The paper is available at [arXiv].


<div align="center">
<img src="img/framework.png">
</div>

## Installation
Make sure you have Python>=3.8 installed on your machine.
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

## Set your OpenAI API key
```
client = OpenAI(
    api_key="YOUR OPENAI API KEY",
    base_url="https://api.chatanywhere.tech/v1"
)
```

## Quick Start

### Zero-shot-CoT + Self-consistency + Our
```
python main.py --cot_trigger=1 --dataset=aqua --limit_dataset_size=30
```

### Zero-shot-CoT + Self-consistency
```
python comparison.py --method=zero_shot_cot --cot_trigger=1 --dataset=aqua --limit_dataset_size=30
```

### Zero-shot + Self-consistency
```
python comparison.py --method=zero_shot --cot_trigger=1 --dataset=aqua --limit_dataset_size=30
```

### A Demo Example
_**Question**_: \
A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower? Answer Choices: (A) 5 (√3 + 1)  (B) 6 (√3 + √2)  (C) 7 (√3 – 1)  (D) 8 (√3 – 2)  (E) None of these.


*****************************
_**Setting:**_
1) Maximum number of iteration rounds = 3
2) The number of self-consistency = 3
3) Semantic entropy threshold = 0

_**Note:** This implies that in the current CoT iteration, all three predictions (from three self-consistency samples) must be consistent to terminate the iteration and avoid proceeding to the next round. If a new iteration begins, the three reasoning processes are updated based on the results from the previous round._

*****************************

_**Filtered Predictions**_:\
pre_first_list:  ['C', 'A', 'B'] \
pre_two_list:  ['E', 'A', 'A'] \
pre_three_list:  ['A', 'A', 'A'] 

**Last_pred** : A \
**Ground Truth** : A





## Citation
```
Upcoming
```
