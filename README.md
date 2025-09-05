# PRSA: Prompt Stealing Attacks against Real-World Prompt Services

This repository contains the code for our USENIX Security Symposium 2025 paper, "**PRSA: Prompt Stealing Attacks against Real-World Prompt Services**". 

## 📂 Assets

We provide the collected open-source datasets (**collected dataset**) used in the **Prompt Generation** phase, along with the necessary code and configuration files to reproduce our experiments.

For **real-world prompts and LLM applications**, we do **not** upload them to any public repository to avoid potential legal and ethical concerns. Instead, we include a **synthetic prompt dataset (demo dataset)** that enables researchers to run and validate our attack pipeline end to end. The synthetic prompts are derived from real-world prompts that we **legally acquired through paid access**, and were subsequently anonymized and modified with consent from the original developers. Therefore, these data do not raise copyright concerns.


### Datasets 

- `./collect_data`: Open-source prompts collected from public sources (collected dataset).
- `./demo_data`: Synthetic prompt data simulating real-world services for demonstration purposes (demo dataset).

### Main Scripts

- `1_prompt_attention_generation.py`:  
  Generates prompt attention for each category using the collected dataset during the **Prompt Generation** phase (see Algorithm 1 in the paper). 


- `1_prompt_attention_generation.sh`:  
  A shell script to run `1_prompt_attention_generation.py` across multiple categories in parallel.
  For convenience, we have provided the all generated results in the `./model` folder.

- `2_run_attack.py`:  
  The main script to execute prompt stealing attacks on test data. For each instance, it uses the `generate_prompt` function to construct a fine-grained stolen prompt, guided by prompt attention from `1_prompt_attention_generation.py`. The prompt is then refined in the **Prompt Pruning** phase using the `prompt_pruning_google` function. The script evaluates the stolen prompt against the target prompt using both prompt-level similarity and output-level functional consistency (semantic, syntactic, and structural similarity), along with LLM-based multi-dimensional evaluation.



## ⚙️ How to Use


### 🔧 Environment Setup

The experiments were conducted using CUDA 11.6 with NVIDIA A100, but any GPU compatible with CUDA 11.6 should work.


#### pip-based Installation (Recommended)
We provide pip-based option for environment setup. Choose one of the following:

```bash
conda create -n PRSA python=3.10
conda activate PRSA
pip install -r requirements.txt
```

#### LLM API Key
Export your OpenAI API key before running:

```bash
export OPENAI_API_KEY=YOUR_KEY
```

### 🛠️ External Dependencies

#### 🔹 spaCy Language Models

```bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```

#### 🔹 Word2Vec Embeddings
We use pre-trained Word2Vec embeddings from the Google News corpus to support semantic-aware prompt processing. Please download `GoogleNews-vectors-negative300.bin.gz` from the following source:
    🔗 [Google Drive Link] https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g

After downloading, place the file under the `./tool/` directory.

#### 🔹 Java (JDK 21)
Some components rely on Java for external tools. If you choose not to install Java system-wide, we’ve provided a prebuilt version of JDK 21.0.1 for your convenience: 🔗 [Google Drive Link] https://drive.google.com/file/d/16I3mKf4ZESa9IO39zdhoJUl06StJ2BkO/view?usp=sharing. Simply download the archive and unzip it into the ./tool/ directory so that the structure looks like:

```bash
./tool/jdk-21.0.1/...
```

#### 🔹 fkassim Package
Syntactic similarity evaluation requires the **fkassim** package.  
We provide a prebuilt archive for your convenience:  
🔗 [Google Drive Link] https://drive.google.com/file/d/1ab_JUtL-itRUxzbc1Y5EzL9cfvGGccGx/view?usp=sharing

After downloading, unzip the archive into the current project directory so that the structure looks like:

```bash
./fkassim/...
```

### ▶️ Run the Code

#### Step 1: Generate Prompt Attention
Adjust settings in `1_prompt_attention_generation.py` (e.g., `theme`, `attention_threshold`, etc.), then run:

```
python 1_prompt_attention_generation.py 
```

Alternatively, run the shell script to process all categories in parallel:

```bash
sh 1_prompt_attention_generation.sh 
```

The result will be located at ./model/, and the log will be located at ./log/. We also provide a pre-generated result in `./model/` to facilitate quick verification.

#### Step 2: Run Prompt Stealing Attacks
Adjust settings in `2_run_attack.py` (e.g., `beam_search`, `pre_pruning`, `alpha`, `m`, `n`, etc.), then run:

```
python 2_run_attack.py
```

The result will be located at ./result/



## 📝 Notes

- The experiments in our paper are conducted using LLM APIs, including GPT-4 and GPT-3.5. Users are expected to obtain their own API access. 

- As LLM APIs are continuously updated (e.g., `gpt-3.5-turbo` may fail to parse some inputs), we recommend using the latest stable versions, such as `gpt-3.5-turbo-1106`.

- In accordance with our Ethics section, we do **not** upload real-world purchased prompts or tested GPTs to any public repository. Verified academic researchers who are interested in obtaining examples of real-world prompts may contact us. We will reach out to the original developers of those prompts to obtain explicit approval before any sharing.


If you have any questions or require further information, please feel free to contact the authors. Thank you!


## Citation
Please kindly cite our work as follows for any purpose of usage.
```bibtex
@inproceedings{yang2025prsa,
  title={{PRSA}: Prompt Stealing Attacks against {Real-World} Prompt Services},
  author={Yang, Yong and Li, Changjiang and Li, Qingming and Ma, Oubo and Wang, Haoyu and Wang, Zonghui and Gao, Yandong and Chen, Wenzhi and Ji, Shouling},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={2283--2302},
  year={2025}
}



