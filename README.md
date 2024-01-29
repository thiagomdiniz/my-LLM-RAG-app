# Application Development with LLM

I created this repository after participating in the [LinuxTips](https://www.linuxtips.io) workshop "Simplifying Application Development with LLM" ([see my digital certificate](https://www.credential.net/2d5d68eb-69fb-4fa7-acce-acab2d969da3)).  
This repository contains the [code created during the workshop](workshop-code) and also [other codes I created after the workshop](my-custom-code), to fix the content and test more features.

[Here are some notes I took while watching content about LLM (pt_BR)](AI_for_Devs-pt_BR.md).

I used the [Visual Studio Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in this project to have the Python development environment inside a Docker container.

## Useful commands

- `conda --version`
- `conda init` #restart the shell after that
- `#conda create --prefix ./env python=3.11` #in devcontainer you can use the base environment
- `conda env list`
- `#conda activate ./env` #restart the shell if necessary
- `#conda deactivate` #exit/deactivate the current environment
- `pip install -r my-custom-code/requirements.txt`
- `pip show langchain`
- Install and run Jupyter:
  - `conda install -c conda-forge jupyterlab`
  - `jupyter lab` #to use autocomplete in Jupyter, just press the TAB key

## Useful links

- APIs
  - https://platform.openai.com/
  - https://www.pinecone.io/
- Docs
  - https://platform.openai.com/docs/overview
  - https://python.langchain.com/docs/get_started/introduction
  - https://python.langchain.com/docs/use_cases/question_answering/
  - https://docs.pinecone.io/
  - https://docs.conda.io/en/latest/
  - https://jupyter.org/
- Original workshop repository
  - https://github.com/infoslack/linuxtips-llm

