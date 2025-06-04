# Requirement

Please run the following command line to install all required packages.

```
pip install -r ./requirements.txt
```

# File Organize

`first_layer_graph_gen.py`: This `.py` file contains code for first layer candidate generation module;
`second_layer_gpt_rerank.py`: This `.py` file contains code for second layer disease re-rank module;
`data`: This folder contains raw datasets;
`processed_data`: This folder contains processed data.


# Pipeline

Please replace the `openai_api_key` in `./first_layer_graph_gen.py` and `./second_layer_gpt_rerank.py` with your own openai api key. Run the following command to get the first layer predicted candidate diseases of `Symptom2Disease` dataset.

```
python ./first_layer_graph_gen.py
```

Run the folloing command to generate the second layer predicted candidate disease of `Symptom2Disease` dataset generate by gpt-4.
```
python ./second_layer_gpt_rerank.py
```
