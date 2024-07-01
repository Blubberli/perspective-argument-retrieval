# Readme Shared Task *Perspective Argument Retrieval*

This readme gives you all an overview of the perspective argument retrieval shared task (co-located with ArgMin 2024).
The shared task website can be found [here](https://blubberli.github.io/perspective-argument-retrieval.github.io/).
The overview paper can be found [here](https://arxiv.org/abs/2203.00000).

The goal of the shared task was to evaluate the ability of retrieval systems to retrieve relevant arguments given a query, 
when taking into account socio-cultural properties of the author, e.g. given the question 
"Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?" and the property "political orientation: left",
the system should retrieve arguments that are relevant to the query and that match the political orientation of the author.

The arguments for the shared task stem from [SmartVote](https://www.smartvote.ch/), a Swiss voting recommendation platform, where politicians and voters
fill out a questionnaire to get voting suggestions. The arguments are annotated with their stance regarding different political issues and are 
formulated by politicians to support their stance. As a consequence, the arguments are provided together with socio-cultural properties of the author (the politician who wrote the argument).

> This repository contains experimental software and is published to give additional background details on the respective publication.



## Retrieve the data

<b> License </b>
The data is provided for research purposes and is not intended for commercial use.
The data is licensed under the CC BY-NC 4.0 licence. By retrieving the data, you agree to the terms of this license.
The copyright of the data lies with SmartVote.

<b> Intended Use </b>
The motivation for this shared task is to foster research on the latent impact of socio-cultural characteristics in argumentation. 
As a motivation we want to facilitate improving retrieval systems to be more fair and inclusive, for example by
ensuring that minority groups are represented in the top-k retrieved arguments or that they are diversified across the ranking.

Another motivation is to reduce affective polarization by providing arguments that match a user in terms some socio-cultural properties, but
also provides arguments that are diverse in other aspects. 

<b> Potential Measurements to Counter Negative Impact </b>
The data can be used to infer the socio-cultural properties of the authors (profiling), therefor, to protect the privacy of the authors,
we anonymized the data. The data should not be used to infer the identity of the authors.
The data could also be used for vote manipulation, therefore, we ask you to use the data responsibly and not to use it for any malicious purposes.
We only share the data when receiving a request via e-mail, together with a statement of your intended use of the data and potential measurements to counter negative impact.

Please reach out via e-mail to:
- [Andreas Waldis](mailto:andreas.waldis@live.com)
- [Neele Falk](mailto:neele.falk@ims.uni-stuttgart.de)



## Instructions

The required packages are listed in the `requirements.txt` file. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

### Scenarios

We distinguish between three scenarios for this retrieval task. Using these different scenarios, we want to verify the
effect of using socio-cultural properties at various stages. To participate, please follow the corresponding
instructions:

1.) **Baseline**: This scenario focuses on text-only retrieval of relevant arguments given and evaluates the general
abilities of the retrieval system. **Note**: do not use any socio-cultural properties for the query or the
corpus.

**Example query**:  _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
**Example candidate**:  _The reduction of sugar in food should be pushed. Not every food needs additional sugar as a
supplement._

2.) **Explicit Perspectivism**: With this scenario, we focus on using explicitly mentioned socio-cultural properties
in the query and the corpus. **Note**: thus, you are allowed to integrate these properties for all queries and all
arguments in the corpus for retrieval.

**Example query**:

- **Text**: _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
- **Age**: 18-34

**Example candidate**:

- **Text**: _Reducing sugar in food should be pushed. Not every food needs additional sugar as a supplement._
- **Age**: 18-34

3.) **Implicit Perspectivism**: With this scenario, we test the ability of a retrieval system to account for latently
encoded socio-cultural properties within the argument. **Note**: you are only allowed to use these properties for the
query **_not_** for the corpus.

**Example query**:

- **Text**: _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
- **Age**: 18-34

**Example candidate**:  _The reduction of sugar in food should be pushed. Not every food needs additional sugar as a
supplement._

## Data

We have two election cycles, the first one in 2019 and the second one in 2023. The train and dev sets stem from the 2019 election, there is no overlap in issues between train, dev and test.
For each test set you will find the corpus to retrieve candidates from in the corresponding folder. The queries are located in the `baseline-queries` and `perspective-queries` folders.

### Test Data
We used three different test sets, the first covering the election 2019, the second covering the latest election and
the last test set is a sample of the second test set but from a reader perspective: we asked annotators to select arguments that
they perceive as relevant for them personally. In this ranking therefor, the goal is not to align based on the socio-cultural properties of the author, but based on the reader's properties.


### Socio-Demographic Properties

We describe the socio-cultural profile of an author using the properties given by smartvote.ch. This includes eight
personal properties: gender, age, residence, education, civil status, denomination, political attitude, and a list of
important political issues, covering: open foreign policy, liberal economic policy, restrictive financial policy, law &
order, restrictive migration policy, and expanded environmental protection.

### Dataformat

The data is provided in JSON format. First, the file corpus.jsonl, which consists of a collection of arguments and
the authors' socio-cultural profiles. Secondly, the queries without (baseline-queries) and with socio-cultural
properties (perspective-queries). Here are examples of what these JSON files will look like:

Example corpus entry `corpus.jsonl`:

    [{
    	"argument_id": "<argument_id>",
    	"text": "Eating is an individual decision. It doesn't need a nanny state.",
    	"target": "Are you in favor of the introduction of a tax on foods containing sugar …",
    	"stance": "CON",
    	"demographic_profile": {...}
    },…]

Example baseline query `baseline-queries/queries_train.jsonl`:

    [{
    	"query_id": "<query_id>",
    	"text": "Are you in favor of the introduction of a tax on foods containing sugar …",
    	"relevant_candidates": [23, 4623, 65, 321, ...]
    },…]

Example perspective query `perspective-queries/queries_train.jsonl`:

    [{
    	"query_id": "<query_id>",
    	"text": "Are you in favor of the introduction of a tax on foods containing sugar …",
    	"demographic_properties": {
    		"age": "18-34"
    	},
    	"relevant_candidates": [23, 4623, 65, 321, ...]
    },…]


### Evaluation

We evaluate the retrieval performance based on two core dimensions:

- **Relevance**: We will evaluate the relevance of the retrieved arguments to the given query. This quantifies the
  ability of the retrieval system to retrieve relevant arguments for a given question for scenario 1 or to retrieve
  relevant arguments for a given question and socio-cultural properties for scenarios 2 and 3.
    - ***nDCG***: Normalized Discounted Cumulative Gain (nDCG): this metric quantifies the quality of the ranking by
      putting more weight on the top-ranked arguments since it is more important to retrieve relevant arguments at
      lower ranks.
    - ***P@k***: Precision at k (P@k): this metric quantifies how many of the top-k retrieved arguments are relevant.
- **Diversity**: We will evaluate the fairness of the retrieval system by considering to what extent the ranking
  represents a diverse set of socio-cultural properties and whether minority groups are represented in the top-k
  retrieved arguments. Note that fairness for each query will be evaluated based on all socio-cultural properties that
  are not part of the query. The metrics will be averaged across all variables.
    - ***alpha-nDCG***: Alpha-nDCG: this metric works like nDCG but, on top of that, penalizes top-ranked items if they
      are not diverse. As a consequence, the metric rewards rankings that represent all relevant different socio-cultural
      properties at the top of the ranking.
    - ***rKL***: normalized discounted Kulback-Leibler divergence (rKL): this metric quantifies fairness independent of
      relevance. It measures whether the top-k retrieved arguments represent the minority groups of specific
      socio-cultural variables in the corpus.

To evaluate your systems, you must dump the predictions and then run the evaluation script on these predictions. 
`baseline.ipynb` shows these steps using `sentence-transformers` and `BM25` as baseline retrieval methods. 
It also shows you how to create the prediction file. 
With this `.jsonl` file, you must provide the corresponding best-matching candidates for each query as a JSON entry.
These entries must look as follows and include to keys `query_id` (id of the corresponding query) and `relevant candidates`, sorted list of the most relevant candidates.

```json
{
  "query_id":0,
  "relevant_candidates":[2019017914,201904055,201908061,201903763, ...]
}
```
After dumping the results, you can run the evaluation with the following command:
```bash
python scripts/evaluation.py --data <path_to_data_dir> \
  --predictions <path_to_predictions.jsonl> \
  --output_dir <path_to_store_results> \
  --diversity True --scenario <baseline or perspective>  --split <train or dev>
```

You can evaluate your predictions as often as you'd like to. You only need to upload the prediction file for the official evaluation run. We will then run the script on the results of the unseen test data. (Please find details under **Submission**)
We will have two evaluations, one for relevance and one for diversity. We will report all four metrics across 4 different k values in both cases. We focus on nDCG and
alpha-nDCG as the main metrics for ranking participants.

## Baseline

We provide two baselines for the retrieval task. The first baseline is a simple BM25-based retrieval system. It uses
the BM25 algorithm to retrieve the top-k arguments for a given query. The second baseline is a simple SBERT-based
retrieval system. It uses the pre-trained SBERT model from the sentence-transformers library to encode the queries and
the arguments. The retrieval is then performed by computing the cosine similarity between the query and the arguments.
The top-k arguments are then returned as the retrieval results. There is no training involved in this baseline system.
The predictions and the results of the baseline system for scenario 1 are stored in the `baseline` folder.


### Reference

```
@inproceedings{
    neele-etal-2024-perspective,
    title = "Overview of PerpectiveArg2024: The First Shared Task on Perspective Argument Retrieval",
    author = "Falk, Neele and Waldis, Andreas and Gurevych, Iryna",
    booktitle = "Proceedings of the 11th Workshop on Argument Mining",
    month = aug,
    year = "2024",
    address = "Bangkok",
    publisher = "Association for Computational Linguistics"
}
```

