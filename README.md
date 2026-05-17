# LLM Directness Experiment

A controlled experiment testing whether an LLM-based startup classifier derives its verdicts from the input features it is shown, or from facts memorized during pretraining.

## Motivation

As part of my work on [Bena, Bian, and Giannetti (2026), *Prompted to Start: How Generative AI is Transforming Entrepreneurship](https://ssrn.com/abstract=5749564)*, I built an LLM classification pipeline identifying AI-native startups from Crunchbase data. The pipeline classified 270k startups and is a core empirical input to the paper.

This repository is a separate strand of that work. Having built the LLM classifier, examining the reliability of our data motivates the following question: to what extent are the classifications  driven by the input features? or is the model recognizing company names from its pretraining corpus and inferring labels from memorized knowledge? 

## Defining LLM Directness

The concept of *directness* comes from [Asirvatham, Mokski, and Shleifer (2026), *GPT as a Measurement Tool](https://shleifer.scholars.harvard.edu/publication/gpt-measurement-tool)*. They define a LLM measurement as **direct** when it is "driven by the signal in the content itself, rather than by leakage, memorized facts, or correlated cues that allow the model to guess the label without substantively reading what the researcher intends it to read."

The paper identifies two failure modes: (1) *contamination*, where the model draws on memorized or post-period knowledge, and (2) *shortcut inference*, where correlated cues in the input allow the model to infer the label without reading the intended signal. This experiment tests for both.

## Experimental Design

We use a three-cell controlled design. All cells run through the same pipeline, same model snapshot, and same prompt body. The only experimentally varied factor is which input fields are populated:


| Cell                          | Company Name | Descriptions | Address | Keywords | Year Founded |
| ----------------------------- | ------------ | ------------ | ------- | -------- | ------------ |
| **Baseline**                  | real         | real         | real    | real     | real         |
| **Arm A** (feature ablation)  | real         | masked       | real    | masked   | masked       |
| **Arm B** (identity ablation) | anonymized   | masked       | real    | masked   | masked       |


- **Arm A** strips the substantive text features (descriptions, keywords) while keeping the real company name. If the classifier's verdicts change relative to baseline, the classifier is direct — the descriptions were doing the work.
- **Arm B** additionally replaces the company name with a deterministic anonymous token. Comparing Arm A to Arm B isolates whether the real name alone carries signal via pretraining memorization, particularly for well-known companies.

## Statistical Methods

Agreement between cells is measured using Cohen's kappa (chance-corrected agreement) rather than raw agreement rates, which can be misleadingly high for imbalanced classes. Paired hypothesis tests — McNemar's test for binary classification axes and Stuart-Maxwell's test for multi-class axes — assess whether observed disagreements are statistically significant. All metrics are stratified by a composite fame proxy that splits companies into quartiles from obscure to well-known, testing whether leakage concentrates among firms the model is most likely to have memorized.

## Repository Overview

- `classify.py` — pipeline CLI for running the classification experiment across all three cells
- `prompts/` — controlled prompt files, byte-identical except for the experimentally varied input format block
- `scripts/` — analysis scripts (agreement metrics, fame proxy computation, interactive dashboard)
- `src/` — pipeline internals (batch processing, name anonymization, state management)
- `tests/` — automated tests enforcing experimental controls (e.g., prompt consistency across arms)

## References

- Bena, J., Bian, B., and Giannetti, M. (2026). "Prompted to Start: How Generative AI is Transforming Entrepreneurship." Available at [SSRN](https://ssrn.com/abstract=5749564).
- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). "GPT as a Measurement Tool." NBER Working Paper 34834. Available at [Harvard Scholar](https://shleifer.scholars.harvard.edu/publication/gpt-measurement-tool).

## License

MIT.