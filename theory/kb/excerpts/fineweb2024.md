---
paper_key: fineweb2024
title: "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"
authors: Penedo, Kydlíček, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf
year: 2024
venue: NeurIPS Datasets & Benchmarks
arxiv: 2406.17557
local_pdf: theory/sources/papers/fineweb2024.pdf
type: excerpts
note: Verbatim quotes from v2 (31 Oct 2024). Methodologically the dominant 2024 web-scale text recipe — proxy-LM ablation for filter selection, individual-snapshot MinHash dedup, custom heuristic filters from histogram inspection, and a Llama-3-70B-derived educational classifier (FineWeb-Edu).
---

# Excerpts — Penedo et al. 2024, "FineWeb"

## Abstract — what FineWeb is {#abstract}

> The performance of a large language model (LLM) depends heavily on the quality and size of its pretraining dataset. … In this work, we introduce **FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots that produces better-performing LLMs than other open pretraining datasets.** To advance the understanding of how best to curate high-quality pretraining datasets, we carefully document and ablate all of the design choices used in FineWeb, including in-depth investigations of deduplication and filtering strategies. In addition, we introduce **FineWeb-Edu, a 1.3-trillion token collection of educational text filtered from FineWeb.** LLMs pretrained on FineWeb-Edu exhibit dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and ARC.

## §3.1 Ablation methodology — proxy LM validation {#sec-3-1-proxy}

> Our design of FineWeb is primarily empirical: we performed a series of "data ablation" experiments to test different methods at each stage of the pipeline. … We compare pipeline design choices at each stage by training data ablation models that are identical apart from the data they were trained on (same number of parameters, architecture hyper-parameters, and trained on an equal number of randomly sampled tokens from each version of the data). We then evaluated them on the same set of downstream task benchmark datasets … To minimize the impact of random data subset selection on evaluation scores, we trained two models for each dataset version, each using a different but equal-sized random subset of the full data and a different initialization seed, and then compared average scores.

> All training was performed using the `nanotron` library. Data ablation models all had 1.71B parameters (including embeddings), used the Llama architecture with a sequence length of 2048, a global batch size of ~2 million tokens, and the GPT2 tokenizer. Within a given experiment, all models were trained on the same amount of data for the same number of steps. Filtering ablations were trained on ~28 billion tokens (roughly the Chinchilla-optimal training size for this model size), while some deduplication ablations and runs to confirm cumulative relative performance improvements after each step of filtering were conducted on 350 billion tokens.

## §3.2 Trafilatura over WET {#sec-3-2-text}

> Common Crawl data is available in two different formats: WARC and WET. … While WET files are commonly used as a starting point for dataset creation … we found that WET files retained too much boilerplate and menu text. We therefore experimented with extracting the text content from the WARC files using the open source **trafilatura** library, which from visual inspection of the results provided good quality extraction when compared to other available libraries (less boilerplate and menu text). Custom text extraction is relatively costly, but its effects are felt on model performance.

## §3.3 Base filter pipeline {#sec-3-3-base}

> As a starting point to our filtering, we applied a basic filtering pipeline using part of the setup from RefinedWeb. Concretely, we applied URL filtering using a blocklist to remove adult content, applied a fastText language classifier to keep only English text with a score >= 0.65, and applied quality and repetition filters from MassiveText, using the original thresholds. After applying this filtering to all of the WARC-based text extracted from the 96 snapshots available at the time of writing, we obtained roughly 36 trillion tokens of data when tokenized with the GPT-2 tokenizer.

## §3.4 MinHash parameters {#sec-3-4-minhash}

> Following RefinedWeb, we experimented with MinHash, a fuzzy hash-based deduplication technique that scales efficiently to many CPU nodes … We chose to collect each document's 5-grams, obtained using an English word tokenizer, and computed MinHashes using 112 hash functions in total, split into **14 buckets of 8 hashes each — targeting documents that are at least 75% similar.** Documents with the same 8 MinHashes in any bucket are considered duplicates of each other. We then perform a transitive clustering step where documents A, B and C will be in the same duplicate cluster if A and C are duplicates and B and C are duplicates, even if A and B do not have 8 matching MinHashes in any bucket with each other. One (randomly chosen) document is kept per duplicate cluster while the remaining duplicates are removed.

## §3.4 Global vs individual MinHash — counterintuitive {#sec-3-4-global}

> Our first approach was to apply MinHash deduplication globally to the entire dataset (all 96 snapshots). … When applied to the oldest snapshots, this process removed as much as 90% of the original base filtered data … This challenged our initial assumption that global deduplication would inevitably result in higher benchmark scores. … Results … show that, for this older crawl *taken in isolation*, the data from it that was kept (10% of the original data) was actually *of worse quality* than the 90% of data that was removed.

> We therefore tried an alternative approach: individually deduplicating each snapshot (independently from the others), using the same parameters as before. **This resulted in 20 trillion tokens of data.**

## §3.6 Custom heuristic filters from histogram inspection {#sec-3-6-custom}

> Past work has mainly developed heuristic filters through data inspection. In this work we devised a more systematic process for designing heuristic filters and tuning their thresholds. We started by collecting over **50 high-level statistics** ranging from document-level metrics (e.g. number of lines, avg. line/word length, etc) to inter-document repetition metrics (inspired by MassiveText) on both a high- and low-quality web dataset. Specifically, we used the individually and globally deduplicated versions of the 2013-48 snapshot (previously mentioned in Section 3.4) as our "high-quality" and "low-quality" datasets respectively. We then identified metrics for which the distribution of values differed significantly across the two datasets, inspected the histograms of the two distributions, and empirically chose thresholds that would target sections of the histogram where the lower quality dataset frequency was higher than on the corresponding higher quality dataset section.

> Out of all those runs, we identified **three filters** … the chosen filters remove documents where the *fraction of lines ending with punctuation is <= 0.12* (10.14% of tokens removed vs. 30% from the original C4 terminal punctuation filter), where the *fraction of characters in duplicated lines is >= 0.1* (12.47% of tokens removed; the original MassiveText threshold for this ratio is >= 0.2), and/or where the *fraction of lines shorter than 30 characters is >= 0.67* (3.73% of tokens removed). When applying the three filters together, **~22% of tokens were removed and the aggregate score increased by about 1%** in the 28B token ablations.

## §3.7 Final 15T pipeline {#sec-3-7-final}

> Combining the decisions made in the previous sections and applying the resulting pipeline to 96 Common Crawl snapshots produces the **15T-token FineWeb dataset.** Specifically, we extract text from WARC files (Section 3.2), apply base filtering (Section 3.3), perform individual per-crawl MinHash deduplication (Section 3.4), apply a selection of C4 filters (Section 3.5), and finally apply custom filters (Section 3.6).

## §4 FineWeb-Edu — distill-then-classify {#sec-4-edu}

> An interesting approach has recently emerged for filtering LLM training datasets: using synthetic data to develop classifiers for identifying educational content. … We applied this technique to FineWeb by filtering it with an educational quality classifier developed from synthetic annotations generated by Llama-3-70B-Instruct. The resulting dataset, **FineWeb-Edu, contains 1.3 trillion tokens.** FineWeb-Edu is specifically optimized for educational content and outperforms all openly accessible web-based datasets on a number of reasoning- and knowledge-intensive benchmarks such as MMLU, ARC, and OpenBookQA by a significant margin.

> To build the synthetic annotations, we use Llama-3-70B-Instruct to score 460,000 randomly sampled webpages from the FineWeb CC-MAIN-2024-10 snapshot for their educational quality on a scale from 0 to 5. … To scale our filtering to the entirety of FineWeb, we trained a linear regression model on top of the *Snowflake-arctic-embed-m* embedding model. We fine-tuned this linear regressor on 410,000 of our Llama 3 synthetic annotations for 20 epochs with a learning rate of 3e-4 (while keeping the embedding and encoder layers frozen). … With a threshold of 3, the model achieved an F1 score of 82% on the validation set.

> Applying the classifier to the 15 trillion tokens of FineWeb required **6,000 H100 GPU hours**.
