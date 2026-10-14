# ChromGraphFM: project explainer

This document explains the full research project in simple language. It uses the ASD-STE100 style (Simplified Technical English):
- Each sentence is short.
- Each sentence gives one idea.
- Sentences use the active voice.
- Each term has one meaning only.

For the binding rules, read [`preregistration.md`](preregistration.md). For a more technical summary, read [`study_overview.md`](study_overview.md).

---

## 1. Words that this document uses

| Term | Meaning in this document |
|---|---|
| DNA sequence | The letters A, C, G and T of the genome. |
| Window | One piece of DNA, 5,000 letters (5 kb) long. |
| Sample | 128 windows in a row. One sample is 640 kb of DNA. |
| Contact | Two windows that touch in 3D space inside the cell nucleus. |
| Hi-C | A laboratory method. It measures how often two windows touch. |
| Contact strength | How often two windows touch, after the distance effect is removed. |
| Graph | A set of windows (nodes) and the contacts between them (edges). |
| Model | The computer program that learns from the data. Our model is ChromGraphFM. |
| Pretraining | The stage where the model learns from DNA and Hi-C. No task labels are used. |
| Arm | One version of the model. Each arm gets a different 3D signal. |
| Seed | A number that sets the random start of one training run. |
| Validation set | Chromosomes that we use to make decisions. |
| Test set | Chromosomes that we keep closed until the end. |
| Pearson r | A score from −1 to 1. A higher score shows a better match. |

---

## 2. The research problem

The genome is a long line of DNA. Inside the cell, this line folds into a 3D shape. Folding brings distant parts of the DNA close together.

A gene can be controlled by DNA that is far away on the line. This control happens because the folding puts the two parts in contact.

Most DNA models read the genome only as a line. They do not know which parts touch in 3D.

Hi-C measures which parts touch. Other models use Hi-C in different ways:
- Some models predict Hi-C from DNA.
- Some models need Hi-C as input every time they run.
- Some models study the Hi-C map alone.

**Our question:** Can Hi-C teach a DNA model during training, and then be removed? Is the model better after this training?

---

## 3. The data

We use two types of data:
1. **DNA:** the human reference genome hg38.
2. **Hi-C:** contacts from one cell line, GM12878. GM12878 is a type of human blood cell. Its Hi-C map has 72 experiments, the deepest map that is available to the public.

We do not use other cell lines yet. K562 and IMR90 are in the plan, but we did not download them.

### How we prepare the contacts

Two windows that are near each other on the line touch very often. This effect comes only from distance.

We remove this distance effect. We divide each contact by the average contact at that distance. The result is the **contact strength**. It shows which contacts are stronger than distance alone predicts.

We found and fixed an error in this step (commit 2b26d56). The old method ignored pairs with no contacts. That error made distant contacts look weaker than they are.

### How we split the data

We split the data by chromosome. This prevents data leakage, because nearby DNA is similar.

| Set | Chromosomes | What we use it for |
|---|---|---|
| Training | 17 other chromosomes | The model learns from these. |
| Validation | chr10, chr11 | We make all decisions on these. |
| Test | chr3, chr13, chr17 | We keep these closed. We open them one time, at the end. |

---

## 4. The model

The model reads one sample: 128 windows in a row.

**Step 1: Read each window.** The model reads the DNA letters of each window. It makes one summary (a list of 256 numbers) for each window. In this step, each window is alone.

**Step 2: Make the graph.** The model connects each window to:
- its 3 nearest neighbours on each side;
- its 16 strongest distant contacts from Hi-C.

The model hides 15% of the distant contacts. It uses these hidden contacts only as questions, to test itself.

**Step 3: Share information.** Windows send information to each other along the graph. A strong contact sends more information. This step repeats 4 times.

**Step 4: Give the output.** The model gives a final summary for each window. This summary is the "representation".

The model has 7.29 million parameters.

---

## 5. Pretraining: how the model learns

The model learns with three tasks at the same time. No person labels the data.

1. **Fill the gap.** The model hides 15% of the windows. It must rebuild them from their neighbours.
2. **Find the partner.** For each hidden contact, the model must find the true partner window. It must not choose other windows at the same distance.
3. **Predict the strength.** The model must predict the strength of each hidden contact.
In addition, the model sometimes trains with all distant contacts removed (30% of samples). This is **structure dropout**. It teaches the model to work without Hi-C.

Note: a fourth task, **structure distillation**, was tested in the v2 recipe. The model had to copy what it knows when Hi-C is present. Seed 0 chose the v1 recipe. The v1 recipe does not use distillation (`lambda_distill: 0.0`).

Each training run has 20,000 steps. One run takes about 7.4 hours on one GPU (NVIDIA L40S).

---

## 6. What is new (novelty)

1. **Hi-C controls the attention.** Hi-C decides which distant windows share information, and how much.
2. **Hi-C is needed only for training.** Structure dropout teaches the model to work without Hi-C. After training, the model needs only DNA. This is important, because most cell types have no Hi-C map.
3. **The test is fair and fixed in advance.** We compare the model with six control versions. We wrote the decision rule before we saw the results.

---

## 7. How we test the model

### The test task

We remove all Hi-C. We give the model only DNA. The model must predict the contact strength between windows that are far apart (150 kb or more). We compare the prediction with the true Hi-C value. The score is Pearson r.

We read the score in two ways:
- **Head:** the model uses its own prediction part.
- **Probe:** we freeze the model. We train a new, small prediction part on top. All arms get the same probe. The probe is fair to arms that did not learn to predict contacts.

### The arms

All arms use the same model, the same data and the same number of steps. Only the 3D signal changes.

| Arm | 3D signal during training | Type |
|---|---|---|
| **Full** | True contacts and true strengths | Our model |
| **B3** | True contacts, strengths mixed at random | Main control |
| **B0** | No contacts. DNA only. | Main control |
| **B1** | Random contacts | Main control |
| B2 | True contacts, all strengths equal | Extra control |
| B4 | Hi-C added after the model, not inside it | Extra control |
| B5 | Hi-C used only in task 2 | Extra control |

### The seeds

Training results change a little with each random start. For this reason, we train each arm with different seeds.
- **Seed 0** was a first look. We used it only to choose the training recipe. It does not count as proof.
- **Seeds 1, 2 and 3** are the proof. They decide the claims.

### The decision rule

"X beats Y" is true only when both of these conditions are true:
1. The 95% confidence interval of the difference (X minus Y) is above 0. We calculate it with 10,000 bootstrap samples, averaged over seeds 1–3.
2. X is better than Y in at least 2 of the 3 seeds.

The script `scripts/decide.py` applies this rule. Nobody can change the rule now.

---

## 8. The claims

**Claim A (true strengths help):** Full beats B3 on the head score.

**Claim B (true contacts help):** All four of these are true on the probe score:
- Full beats B0.
- Full beats B1.
- B3 beats B0.
- B3 beats B1.

### What the paper says for each result

| Claim A | Claim B | Message of the paper |
|---|---|---|
| Pass | Pass | True 3D contacts make the DNA model better: which parts touch, and how strongly. |
| Fail | Pass | True 3D contacts make the DNA model better. The strength values add no clear benefit. |
| Pass | Fail | No main claim of improvement. We report Full > B3 as a secondary result. |
| Fail | Fail | A clear negative result: Hi-C during training did not make the DNA model better. |

All four results give a paper that we can publish. A negative result is honest science if the test is fair.

---

## 9. Results until now

These scores are on the validation set. Long-range Pearson r. They come from `results/<arm>_seed<N>/metrics.json` and `probe_metrics.json`. They are not the final decision.

**Claim A (head score):**

| Seed | Full | B3 | Full − B3 |
|---|---|---|---|
| 0 (first look) | 0.2066 | 0.1848 | +0.022 |
| 1 | 0.1876 | 0.1882 | −0.001 |
| 2 | 0.2061 | 0.1965 | +0.010 |
| 3 | waiting | waiting | — |

**Claim B (probe score):**

| Seed | Full | B3 | B0 | B1 |
|---|---|---|---|---|
| 0 (first look) | 0.161 | 0.193 | 0.188 | — |
| 1 | 0.197 | 0.140 | 0.164 | 0.145 |
| 2 | 0.171 | 0.197 | waiting | waiting |

**What this shows until now:**
- Full wins over B3 on seed 2, but not on seed 1. Seed 3 decides Claim A.
- On seed 1, Full beats B0 and B1. B3 does not beat them.
- The scores change much from seed to seed. For this reason, we need three seeds.

---

## 10. What we cannot say yet

The test compares the model only with its own control versions. Thus:
- We **can** say whether Hi-C helps this model.
- We **cannot** say that our model is better than other published models.
- We **cannot** say that the model is useful for other biology tasks.

To say those things, we need the extra tests in Section 11.

---

## 11. Extra tests (planned, after the decision)

These tests are not in the fixed rule. We do them only after the claims are decided. We label them "secondary" in the paper. We use the trained models as they are. We do not train again and we do not change the model.

### Three downstream tasks

The labels for these tasks are already listed in `data/manifest.json`, for GM12878.

1. **TAD boundary detection.** The DNA folds into domains (TADs). The task: is this window at the edge of a domain?
2. **A/B compartment prediction.** The DNA is in an active part (A) or an inactive part (B) of the nucleus. The task: is this window in A or in B?
3. **Insulation score prediction.** Some windows block contacts across them. The task: how strongly does this window block contacts?

### Comparison with other models

- **A public DNA language model** (for example, Nucleotide Transformer or HyenaDNA). We put its output through the same probe. This shows if our model is better than a general DNA model.
- **Akita or Orca.** These models are made to predict Hi-C from DNA. We show them only as a reference point.

### Another cell line (optional)

We can test the GM12878 model on K562 cells. This shows if the model works on a cell type that it did not see. The K562 map is smaller (6 experiments), and the paper must say this.

---

## 12. Timeline

All dates are in 2026. They are estimates. A server restart can delay them.

| Date | Step |
|---|---|
| Sep 19 – Oct 8 | Build the data. Fix errors. Do the seed-0 first look. |
| Oct 8 | Write and fix the decision rule. |
| Oct 9 – 13 | Choose the recipe (v1). Start seeds 1 and 2. |
| Oct 14, about 20:00 | Full and B3 seed 3 finish. |
| Oct 15, about 03:30 | B0 and B1 seed 3 finish. Run `decide.py`. Claims A and B are decided. |
| Oct 15, about 18:00 | Extra control arms (B2, B4, B5) finish. |
| Oct 15 – 16 | Open the test set one time. Record the test scores. |
| Oct 16 – 20 | Do the downstream tasks and the comparison (only if approved). |
| Oct 20 – 31 | Write the Results, Discussion and Conclusion from the measured scores. |
| November | Do a final literature check. Send the paper to a journal. |

---

## 13. Rules that keep the study honest

- Do not use the test set before the decision.
- Do not change the decision rule after you see results.
- Do not select the best seed. Use all seeds 1–3.
- Do not change the model or the losses to make a claim pass.
- If a run stops, continue it from its checkpoint. Do not start a new seed.
- Write every number from a file. Give the name of the file.
