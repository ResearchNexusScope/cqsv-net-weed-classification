# Plant/weed dataset research - browser-based survey (2026-11-23)

> Method: browsed the public internet with the connected `browse` MCP agent.
> Searches via DuckDuckGo HTML; every size/license/access claim below was then
> checked by opening the dataset's own page (GitHub, Zenodo, TensorFlow Datasets,
> publisher or dataset-catalog page). URLs are the ones actually visited.
> Nothing here is private-paper content and no dataset was downloaded in this step.

## 1. Why this survey was requested

Our current protocol uses Kaggle `aminelaatam/weed-classification` (`CornWeed`,
1,993 train / 809 test) as the source domain and Ninja maize/weed crops as the
OOD target. The corrected QNN removes the single-class collapse but OOD AUC
remains below chance. Two hypotheses are being tested in parallel:

1. The QNN receives no labelled target-domain adaptation data, unlike the
   classical mixed arm (see `docs/METHODOLOGY_V2.md` §4). Worker QHEAD-D02 is
   implementing matched labelled adaptation on the current data.
2. The current source/target pair may be a poor domain match. This document
   identifies better candidate datasets for a follow-up protocol.

## 2. Strong external evidence on domain shift

A 2026 multi-dataset benchmark (7 public weed-detection datasets, YOLO26)
reports exactly the failure mode we observe:

- In-domain mean mAP50:95 `0.603`; off-domain (unified one-class weed) `0.148`.
- "Multi-source training improves performance on several datasets, but does not
  fully eliminate domain shift"; the conclusion stresses "target-domain
  adaptation" as necessary for robust deployment.
- Source: `https://arxiv.org/html/2609.33991v1` (CC BY 4.0), Table 1 lists the
  seven datasets and their sizes below.

This independently corroborates our finding that memorising the source domain is
not enough, and that labelled target adaptation is the missing ingredient rather
than a quantum-vs-classical effect.

## 3. Candidate datasets (verified)

| Dataset | Size / task | Classes | License (verified) | Access | Fit for our maize/weed OOD protocol |
|---|---|---|---|---|---|
| **MaizeSet** | 3,330 images; real field, Swabi Pakistan; image-level | maize plants vs weeds | Not stated on repo - must confirm before use | `https://github.com/mshahabalam/MaizeSet` (9 zip files, direct download) | **Best task match**: same crop and binary crop/weed labels. Different geography gives a genuine domain shift. License gap is the main risk. |
| **ACRE Crop-Weed** | 1,000 RGB images; instance segmentation; 2046x1080 | maize + bean vs 4 weeds (ryegrass, mustard, matricaria, lamb's quarter) | **CC BY 4.0** (Zenodo record) | `https://zenodo.org/records/8102217` (1.2 GB zip, DOI 10.5281/zenodo.8102217) | **Strong**: same crop (maize), clean license, robot-acquired imagery, standard XML instance masks. Can collapse to crop-vs-weed crops. |
| **CornWeed** (full benchmark version) | 3,574 images; 281,725 instances; COCO; 640x480 + 1280x720 | maize vs weed | Check source; benchmark paper is CC BY 4.0 | Listed in `https://arxiv.org/html/2609.33991v1` Table 1; ref [8] Iqbal et al. 2023 | Likely a larger/annotated version of the family we already use; worth checking whether it supersedes our Kaggle copy. |
| **DeepWeeds** | 17,509 images, 256x256 RGB; image classification | 8 Australian weed species + negative class | **CC BY 4.0** data, Apache-2.0 code (confirmed on dataset catalog and publisher) | `https://github.com/AlexOlsen/DeepWeeds`, `https://www.tensorflow.org/datasets/catalog/deep_weeds` (469 MiB), HF mirror `Project-AgML/DeepWeeds_classification` | **Largest clean classification set**. Negative class can serve as "no target weed". Very different domain (Australian rangeland) - excellent as a hard OOD target, poor as a drop-in source. |
| **Weed25** | 14,035 images; ~500-550 per species; 25 species, monocot + dicot, growth stages | 25 weed species | Check publisher; not stated on GitHub repo | `https://github.com/yinandkang/Weed25-a-dataset-for-deep-learning` (folders `new7`, `Weed25-Dataset`); paper `https://pmc.ncbi.nlm.nih.gov/articles/PMC9748680/` | Species-level; needs a crop source to form a binary task. Good for a harder fine-grained OOD test. |
| **CottonWeedDet12** | 5,648 images, 9,370 boxes; natural field light | 12 cotton-field weed species | Check Zenodo record | `https://zenodo.org/records/7535814`; `https://weed-ai.sydney.edu.au/datasets/2c14915b-0827-4b65-9908-d2a6df0d48f3`; Roboflow mirror | Detection-oriented; different crop. Useful as an additional source in multi-source training. |
| **FoodCropsWeeds** | 1,176 images; greenhouse + open field | 8 weed + 6 crop species; annotations collapsed to weed/crop | Check Data in Brief paper | Ref [24] in `https://arxiv.org/html/2609.33991v1`; Data in Brief 31 (2020) 105833 | Binary crop/weed with two acquisition conditions; useful for controlled-vs-field shift. |
| **Weed-AI** | Aggregator; MS-COCO-derived format with standardised agricultural metadata | Many (includes Palmer Amaranth, CottonWeedDet12, DeepWeeds) | Per-dataset | `https://weed-ai.sydney.edu.au/datasets/` | Best single entry point for multi-source experiments; consistent format reduces preprocessing risk. |

## 4. Recommendation

For the *next* experiment, keep the current data and first test the labelled
adaptation fix (QHEAD-D02). It isolates the confound without changing the
feature distribution.

If a dataset change is desired afterwards, the strongest maize/weed upgrade is:

1. **Primary replacement: ACRE Crop-Weed** (CC BY 4.0, maize + 4 weeds, Zenodo).
   Same crop as our task, explicit license, instance masks for group-disjoint
   crop extraction, and a different acquisition setup from both Kaggle and Ninja.
2. **Second source / hard OOD: DeepWeeds** (CC BY 4.0, 17,509 images, TFDS/HF
   mirror). Use as a multi-source training partner or as a deliberately hard
   external test domain - not as a same-domain source.
3. **Direct same-task alternative: MaizeSet** (3,330 images, maize vs weeds in
   the field), **only after confirming its license** - the repository does not
   state one. The size and label match are excellent; the licensing gap is a
   hard blocker for inclusion until resolved.

## 5. Caveats

- Detection datasets need crop extraction before they can feed our image-level
  QNN feature loader; budget preprocessing and group-disjointness checks.
- Licenses change. Recheck the source page immediately before downloading;
  DeepWeeds in particular has conflicting secondary catalog notes (CC BY vs
  CC BY-SA), so the original repository and paper govern.
- A new dataset changes the split manifest and therefore every checksum and
  provenance hash. Do not mix it into the current result tables; start a new
  result tree with a new D-ID and pass the same verifier + Critic gates.
