# Plant-pretrained backbone research note (PLANT-D03)

**Status:** decision record / research note. No model was trained or evaluated for
this note, and it contains no numbers from project runs. Upstream figures quoted
from public sources are labelled as upstream, never as our measurements.

**Context:** the user flagged a generic/random hybrid head; Mother selected a
plant-pretrained backbone (SWARM_BOARD [82]). This note records the candidate
survey, the selection and its rationale, the exact loading recipe, provenance,
limitations, and the acceptance criteria the new-backbone experiment must meet.

## 1. Candidate comparison

| Candidate | License | Download size | Feature dim | Task domain | Assessment |
| --- | --- | --- | --- | --- | --- |
| **PlantNet-300K ResNet18** — `cpoisson/plantnet300k-resnet18` | Apache-2.0 | ~45 MB | 512-d (penultimate avgpool) | 1,081 plant species, leaf/flower photos | **SELECTED** — plant-domain, permissive, small, plain torchvision `state_dict` |
| PlantNet v2 / DINOv3 | license-gated (gated access, non-permissive terms) | not established here | not established here | plant species | Rejected: weights access is gated; cannot satisfy reproducible, offline-cached loading under a permissive license |
| BioCLIP ViT-B/16 | upstream repo license (see source) | ~600 MB | ViT-B/16 CLS token (768-d) | broad taxonomy / species names | Rejected: ~13x larger than 45 MB and a heavier ViT forward pass on a CPU-bounded pipeline |
| BioCLIP2 | upstream repo license (see source) | ~1.7 GB | ViT backbone | broad taxonomy / species names | Rejected: ~38x larger; impractical for the capped, CPU-simulated QML pipeline |
| WeedsGalore | see source repo | not established here | segmentation masks | weed instance segmentation | Rejected: segmentation task, not an image-level feature extractor for our crop-vs-weed head |
| AgriNet (arXiv:2207.03881) | paper (BY-NC-ND) | no public weights | VGG / Inception / Xception | 423 plant, disease, pest, weed classes | Rejected: paper-only; no public weight artifact to pin or cache |

"Not established here" means this note did not verify a value; it is not a
measurement. No table above reports a result from any of our runs.

## 2. Selection: PlantNet-300K ResNet18 (512-d)

Rationale:

1. **Domain match.** Pretrained on 306,146 Pl@ntNet-300K plant images spanning
   1,081 species — closer to our leaf-image weed/crop task than ImageNet's
   generic object classes.
2. **License.** Apache-2.0 permits reuse and redistribution in this research
   pipeline without gated access.
3. **Size.** A ~45 MB checkpoint is fast to cache and load and needs no GPU.
4. **Standard torchvision loading.** A plain
   `torchvision.models.resnet18(weights=None, num_classes=1081)` plus
   `load_state_dict` — no `trust_remote_code` or custom architecture path.
5. **Cached locally.** Weights resolve through `huggingface_hub` into the local
   HF cache and load offline after the first fetch; no per-run network dependency.

## 3. Exact loading code (frozen 512-d feature extractor)

```python
import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

REPO_ID = "cpoisson/plantnet300k-resnet18"
WEIGHTS_FILE = "plantnet_resnet18.pth"

# 1) Resolve weights into the local HF cache (offline after the first fetch).
weights_path = hf_hub_download(REPO_ID, WEIGHTS_FILE)

# 2) Build the exact torchvision architecture with its 1081-class head.
model = models.resnet18(weights=None, num_classes=1081)
model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
model.eval()

# 3) Freeze the backbone: a fixed feature extractor, not a trainable head.
for param in model.parameters():
    param.requires_grad_(False)

# 4) Drop the 1081-way classifier -> 512-d embedding (avgpool output).
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])


@torch.no_grad()
def embed(images: torch.Tensor) -> torch.Tensor:
    # images: (B, 3, 224, 224), ImageNet-normalized
    return feature_extractor(images).flatten(1)  # (B, 512)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 4. Provenance

| Field | Value |
| --- | --- |
| HF repo id | `cpoisson/plantnet300k-resnet18` |
| Weights file | `plantnet_resnet18.pth` |
| Architecture | torchvision ResNet18 (~11.7M parameters) |
| Classes | 1,081 plant species |
| Source dataset | Pl@ntNet-300K, 306,146 images (NeurIPS 2021 Datasets & Benchmarks; Zenodo DOI 10.5281/zenodo.5645731) |
| Input | 224 x 224 RGB, Resize(256) -> CenterCrop(224) |
| Normalization | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Upstream code repo | `github.com/plantnet/PlantNet-300K` |
| License | Apache-2.0 |

Upstream-reported accuracy figures on the model card are deliberately not
restated here; this note reports no measured performance of any kind.

## 5. Limitations

- **Task/domain gap.** The backbone was trained for 1,081 *plant species*
  identification, not the project's weed-vs-crop task under the same imaging
  conditions. Species-level features may not separate weeds from crops; this is
  an empirical question to test, not an assumption to assert.
- **512-d features.** The extractor yields 512-d vectors. The QML amplitude
  embedding requires width `2**n_qubits` (1,024 at 10 qubits), so a documented
  projection (512 -> 1,024) is still required; do not claim the backbone
  directly supplies the QML input width.
- **Frozen backbone.** No end-to-end fine-tuning; only the downstream/QML head
  adapts. This bounds what the new backbone can change and keeps the comparison
  controlled.
- **No target-label leakage.** The backbone is loaded independently of the
  target split; no target labels from train/adaptation/test are used to select
  or tune it. Photo-level group and content-hash disjointness still governs
  splitting.
- **No result claimed here.** This is a design/selection record. No accuracy,
  AUC or timing from any run is reported; absent or unmeasured values read
  `not measured`.

## 6. Acceptance criteria for the new-backbone experiment

1. A real QML run is executed with the PlantNet ResNet18 backbone and records
   provenance (repo id, weights file and hash, backbone id `plantnet`, config
   hash, seed, split manifest) in `metrics.json` / sidecar, using the loading
   code in section 3 verbatim — no silent substitution or feature-fed transfer.
2. It is compared against the ImageNet-backed control (`backbone="imagenet"`)
   on the *same* split, seed and metric suite, with both rows in the same
   artifact. No advantage language without the matched control and the
   pre-registered statistics (QML_IMPROVEMENT_PLAN P0/P1).
3. Honest reporting: `n`, mean, `std` (ddof=1), 95% CI and per-run vectors;
   `not measured` for absent values; no legacy or placeholder numbers copied
   from the archived paper; the "no quantum advantage" and CPU-simulation
   limitations remain in the model card.
4. The experiment passes Mother's central verifier and a Critic audit
   (SWARM_BOARD gate rule) before merge.

## 7. References

- PlantNet-300K ResNet18 weights and model card: https://huggingface.co/cpoisson/plantnet300k-resnet18
- Pl@ntNet-300K dataset and code: https://github.com/plantnet/PlantNet-300K
- BioCLIP: https://github.com/Imageomics/BioCLIP
- BioCLIP2 weights: https://huggingface.co/imageomics/bioclip-2
- WeedsGalore: https://github.com/GFZ/weedsgalore
- AgriNet paper (arXiv:2207.03881): https://arxiv.org/abs/2207.03881
