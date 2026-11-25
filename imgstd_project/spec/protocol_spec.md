# Standard Semantic-Lossless Image Codec Protocol Specification (IMGSTD V01 - V04)

## 1. Overview & Architectural Principles

The **IMGSTD** protocol defines a deterministic, self-describing image container and serialization format based on **Hierarchical Lossless Decomposition**:

$$\text{Original Image Matrix } (I) \equiv \text{Semantic Base Canvas } (B) + \text{Exact Lossless Residual } (R)$$

The serialization guarantees:
1. **100% Bit-Exact Lossless Invertibility:** $SHA256(Decoded(Encoded(I))) \equiv SHA256(I)$ with zero error ($\text{MSE} \equiv 0.0$).
2. **Zero-Vision AI Comprehension:** Any AI or parser can inspect image semantics, scene contents, objects, bounding boxes, colors, layout, and visible typography directly from the header without decoding the residual stream.
3. **Sub-Second Execution:** Vectorized numpy operations with modular arithmetic $(I - B) \pmod{256}$ and $(B + R) \pmod{256}$ achieve $<150\text{ms}$ latency even on 5K images.

---

## 2. Container Structure (V04 Multimodal Standard)

An `IMGSTD::V04` stream is composed of five strictly ordered layers:

```text
IMGSTD::V04
[HDR]
VER:V04
W:<width>
H:<height>
C:<channels>
B:<bit_depth>
PRED:YCOCG_HDELTA
SHA256:<sha256_hex_digest_of_raw_pixels>
SEM_OFFSET:<byte_offset_of_SEM_block>
SEM_LEN:<byte_length_of_SEM_block>
TXT_OFFSET:<byte_offset_of_TXT_block>
TXT_LEN:<byte_length_of_TXT_block>
RES_OFFSET:<byte_offset_of_RES_block>
RES_LEN:<byte_length_of_RES_block>
LABELMAP_OFFSET:<byte_offset_of_LABEL_MAP_Z85>
LABELMAP_LEN:<byte_length_of_LABEL_MAP_Z85>
[/HDR]
[SEM]
REGIONS:<count>
LABEL_MAP_Z85:<base85_encoded_zstd_compressed_label_map>
REG:001|KIND:STRUCT|CLS:0106|CONF:0.95|BOX:<x>,<y>,<w>,<h>|RGB:<r>,<g>,<b>|COV:<coverage>%
REG:002|KIND:STRUCT|CLS:0102|CONF:0.90|BOX:<x>,<y>,<w>,<h>|RGB:<r>,<g>,<b>|COV:<coverage>%
REG:003|KIND:OBJ|CLS:0003|CONF:0.71|BOX:<x>,<y>,<w>,<h>|RGB:<r>,<g>,<b>|COV:<coverage>%|REFL:0102
...
[/SEM]
[TXT]
TEXTS:<count>
TXT:001|CONF:0.86|BOX:<x>,<y>,<w>,<h>|STR:"HEALTH RANGE"
TXT:002|CONF:0.89|BOX:<x>,<y>,<w>,<h>|STR:"مطعم صحي"
[/TXT]
[RES]
<base85_encoded_zstd_compressed_predicted_residual_bytes>
[/RES]
[END]
CRC32:<hex_crc32_checksum>
TOTAL_BYTES:<total_container_byte_length>
[/END]
```

---

## 3. V04 Multimodal Semantic Engine Upgrades

### Upgrade A: Fast Multilingual Scene OCR Layer (`[TXT]`)
- Integrated CPU OCR engine (`rapidocr-onnxruntime`) capable of reading English, Arabic, numerals, and multilingual typography in $<45\text{ms}$.
- Indexed directly via `TXT_OFFSET` and `TXT_LEN` in `[HDR]` for instant $\mathcal{O}(1)$ inspection without reading or parsing residual bytes.

### Upgrade B: Open-Vocabulary Architectural Structural Detection
- Resolves COCO-80 blind spots by capturing architectural and environmental structures (`0100`–`0299`):
  - `0101`: Building / Storefront Facade
  - `0102`: Door / Entrance
  - `0103`: Window / Glass Panel
  - `0104`: Stairs / Steps
  - `0105`: Signboard / Banner
  - `0106`: Wall / Pillar
  - `0107`: Floor / Pavement
  - `0108`: Roof / Awning / Canopy
- Dual-model vision extraction: YOLO11n-seg primary sensor + open-vocabulary secondary detector (YOLO-World / YOLOE) with graceful fallback to signage/geometry heuristics.
- Structural regions are identified and tagged as `KIND:STRUCT` in `[SEM]`.

### Upgrade C: Spatial Prominence, Containment & Scene Context
- Eliminates flat spatial hierarchy in LLM narration:
  - Entities classified into `PRIMARY` ($\text{COV} \ge 5\%$), `SECONDARY` ($1.5\% \le \text{COV} < 5\%$), and `MINOR_DETAIL` ($\text{COV} < 1.5\%$).
  - Spatial containment via $\text{IoA} \ge 0.75$: Objects situated inside doors or windows (e.g. tiny car reflections in storefront entrance doors) are tagged with `contained_in: "REG:... (Door / Entrance)"`, `likely_reflection_or_inset: true`, and `is_reflection: true`.
  - High-level `scene_context` and `scene_summary` inference synthesizes structural elements + visible typography into macro scene descriptions for Zero-Vision LLMs.
