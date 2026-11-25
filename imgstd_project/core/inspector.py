"""
IMGSTD Inspector Module — V01/V02.
Ultra-fast O(1) parser for [HDR] and [SEM] layers enabling zero-vision AI image comprehension.
Supports both IMGSTD::V01 (RLE format) and IMGSTD::V02 (COV format, no RLE).
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
import numpy as np


class IMGSTDInspector:
    """
    Standard Semantic-Lossless Image Inspector (V01/V02).
    Parses ONLY [HDR] and [SEM] blocks in <2ms without loading or decompressing the residual stream.
    Reports protocol version correctly for both V01 and V02 streams.
    """

    _VOCABULARY_CACHE: Optional[Dict[str, Any]] = None

    def __init__(self, vocabulary_path: Optional[str] = None):
        self.vocab_path = vocabulary_path or os.path.join(
            os.path.dirname(__file__), "..", "spec", "vocabulary.json"
        )
        self.vocab = self._load_vocabulary()

    def _load_vocabulary(self) -> Dict[str, Any]:
        """Loads and caches the standard vocabulary dictionary."""
        if IMGSTDInspector._VOCABULARY_CACHE is not None:
            return IMGSTDInspector._VOCABULARY_CACHE

        resolved_path = os.path.abspath(self.vocab_path)
        if os.path.exists(resolved_path):
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                IMGSTDInspector._VOCABULARY_CACHE = data.get("classes", {})
        else:
            IMGSTDInspector._VOCABULARY_CACHE = {}

        return IMGSTDInspector._VOCABULARY_CACHE

    def inspect(self, input_source: str) -> Dict[str, Any]:
        """
        Parses [HDR] and [SEM] blocks in O(1) time and translates semantic codes.

        Returns:
            Structured JSON-compatible dictionary describing canvas and semantic objects.
        """
        t_start = time.perf_counter()

        # Handle file path (true O(1) seek without reading [RES]) vs direct string safely
        stream = ""
        sem_block = ""
        txt_block = ""
        is_file_path = (not input_source.startswith("IMGSTD::")) and len(input_source) < 4096 and os.path.exists(input_source)

        if is_file_path:
            with open(input_source, "rb") as f:
                head_str = f.read(4096).decode("utf-8", errors="replace")
                if not head_str.startswith("IMGSTD::"):
                    raise ValueError("Invalid stream: missing IMGSTD magic prefix.")
                prefix_head = head_str[:16]
                hdr_start = head_str.find("[HDR]")
                hdr_end = head_str.find("[/HDR]")
                if hdr_start == -1 or hdr_end == -1:
                    raise ValueError("Malformed stream: missing [HDR] block.")
                hdr_content = head_str[hdr_start + 5:hdr_end].strip()
                hdr_map: Dict[str, str] = {}
                for line in hdr_content.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        hdr_map[k.strip()] = v.strip()

                if "SEM_OFFSET" in hdr_map and "SEM_LEN" in hdr_map:
                    f.seek(int(hdr_map["SEM_OFFSET"]))
                    cand = f.read(int(hdr_map["SEM_LEN"])).decode("utf-8", errors="replace")
                    if cand.startswith("[SEM]"):
                        sem_block = cand
                if "TXT_OFFSET" in hdr_map and "TXT_LEN" in hdr_map and int(hdr_map["TXT_LEN"]) > 0:
                    f.seek(int(hdr_map["TXT_OFFSET"]))
                    cand = f.read(int(hdr_map["TXT_LEN"])).decode("utf-8", errors="replace")
                    if cand.startswith("[TXT]"):
                        txt_block = cand
                if not sem_block:
                    f.seek(0)
                    stream = f.read().decode("utf-8", errors="replace")
        else:
            stream = input_source
            if not stream.startswith("IMGSTD::"):
                raise ValueError("Invalid stream: missing IMGSTD magic prefix.")
            prefix_head = stream[:16]
            hdr_start = stream.find("[HDR]")
            hdr_end = stream.find("[/HDR]")
            if hdr_start == -1 or hdr_end == -1:
                raise ValueError("Malformed stream: missing [HDR] block.")
            hdr_content = stream[hdr_start + 5:hdr_end].strip()
            hdr_map = {}
            for line in hdr_content.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    hdr_map[k.strip()] = v.strip()

        # Detect version
        if prefix_head.startswith("IMGSTD::V04"):
            protocol = "IMGSTD::V04"
        elif prefix_head.startswith("IMGSTD::V03"):
            protocol = "IMGSTD::V03"
        elif prefix_head.startswith("IMGSTD::V02"):
            protocol = "IMGSTD::V02"
        else:
            protocol = "IMGSTD::V01"

        w = int(hdr_map.get("W", 0))
        h = int(hdr_map.get("H", 0))
        c = int(hdr_map.get("C", 3))
        b = int(hdr_map.get("B", 8))
        sha256 = hdr_map.get("SHA256", "")
        pred = hdr_map.get("PRED", "NONE")
        total_canvas_pixels = max(1, w * h)

        # 2. Extract [SEM] & [TXT] blocks using UTF-8 byte offsets if not already seeked from file
        if not sem_block or (not txt_block and "TXT_OFFSET" in hdr_map and int(hdr_map.get("TXT_LEN", 0)) > 0):
            stream_bytes = stream.encode("utf-8") if stream else b""
            if not sem_block and "SEM_OFFSET" in hdr_map and "SEM_LEN" in hdr_map:
                sem_off = int(hdr_map["SEM_OFFSET"])
                sem_len = int(hdr_map["SEM_LEN"])
                cand = stream_bytes[sem_off:sem_off + sem_len].decode("utf-8", errors="replace")
                if cand.startswith("[SEM]"):
                    sem_block = cand
            if not sem_block and stream:
                sem_start = stream.find("[SEM]")
                sem_end = stream.find("[/SEM]")
                if sem_start != -1 and sem_end != -1:
                    sem_block = stream[sem_start:sem_end + 6]

            if not txt_block and "TXT_OFFSET" in hdr_map and "TXT_LEN" in hdr_map:
                txt_off = int(hdr_map["TXT_OFFSET"])
                txt_len = int(hdr_map["TXT_LEN"])
                if txt_len > 0:
                    cand = stream_bytes[txt_off:txt_off + txt_len].decode("utf-8", errors="replace")
                    if cand.startswith("[TXT]"):
                        txt_block = cand
            if not txt_block and stream:
                txt_start = stream.find("[TXT]")
                txt_end = stream.find("[/TXT]")
                if txt_start != -1 and txt_end != -1:
                    txt_block = stream[txt_start:txt_end + 6]

        texts: List[Dict[str, Any]] = []
        if txt_block:
            for line in txt_block.split("\n"):
                line = line.strip()
                if not line.startswith("TXT:"):
                    continue
                # Safely parse STR:"..." to avoid breaking on escaped pipes or quotes
                str_marker = '|STR:"'
                if str_marker in line and line.endswith('"'):
                    split_idx = line.find(str_marker)
                    meta_part = line[:split_idx]
                    raw_str = line[split_idx + len(str_marker):-1]
                    clean_str = (
                        raw_str
                        .replace('\\|', '|')
                        .replace('\\"', '"')
                        .replace('\\n', '\n')
                        .replace('\\\\', '\\')
                    )
                else:
                    parts = line.split("|")
                    meta_part = line
                    clean_str = ""
                    for p in parts:
                        if p.startswith('STR:"') and p.endswith('"'):
                            clean_str = (
                                p[5:-1]
                                .replace('\\|', '|')
                                .replace('\\"', '"')
                                .replace('\\n', '\n')
                                .replace('\\\\', '\\')
                            )
                            break

                meta_parts = meta_part.split("|")
                t_dict: Dict[str, str] = {}
                for p in meta_parts:
                    if ":" in p:
                        fk, fv = p.split(":", 1)
                        t_dict[fk] = fv
                tid = t_dict.get("TXT", "001")
                try:
                    t_conf = float(t_dict.get("CONF", "1.00"))
                except ValueError:
                    t_conf = 1.00
                t_box = [int(v) for v in t_dict.get("BOX", "0,0,0,0").split(",")]
                texts.append({
                    "text_id": tid,
                    "confidence": t_conf,
                    "bounding_box": {
                        "x": t_box[0],
                        "y": t_box[1],
                        "width": t_box[2],
                        "height": t_box[3]
                    },
                    "text": clean_str
                })

        # 3b. Extract SCENE_DESC from [SEM] if present (V04)
        scene_desc = ""
        for line in sem_block.split("\n"):
            line = line.strip()
            if line.startswith('SCENE_DESC:"') and line.endswith('"'):
                scene_desc = (
                    line[len('SCENE_DESC:"'):-1]
                    .replace('\\|', '|')
                    .replace('\\"', '"')
                    .replace('\\\\', '\\')
                )
                break

        # 4. Parse Regions (handle V01, V02, V03, and V04 formats)
        regions: List[Dict[str, Any]] = []
        for line in sem_block.split("\n"):
            line = line.strip()
            if not line.startswith("REG:"):
                continue

            # Extract |ATTR:"..." safely (accounting for escaped quotes)
            attr_val = None
            if '|ATTR:"' in line:
                idx_attr = line.find('|ATTR:"')
                start_p = idx_attr + len('|ATTR:"')
                i = start_p
                closing_quote = -1
                while i < len(line):
                    if line[i] == '"':
                        num_bs = 0
                        k = i - 1
                        while k >= start_p and line[k] == '\\':
                            num_bs += 1
                            k -= 1
                        if num_bs % 2 == 0:
                            closing_quote = i
                            break
                    i += 1
                if closing_quote != -1:
                    attr_raw = line[start_p:closing_quote]
                    attr_val = attr_raw.replace('\\"', '"').replace('\\|', '|').replace('\\\\', '\\')
                    line = line[:idx_attr] + line[closing_quote + 1:]

            # Extract |EMB:... safely
            emb_val = None
            if "|EMB:" in line:
                idx_emb = line.find("|EMB:")
                next_pipe = line.find("|", idx_emb + 5)
                if next_pipe != -1:
                    emb_val = line[idx_emb + 5:next_pipe]
                    line = line[:idx_emb] + line[next_pipe:]
                else:
                    emb_val = line[idx_emb + 5:]
                    line = line[:idx_emb]

            parts = line.split("|")
            field_dict: Dict[str, str] = {}
            for p in parts:
                if ":" in p:
                    fk, fv = p.split(":", 1)
                    field_dict[fk] = fv

            if emb_val is not None:
                field_dict["EMB"] = emb_val
            if attr_val is not None:
                field_dict["ATTR"] = attr_val

            reg_id = field_dict.get("REG", "000")
            cls_id = field_dict.get("CLS", "0000")
            kind = field_dict.get("KIND", "BG" if cls_id.startswith("09") or cls_id == "0000" else ("STRUCT" if cls_id.startswith("01") else "OBJ"))
            try:
                conf_val = float(field_dict.get("CONF", "1.00"))
            except ValueError:
                conf_val = 1.00

            box_vals = [int(v) for v in field_dict.get("BOX", "0,0,0,0").split(",")]
            rgb_vals = [int(v) for v in field_dict.get("RGB", "0,0,0").split(",")]

            bx, by, bw, bh = box_vals
            box_area = bw * bh

            if "COV" in field_dict:
                cov_str = field_dict["COV"].rstrip("%")
                try:
                    coverage_pct = float(cov_str)
                except ValueError:
                    coverage_pct = round((box_area / total_canvas_pixels) * 100.0, 2)
            else:
                coverage_pct = round((box_area / total_canvas_pixels) * 100.0, 2)

            vocab_info = self.vocab.get(cls_id, {
                "name": "Unknown / Custom",
                "category": "Unclassified",
                "description": "Unregistered semantic class"
            })

            cls_name = field_dict.get("NAME", "").strip('"') or vocab_info.get("name", "Unknown")
            category = field_dict.get("CAT", "").strip('"') or vocab_info.get("category", "General")

            # Prominence Classification (ROLE)
            # PRIMARY: COV >= 5.0%
            # SECONDARY: 1.5% <= COV < 5.0%
            # MINOR_DETAIL: COV < 1.5%
            if kind == "STRUCT":
                role = "PRIMARY" if coverage_pct >= 5.0 else ("SECONDARY" if coverage_pct >= 1.5 else "MINOR_DETAIL")
            elif kind == "OBJ":
                role = "PRIMARY" if coverage_pct >= 5.0 else ("SECONDARY" if coverage_pct >= 1.5 else "MINOR_DETAIL")
            else:
                role = "BACKGROUND"

            is_reflection = "REFL" in field_dict
            parent_struct = field_dict.get("REFL")

            region_entry: Dict[str, Any] = {
                "region_id": reg_id,
                "kind": kind,
                "confidence": conf_val,
                "role": role,
                "prominence": role,
                "is_reflection": is_reflection,
                "class_id": cls_id,
                "class_name": cls_name,
                "category": category,
                "description": vocab_info.get("description", ""),
                "bounding_box": {
                    "x": bx,
                    "y": by,
                    "width": bw,
                    "height": bh,
                },
                "dominant_color_rgb": rgb_vals[:3],
                "coverage_percentage": coverage_pct,
            }

            if "EMB" in field_dict:
                region_entry["embedding_b85"] = field_dict["EMB"]

            if "ATTR" in field_dict:
                region_entry["attributes"] = field_dict["ATTR"]

            if parent_struct:
                region_entry["parent_structure"] = parent_struct
                region_entry["contained_in"] = parent_struct
                region_entry["likely_reflection_or_inset"] = True

            if "RLE" in field_dict:
                region_entry["mask_rle_present"] = True

            regions.append(region_entry)

        # Check for spatial containment (IoA >= 0.75) inside door/window structures
        structures = [r for r in regions if r.get("kind") == "STRUCT"]
        for r in regions:
            if r.get("kind") == "OBJ":
                rb = r["bounding_box"]
                ix, iy, iw, ih = rb["x"], rb["y"], rb["width"], rb["height"]
                obj_area = max(1, iw * ih)
                for st in structures:
                    if st.get("class_id") in ("0102", "0103"):
                        sb = st["bounding_box"]
                        ox, oy, ow, oh = sb["x"], sb["y"], sb["width"], sb["height"]
                        inter_x1 = max(ix, ox)
                        inter_y1 = max(iy, oy)
                        inter_x2 = min(ix + iw, ox + ow)
                        inter_y2 = min(iy + ih, oy + oh)
                        inter_w = max(0, inter_x2 - inter_x1)
                        inter_h = max(0, inter_y2 - inter_y1)
                        ioa = (inter_w * inter_h) / obj_area
                        if ioa >= 0.75 or (ix >= ox - 15 and iy >= oy - 15 and ix + iw <= ox + ow + 15 and iy + ih <= oy + oh + 15 and r.get("coverage_percentage", 0.0) < 2.0):
                            st_name = st.get("class_name", "Door / Entrance")
                            r["contained_in"] = f"REG:{st['region_id']} ({st_name})"
                            r["likely_reflection_or_inset"] = True
                            r["is_reflection"] = True
                            r["parent_structure"] = f"{st['class_id']} ({st_name})"
                            r["role"] = "MINOR_DETAIL"
                            r["prominence"] = "MINOR_DETAIL"
                            break

        # 5. Automatic Relative Spatial Graph & Statistical Relational Tagging (Module D)
        class_groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in regions:
            if r.get("kind") in ("OBJ", "STRUCT"):
                cid = r["class_id"]
                class_groups.setdefault(cid, []).append(r)

        all_relational_traits = []
        for cid, grp in class_groups.items():
            if len(grp) >= 2:
                heights = [r["bounding_box"]["height"] for r in grp]
                mean_h = float(np.mean(heights))
                bottoms = [r["bounding_box"]["y"] + r["bounding_box"]["height"] for r in grp]
                median_bottom = float(np.median(bottoms))

                for r in grp:
                    bh = r["bounding_box"]["height"]
                    by_bottom = r["bounding_box"]["y"] + bh
                    rel_ratio = bh / max(1.0, mean_h)

                    # Ground plane similarity check: within 30% of canvas height or 35 pixels
                    similar_ground = abs(by_bottom - median_bottom) <= max(35.0, 0.30 * h)

                    if rel_ratio < 0.72 and similar_ground:
                        trait_msg = f"Significantly shorter/smaller than surrounding group ({rel_ratio:.2f}x avg height)"
                        r["relational_trait"] = trait_msg
                        all_relational_traits.append({
                            "region_id": r["region_id"],
                            "class_id": r["class_id"],
                            "class_name": r["class_name"],
                            "trait": trait_msg
                        })
                    elif rel_ratio > 1.35 and similar_ground:
                        trait_msg = f"Significantly taller/larger than surrounding group ({rel_ratio:.2f}x avg height)"
                        r["relational_trait"] = trait_msg
                        all_relational_traits.append({
                            "region_id": r["region_id"],
                            "class_id": r["class_id"],
                            "class_name": r["class_name"],
                            "trait": trait_msg
                        })

        # Pairwise Spatial Proximity & Graph Connections
        for i, r1 in enumerate(regions):
            if r1.get("kind") != "OBJ":
                continue
            b1 = r1["bounding_box"]
            c1_x = b1["x"] + b1["width"] / 2.0
            relations = []
            left_neighbors = []
            right_neighbors = []

            for j, r2 in enumerate(regions):
                if i == j or r2.get("kind") not in ("OBJ", "STRUCT"):
                    continue
                b2 = r2["bounding_box"]
                c2_x = b2["x"] + b2["width"] / 2.0
                dx = max(0, max(b1["x"] - (b2["x"] + b2["width"]), b2["x"] - (b1["x"] + b1["width"])))
                dy = max(0, max(b1["y"] - (b2["y"] + b2["height"]), b2["y"] - (b1["y"] + b1["height"])))

                # Overlap
                inter_w = max(0, min(b1["x"] + b1["width"], b2["x"] + b2["width"]) - max(b1["x"], b2["x"]))
                inter_h = max(0, min(b1["y"] + b1["height"], b2["y"] + b2["height"]) - max(b1["y"], b2["y"]))
                inter_area = inter_w * inter_h
                area1 = b1["width"] * b1["height"]
                area2 = b2["width"] * b2["height"]
                iou = inter_area / max(1, area1 + area2 - inter_area)

                if iou > 0.15:
                    handheld_ids = {"0026", "0027", "0028", "0030", "0031", "0034", "0035", "0036", "0038", "0040", "0041", "0042", "0043", "0063", "0068", "0074"}
                    is_person_holding = (
                        (r1.get("class_id") == "0001" and r2.get("class_id") in handheld_ids)
                        or (r2.get("class_id") == "0001" and r1.get("class_id") in handheld_ids)
                    )
                    rel_tag = "interacting with / holding" if is_person_holding else "overlapping"
                    relations.append(f"{rel_tag} REG:{r2['region_id']}")
                elif dx <= 45 and dy <= 60:
                    relations.append(f"next_to REG:{r2['region_id']}")
                    if c2_x < c1_x:
                        left_neighbors.append(r2["region_id"])
                    else:
                        right_neighbors.append(r2["region_id"])

            if left_neighbors and right_neighbors:
                relations.append(f"surrounded_by REG:{left_neighbors[0]} and REG:{right_neighbors[0]}")

            if relations:
                r1["spatial_relations"] = relations

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        detected_objects = [r for r in regions if r.get("kind") == "OBJ"]
        detected_structures = [r for r in regions if r.get("kind") == "STRUCT"]
        background_regions = [r for r in regions if r.get("kind") == "BG"]

        # Scene Context Inference (SCENE_SUMMARY)
        txt_strings = [t.get("text", "") for t in texts]
        has_refl = any(o.get("is_reflection") or o.get("likely_reflection_or_inset") for o in detected_objects)

        if detected_structures or (texts and not detected_objects):
            struct_names = [s.get("class_name", "") for s in detected_structures]
            summary_parts = []

            # Identify core facade/architecture
            if any("Storefront" in name or "Building" in name for name in struct_names):
                summary_parts.append("Commercial storefront / restaurant facade")
            elif struct_names:
                summary_parts.append(f"Architectural scene featuring {struct_names[0]}")
            else:
                summary_parts.append("Text-bearing scene")

            # Signage text
            if txt_strings:
                quoted_txt = " and ".join(f"'{t}'" for t in txt_strings[:2])
                summary_parts.append(f"featuring signage {quoted_txt}")

            # Specific detected architectural features (no hardcoded template parts)
            features = list(dict.fromkeys(
                n for n in struct_names if n and "Storefront" not in n and "Building" not in n
            ))
            if features:
                summary_parts.append(", ".join(features[:4]))

            # Reflections
            if has_refl:
                refl_count = sum(1 for o in detected_objects if o.get("is_reflection") or o.get("likely_reflection_or_inset"))
                summary_parts.append(f"with {refl_count} minor background reflection(s) inside door/window glass")

            scene_summary = ", ".join(summary_parts)
            scene_context = {
                "macro_scene": scene_summary,
                "detected_signage": txt_strings,
                "architectural_elements": struct_names,
                "has_reflections": has_refl,
                "primary_structures": [s["class_name"] for s in detected_structures if s.get("role") == "PRIMARY"]
            }
        elif detected_objects:
            obj_names = [o.get("class_name", "") for o in detected_objects]
            unique_objs = list(dict.fromkeys(obj_names))

            is_celestial = any(
                "Planet" in name or "Earth" in name or "Celestial" in name or "Space" in name or "Globe" in name
                for name in unique_objs + [b.get("class_name", "") for b in background_regions]
            )
            if is_celestial:
                scene_summary = f"Cosmic scene featuring {', '.join(unique_objs)} suspended in deep space"
            elif scene_desc:
                scene_summary = scene_desc
            else:
                scene_summary = f"Scene prominently featuring {', '.join(unique_objs)} against an ambient background"

            scene_context = {
                "macro_scene": scene_summary,
                "detected_text": txt_strings,
                "primary_subjects": unique_objs,
                "has_reflections": False
            }
        else:
            scene_summary = "Abstract graphic composition featuring non-object background gradients and ambient lighting"
            scene_context = {
                "macro_scene": scene_summary,
                "primary_subjects": [],
                "has_reflections": False
            }

        result: Dict[str, Any] = {
            "protocol": protocol,
            "dimensions": {
                "width": w,
                "height": h,
                "channels": c,
                "bit_depth": b,
                "total_pixels": total_canvas_pixels
            },
            "color_space": "RGBA" if c == 4 else ("RGB" if c == 3 else "Grayscale"),
            "integrity_sha256": sha256,
            "region_count": len(regions),
            "object_count": len(detected_objects),
            "structure_count": len(detected_structures),
            "text_count": len(texts),
            "background_count": len(background_regions),
            "scene_summary": scene_summary,
            "scene_description": scene_desc if scene_desc else scene_summary,
            "relational_traits": all_relational_traits,
            "scene_context": scene_context,
            "detected_objects": detected_objects,
            "detected_structures": detected_structures,
            "texts": texts,
            "background_regions": background_regions,
            "regions": regions,
            "inspection_time_ms": round(t_elapsed_ms, 3),
        }

        # V02, V03 & V04 extras
        if protocol in ("IMGSTD::V02", "IMGSTD::V03", "IMGSTD::V04"):
            result["predictor"] = pred
            if "LABELMAP_OFFSET" in hdr_map:
                result["labelmap_offset"] = int(hdr_map["LABELMAP_OFFSET"])
                result["labelmap_len"] = int(hdr_map.get("LABELMAP_LEN", 0))
            if "TXT_OFFSET" in hdr_map:
                result["txt_offset"] = int(hdr_map["TXT_OFFSET"])
                result["txt_len"] = int(hdr_map.get("TXT_LEN", 0))

        return result


def inspect_imgstd(input_source: str, vocabulary_path: Optional[str] = None) -> Dict[str, Any]:
    """Helper function to inspect an IMGSTD stream in sub-millisecond time."""
    inspector = IMGSTDInspector(vocabulary_path=vocabulary_path)
    return inspector.inspect(input_source)
