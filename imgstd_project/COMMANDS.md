# دليل أوامر واستخدام مشروع IMGSTD / VIDSTD (v1.0)
# IMGSTD & VIDSTD Complete Command Reference & Quickstart Guide

أداة سطر أوامر عالمية (مثل `ffmpeg`) لتحويل الصور والفيديوهات إلى مراجع دلالية سريعة يفهمها أي نموذج AI غير بصري (`--llm` / `--json`) مع دعم الضغط البكسلي المتطابق 100% ($\text{MSE} = 0.0$).

---

## 1. التثبيت كأداة عالمية في النظام (Setup & Installation)

لتثبيت التبعيات وتفعيل أمري `vidstd` و `imgstd` من أي مجلد في النظام:
```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```
للتحقق من التثبيت من أي مكان:
```bash
vidstd --version
vidstd --help
```

---

## 2. أوامر الفيديو (VIDSTD — المرجع الدلالي والضغط الكامل)

### أ) إنشاء ملف مرجعي دلالي خفيف وفائق السرعة (`< 1 MB` للـ AI غير البصري):
عندما يكون الفيديو الأصلي موجوداً لديك بالفعل وتريد ملفاً صغيراً وسريعاً (~دقيقتين لفيديو 5 دقائق على CPU):
```bash
vidstd vid-ref "video.mp4" -o "video_ref.vidstd"
```
*(أو تحكم بمعدل العينات: `vidstd vid-ref "video.mp4" -o "video_ref.vidstd" --sample-fps 2.0`).*

### ب) قراءة محتوى الفيديو في أقل من 1 ميلي ثانية لأي AI (`--llm` / `--json`):
```bash
vidstd vid-inspect "video_ref.vidstd" --llm
vidstd vid-inspect "video_ref.vidstd" --json
```

### جـ) الضغط البكسلي الكامل بدون فقدان (`100% Bit-Exact`) والاسترجاع:
```bash
vidstd vid-encode "clip.mp4" -o "clip.vidstd"
vidstd vid-decode "clip.vidstd" -o "clip_restored.mp4"
vidstd vid-frame "clip.vidstd" --frame 45 -o "frame45.png"
```

---

## 3. أوامر الصور (IMGSTD)

```bash
vidstd encode "image.png" -o "image.imgstd"
vidstd inspect "image.imgstd" --llm
vidstd decode "image.imgstd" -o "restored.png"
vidstd roundtrip "image.png"
```
*(ملاحظة: الأمران `vidstd` و `imgstd` متطابقان تماماً ويمكن استخدام أي منهما، أو تشغيل `python imgstd_project/cli.py ...` مباشرة).*

---

## 3.5 تهيئة النماذج والإعدادات (Setup, Models & Auto-Download — MobileCLIP2 & SmolVLM2)

عند استنساخ المشروع من GitHub أو تثبيته على جهاز جديد، يمكنك تهيئة النماذج الأساسية دفعة واحدة، أو اختيار أي نموذج من الإعدادات ليتم تحميله تلقائياً:

```bash
# تهيئة وتحميل النماذج الأساسية (YOLO11n-seg و YOLO-World و RapidOCR و MobileCLIP2 الافتراضي):
vidstd setup
vidstd setup --model mobileclip2-s2 --smolvlm2

# عرض قائمة النماذج المتاحة مع نوع كل نموذج (خفيف / متوازن / ثقيل / VLM) والنصائح:
vidstd models
vidstd models --json

# عرض الإعدادات الحالية وتوصيات العتاد:
vidstd config show

# حفظ نموذج افتراضي مع تحميل أوزانه تلقائياً أثناء الإعداد (mobileclip2-s0, s2, b, s3, s4, l-14):
vidstd config set --scene-model mobileclip2-s2 --smolvlm2 off
vidstd config set --smolvlm2 on --sample-fps 2.0 --yolo-conf 0.40

# حفظ الإعداد دون التحميل الفوري:
vidstd config set --scene-model mobileclip2-b --no-download

# إعادة الضبط للإعدادات الافتراضية الخفيفة المناسبة للـ CPU:
vidstd config reset

# أو تمرير نموذج مخصص لأمر واحد دون تغيير الإعدادات المحفوظة:
vidstd vid-ref "video.mp4" --model mobileclip2-b -o "ref.vidstd"
vidstd encode "image.png" --model mobileclip2-s2 --smolvlm2 -o "image.imgstd"
```

---

## 4. التحقق السريع والعرض العملي (Verification & Demo)

### أ) التحقق الكامل على أي صورة (`MSE = 0.0` و `SHA-256`):
```bash
vidstd roundtrip "image.png"
```

### ب) تشغيل العرض العملي التفاعلي (Interactive Demo):
```bash
python imgstd_project/demo.py
```

---

## 4. تشغيل سيرفر بروتوكول السياق للذكاء الاصطناعي (MCP Server)

لتشغيل خادم MCP وربطه بنماذج الـ LLM أو تطبيقات Claude / Cursor / Windsurf / Antigravity:
```bash
python imgstd_project/server/mcp_server.py
```

الأدوات والموارد المتاحة داخل السيرفر:
* `imgstd://spec/v01` (Resource): المعجم القياسي للمفردات الدلالية.
* `inspect_imgstd_code`: فحص سريع للصورة في أقل من ميلي ثانية.
* `decode_imgstd_to_image`: فك تشفير وتخزين الصورة والتحقق من بصمة SHA-256.
* `encode_image_to_imgstd`: تشفير أي صورة رقمية إلى كود IMGSTD V04 (مع إمكانية تمرير `scene_model` و `enable_smolvlm2`).
* `encode_video_to_vidstd_tool`: ترميز الفيديو أو توليد مرجع خفيف (`semantic_only=True`) مع اختيار `scene_model` و `enable_smolvlm2`.
* `list_available_models`: عرض جميع نسخ MobileCLIP2 و SmolVLM2 مع نوع كل نموذج وتوصيات العتاد.
* `get_settings` / `update_settings`: قراءة وتعديل الإعدادات المحفوظة من داخل الـ AI مباشرة.
* `recolor_imgstd_region`: إعادة تلوين كائن أو منطقة محددة دون فك التشفير مع تحديث الإزاحات وCRC32.
* `query_region_at_point`: استعلام عن الكائن الموجود عند إحداثيات (X, Y).

---

## 5. أمثلة برمجية داخل كود بايثون (Python API Examples)

### تشفير واسترجاع صورة برمجياً:
```python
from imgstd_project.core.encoder import encode_image
from imgstd_project.core.decoder import decode_imgstd
from imgstd_project.core.inspector import inspect_imgstd
from imgstd_project.core.verifier import IntegrityVerifier

# 1. التشفير
code, metrics = encode_image("input.png", output_path="saved.imgstd")
print(f"Encoded in {metrics['total_encode_time_ms']} ms")

# 2. الفحص اللحظي
summary = inspect_imgstd(code)
print(f"Found {summary['region_count']} objects in {summary['inspection_time_ms']} ms")

# 3. الاسترجاع المطابق 100%
restored_img, dec_metrics = decode_imgstd("saved.imgstd", output_image_path="restored.png")

# 4. التدقيق الرياضي
audit = IntegrityVerifier.verify_bit_exact(original_img, restored_img)
assert audit["verified"] is True
```

### استعلام مكان النقطة (Spatial Query):
```python
from imgstd_project.server.mcp_server import query_region_at_point

result = query_region_at_point("saved.imgstd", x=250, y=180)
print("Object at point:", result["class_name"])
```

### تعديل لون منطقة دون المساس بالتفاصيل (Non-Destructive Recolor):
```python
from imgstd_project.server.mcp_server import recolor_imgstd_region

# تلوين المنطقة رقم 001 باللون الأحمر [255, 0, 0]
res = recolor_imgstd_region("saved.imgstd", region_id="001", new_rgb=[255, 0, 0])
with open("recolored.imgstd", "w") as f:
    f.write(res["modified_code"])
```

### ترميز فروق الفيديو اللحظية (VIDSTD Video Delta):
```python
from imgstd_project.core.video_delta import encode_delta_frame, decode_delta_frame

# تشفير الفارق الزمني بين إطارين
delta_code, metrics = encode_delta_frame(frame_t0, frame_t1)

# استرجاع الإطار الجديد باستخدام الإطار السابق + كود الفارق
reconstructed_frame, _ = decode_delta_frame(delta_code, frame_t0)
```
