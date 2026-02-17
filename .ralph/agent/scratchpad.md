## 2026-02-17T23:50:40Z
Objective: verify/fix GLiNERConfig.model_type token-level vs token_level mismatch and determine intent from history.
Findings so far:
- Working tree already contains uncommitted replacement of "token-level" -> "token_level" in gliner/config.py model_type property.
- commit 62a831fcc83dc4a43c08aa90c86119874193714b
Author: Ingvar <igor.stepanov2000@gmail.com>
Date:   Thu Nov 20 17:12:28 2025 +0200

    fix loading different models

diff --git a/gliner/config.py b/gliner/config.py
index f96f79e..8cdd817 100644
--- a/gliner/config.py
+++ b/gliner/config.py
@@ -110,8 +110,8 @@ class UniEncoderSpanConfig(UniEncoderConfig):
 
     def __init__(self, **kwargs):
         super().__init__(**kwargs)
-        if self.span_mode == "token-level":
-            raise ValueError("UniEncoderSpanConfig requires span_mode != 'token-level'")
+        if self.span_mode == "token_level":
+            raise ValueError("UniEncoderSpanConfig requires span_mode != 'token_level'")
 
         self.model_type = "gliner_uni_encoder_span"
 
@@ -121,7 +121,7 @@ class UniEncoderTokenConfig(UniEncoderConfig):
 
     def __init__(self, **kwargs):
         super().__init__(**kwargs)
-        self.span_mode = "token-level"
+        self.span_mode = "token_level"
         self.model_type = "gliner_uni_encoder_token"
 
 
@@ -168,8 +168,8 @@ class UniEncoderSpanDecoderConfig(UniEncoderConfig):
         self.decoder_loss_coef = decoder_loss_coef
         self.span_loss_coef = span_loss_coef
         self.model_type = "gliner_uni_encoder_span_decoder"
-        if self.span_mode == "token-level":
-            raise ValueError("UniEncoderSpanDecoderConfig requires span_mode != 'token-level'")
+        if self.span_mode == "token_level":
+            raise ValueError("UniEncoderSpanDecoderConfig requires span_mode != 'token_level'")
 
 
 class UniEncoderSpanRelexConfig(UniEncoderConfig):
@@ -203,7 +203,7 @@ class UniEncoderSpanRelexConfig(UniEncoderConfig):
             **kwargs: Additional keyword arguments passed to UniEncoderConfig.
 
         Raises:
-            ValueError: If span_mode is 'token-level', which is incompatible with this config.
+            ValueError: If span_mode is 'token_level', which is incompatible with this config.
         """
         super().__init__(**kwargs)
 
@@ -216,8 +216,8 @@ class UniEncoderSpanRelexConfig(UniEncoderConfig):
         self.adjacency_loss_coef = adjacency_loss_coef
         self.relation_loss_coef = relation_loss_coef
         self.model_type = "gliner_uni_encoder_span_relex"
-        if self.span_mode == "token-level":
-            raise ValueError("UniEncoderSpanRelexConfig requires span_mode != 'token-level'")
+        if self.span_mode == "token_level":
+            raise ValueError("UniEncoderSpanRelexConfig requires span_mode != 'token_level'")
 
 
 class BiEncoderConfig(BaseGLiNERConfig):
@@ -247,8 +247,8 @@ class BiEncoderSpanConfig(BiEncoderConfig):
 
     def __init__(self, **kwargs):
         super().__init__(**kwargs)
-        if self.span_mode == "token-level":
-            raise ValueError("BiEncoderSpanConfig requires span_mode != 'token-level'")
+        if self.span_mode == "token_level":
+            raise ValueError("BiEncoderSpanConfig requires span_mode != 'token_level'")
         self.model_type = "gliner_bi_encoder_span"
 
 
@@ -257,7 +257,7 @@ class BiEncoderTokenConfig(BiEncoderConfig):
 
     def __init__(self, **kwargs):
         super().__init__(**kwargs)
-        self.span_mode = "token-level"
+        self.span_mode = "token_level"
         self.model_type = "gliner_bi_encoder_token"
 
 
@@ -297,15 +297,13 @@ class GLiNERConfig(BaseGLiNERConfig):
     @property
     def model_type(self):
         """Auto-detect model type based on configuration."""
-        span_mode_normalized = self.span_mode.replace("_", "-") if self.span_mode else None
-
         if self.labels_decoder:
             return "gliner_uni_encoder_span_decoder"
         elif self.labels_encoder:
-            return "gliner_bi_encoder_span" if span_mode_normalized != "token-level" else "gliner_bi_encoder_token"
+            return "gliner_bi_encoder_span" if self.span_mode != "token-level" else "gliner_bi_encoder_token"
         elif self.relations_layer is not None:
             return "gliner_uni_encoder_span_relex"
-        elif span_mode_normalized == "token-level":
+        elif self.span_mode == "token-level":
             return "gliner_uni_encoder_token"
         else:
             return "gliner_uni_encoder_span"
diff --git a/gliner/model.py b/gliner/model.py
index 1516324..f05f92f 100644
--- a/gliner/model.py
+++ b/gliner/model.py
@@ -323,11 +323,12 @@ class BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin):
         return config
 
     @classmethod
-    def _load_tokenizer(cls, model_dir: Path, cache_dir: Optional[Path] = None):
+    def _load_tokenizer(cls, config: GLiNERConfig, model_dir: Path, cache_dir: Optional[Path] = None):
         """
         Load tokenizer from directory.
 
         Args:
+            config: GLiNER config instance
             model_dir: Directory containing tokenizer files
             cache_dir: Cache directory for downloads
 
@@ -336,6 +337,8 @@ class BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin):
         """
         if os.path.exists(model_dir / "tokenizer_config.json"):
             return AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
+        else:
+            return AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
         return None
 
     @classmethod
@@ -612,7 +615,7 @@ class BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin):
 
         tokenizer = None
         if load_tokenizer:
-            tokenizer = cls._load_tokenizer(model_dir, cache_dir)
+            tokenizer = cls._load_tokenizer(config, model_dir, cache_dir)
 
         if not load_onnx_model:
             # Create model instance
@@ -1126,10 +1129,12 @@ class BaseEncoderGLiNER(BaseGLiNER):
         if set_class_token_index:
             self.set_class_indices()
 
-        if len(self.data_processor.transformer_tokenizer) != self.config.vocab_size and self.config.vocab_size != -1:
+        if len(self.data_processor.transformer_tokenizer) != self.config.vocab_size:
             new_num_tokens = len(self.data_processor.transformer_tokenizer)
             model_embeds = self.model.token_rep_layer.resize_token_embeddings(new_num_tokens, None)
             self.config.vocab_size = model_embeds.num_embeddings
+            if hasattr(self.config, "encoder_config"):
+                self.config.encoder_config.vocab_size = model_embeds.num_embeddings
 
     def prepare_inputs(self, texts: List[str]):
         """Prepare inputs for the model by tokenizing and creating index mappings. indicates this mismatch was introduced accidentally while broader code switched to token_level.
## 2026-02-17T23:50:54Z
Objective: verify/fix GLiNERConfig.model_type token-level vs token_level mismatch and determine intent from history.
Findings so far:
- Working tree already contains uncommitted replacement of "token-level" -> "token_level" in gliner/config.py model_type property.
- `git show 62a831f` indicates this mismatch was introduced accidentally while broader code switched to token_level.
- `_get_gliner_class()` and config classes use token_level, so hyphen checks in GLiNERConfig.model_type are inconsistent and can misclassify when model_type is derived from span_mode.
- No clear maintainer rationale in commit messages for preserving hyphen checks.
Plan:
1) Create one runtime task: finalize fix + add regression tests.
2) Add/adjust tests covering token_level routing in GLiNERConfig.model_type.
3) Run targeted tests, commit only relevant files, close task, and emit event.
Confidence: 91/100 (proceed autonomously).

## 2026-02-17T23:53:55Z
Completed task: Fix GLiNERConfig token_level model_type detection.
What I verified:
- `git show 62a831f`/`git show 2f27ea6` indicates a historical underscore migration left hyphen checks in `GLiNERConfig.model_type`, with no rationale indicating intentional behavior.
- Current code paths (`GLiNER._get_gliner_class`, config classes, config defaults) use `token_level`.
What changed:
- Kept minimal fix in `gliner/config.py` replacing four `"token-level"` comparisons with `"token_level"` in `GLiNERConfig.model_type`.
- Added regression coverage in `ptbr/tests/test_config_cli.py` for token-level model routing across base/bi-encoder/decoder/relex variants.
Verification executed:
- Static compile check: `python3 -m compileall gliner/config.py ptbr/tests/test_config_cli.py` (pass).
- Static assertion check that `model_type` property no longer contains `token-level` and includes `token_level` checks (pass).
- Runtime pytest execution blocked by environment dependencies (`pytest` missing; `uv run pytest` dependency resolution failure due jieba3 constraints; `torch`/`transformers` unavailable in base python). Logged as fix memories.
Commit: 6972469
