

<h1>GPT-based Text Classification with LoRA</h1>

<p>
This project implements a <strong>binary text classification model (Positive / Negative)</strong>
using a <strong>GPT-style Transformer</strong> with
<strong>LoRA (Low-Rank Adaptation)</strong> for parameter-efficient fine-tuning.
</p>

<p>
The focus is on <strong>correct padding-aware pooling</strong>,
efficient training, and avoiding silent model failures that can occur in GPT classifiers.
</p>

<hr>

<h2>Features</h2>
<ul>
    <li>GPT-based sequence classification</li>
    <li>Parameter-efficient fine-tuning using <strong>LoRA</strong></li>
    <li>Padding-aware pooling for robust training</li>
    <li>Frozen backbone with trainable adapters</li>
    <li>Clean PyTorch training and inference pipeline</li>
    <li>GPU-compatible (CUDA / Google Colab)</li>
</ul>

<hr>

<h2>Model Architecture</h2>

<pre>
Input Text
   ↓
Tokenizer (pad + truncate)
   ↓
Input IDs (batch, seq_len)
   ↓
GPT Transformer (frozen)
   ↓
Padding-aware last-token pooling
   ↓
Classification Head
   ↓
Logits (Positive / Negative)
</pre>

<hr>

<h2>Requirements</h2>
<ul>
    <li>Python ≥ 3.9</li>
    <li>PyTorch</li>
    <li>Transformers library or custom GPT implementation</li>
    <li>CUDA (optional but recommended)</li>
</ul>

<hr>

<h2>Tokenization</h2>

<p>Fixed-length padding is used for efficient batching:</p>

<pre><code>
tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
</code></pre>

<p>
The <code>pad_token_id</code> is explicitly used during pooling to prevent PAD-token leakage.
</p>

<hr>

<h2>GPTForClassification Wrapper</h2>

<ul>
    <li>Extracts token embeddings</li>
    <li>Applies padding-aware pooling</li>
    <li>Feeds the pooled representation into a classification head</li>
</ul>

<h3>Padding-aware pooling (Critical Fix)</h3>

<p><strong>Incorrect:</strong></p>
<pre><code>
x = x[:, -1, :]
</code></pre>

<p><strong>Correct:</strong></p>
<pre><code>
attention_mask = (input_ids != pad_token_id)
last_token_idx = attention_mask.sum(dim=1) - 1
x = x[torch.arange(x.size(0)), last_token_idx]
</code></pre>

<hr>

<h2>LoRA (Low-Rank Adaptation)</h2>

<p>
LoRA adapters are injected into the
<strong>Q, K, V attention projections</strong> while keeping GPT weights frozen.
</p>

<pre><code>
output = W(x) + scaling * (x @ A @ B)
</code></pre>

<ul>
    <li>Only LoRA parameters are trained</li>
    <li>Drastically reduces memory and compute cost</li>
</ul>

<hr>

<h2>Parameter Freezing Strategy</h2>

<pre><code>
if "lora_" in name or "classifier" in name:
    p.requires_grad = True
</code></pre>

<hr>

<h2>Optimizer</h2>

<pre><code>
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, clf_model.parameters()),
    lr=3e-4
)
</code></pre>

<hr>

<h2>Training Loop</h2>

<pre><code>
clf_model.train()
...
</code></pre>

<hr>

<h2>Inference</h2>

<pre><code>
return "Positive" if pred == 1 else "Negative"
</code></pre>

<hr>

<h2>Author</h2>

<p>
<strong>Vineeth Benakashetty</strong><br>
Focused on practical and efficient NLP model design.
</p>
