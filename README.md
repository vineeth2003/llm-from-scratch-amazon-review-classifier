<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: auto;
            padding: 20px;
            color: #222;
        }
        h1, h2, h3 {
            color: #0b5394;
        }
        code, pre {
            background: #f4f4f4;
            padding: 6px;
            border-radius: 4px;
            font-family: Consolas, monospace;
        }
        pre {
            overflow-x: auto;
        }
        ul, ol {
            margin-left: 20px;
        }
        hr {
            margin: 30px 0;
        }
    </style>
</head>
<body>

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

<p>
Fixed-length padding is used for efficient batching:
</p>

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

<p>
This wrapper around the GPT model:
</p>
<ul>
    <li>Extracts token embeddings</li>
    <li>Applies padding-aware pooling</li>
    <li>Feeds the pooled representation into a classification head</li>
</ul>

<h3>Padding-aware pooling (Critical Fix)</h3>

<p><strong>Incorrect (causes model collapse):</strong></p>
<pre><code>
x = x[:, -1, :]
</code></pre>

<p><strong>Correct implementation:</strong></p>
<pre><code>
attention_mask = (input_ids != pad_token_id)
last_token_idx = attention_mask.sum(dim=1) - 1
x = x[torch.arange(x.size(0)), last_token_idx]
</code></pre>

<p>
This ensures the model always uses the <strong>last real token</strong> instead of padding.
</p>

<hr>

<h2>LoRA (Low-Rank Adaptation)</h2>

<p>
Instead of fine-tuning all GPT parameters, LoRA adapters are injected into
the <strong>Q, K, V attention projections</strong>.
</p>

<h3>LoRA Formula</h3>
<pre><code>
output = W(x) + scaling * (x @ A @ B)
</code></pre>

<ul>
    <li>Original GPT weights are frozen</li>
    <li>Only LoRA matrices <code>A</code> and <code>B</code> are trained</li>
    <li>Significantly reduces trainable parameters</li>
</ul>

<hr>

<h2>Parameter Freezing Strategy</h2>

<ol>
    <li>Freeze the entire GPT model</li>
    <li>Unfreeze only:
        <ul>
            <li>LoRA parameters</li>
            <li>Classification head</li>
        </ul>
    </li>
</ol>

<pre><code>
if "lora_" in name or "classifier" in name:
    p.requires_grad = True
</code></pre>

<p>Trainable parameters include:</p>
<ul>
    <li><code>lora_A</code>, <code>lora_B</code> (attention layers)</li>
    <li><code>classifier.weight</code>, <code>classifier.bias</code></li>
</ul>

<hr>

<h2>Optimizer</h2>

<p>Only trainable parameters are passed to the optimizer:</p>

<pre><code>
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, clf_model.parameters()),
    lr=3e-4
)
</code></pre>

<p>This ensures frozen GPT weights remain unchanged.</p>

<hr>

<h2>Training Loop</h2>

<pre><code>
clf_model.train()

for epoch in range(2):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = clf_model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
</code></pre>

<ul>
    <li>Uses <code>CrossEntropyLoss</code></li>
    <li>Short training to prevent overfitting</li>
    <li>Fully GPU-compatible</li>
</ul>

<hr>

<h2>Inference</h2>

<pre><code>
def predict(text):
    clf_model.eval()
    ...
    pred = torch.argmax(logits, dim=-1).item()
    return "Positive" if pred == 1 else "Negative"
</code></pre>

<p>Predictions correctly handle padding and batching.</p>

<hr>

<h2>Common Pitfall (Solved)</h2>

<p><strong>Problem:</strong> Model always predicted “Positive”.</p>

<p><strong>Root Cause:</strong></p>
<ul>
    <li>Using last token embedding</li>
    <li>Last token often PAD</li>
    <li>PAD embeddings are constant → representation collapse</li>
</ul>

<p><strong>Solution:</strong> Padding-aware pooling using <code>pad_token_id</code>.</p>

<hr>

<h2>Key Takeaways</h2>
<ul>
    <li>Proper handling of padding is critical for GPT classifiers</li>
    <li>LoRA enables efficient fine-tuning on limited hardware</li>
    <li>Always verify which parameters are trainable</li>
    <li>Silent bugs can completely prevent learning</li>
</ul>

<hr>

<h2>Future Extensions</h2>
<ul>
    <li>Multi-class classification</li>
    <li>Evaluation metrics (Accuracy, F1)</li>
    <li>Saving and loading LoRA adapters</li>
    <li>Integration with HuggingFace Transformers</li>
</ul>

<hr>

<h2>Author</h2>
<p>
Built by <strong>Vineeth Benakashetty</strong><br>
Focused on practical and efficient NLP model design.
</p>

<hr>

<h2>License</h2>
<p>MIT License</p>

</body>
</html>
