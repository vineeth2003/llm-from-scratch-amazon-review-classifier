<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GPT-based Text Classification with LoRA</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            max-width: 900px;
            margin: auto;
            padding: 20px;
            color: #222;
            background-color: #fdfdfd;
        }
        h1, h2, h3 {
            color: #0b5394;
            margin-top: 1.5em;
        }
        h1 {
            font-size: 2.2em;
            margin-bottom: 0.5em;
        }
        h2 {
            font-size: 1.7em;
        }
        h3 {
            font-size: 1.3em;
        }
        code, pre {
            background: #f4f4f4;
            padding: 6px 8px;
            border-radius: 5px;
            font-family: Consolas, monospace;
            font-size: 0.95em;
        }
        pre {
            overflow-x: auto;
        }
        ul, ol {
            margin-left: 20px;
            margin-bottom: 1em;
        }
        hr {
            margin: 40px 0;
            border: 1px solid #ddd;
        }
        p {
            margin-bottom: 1em;
        }
        .highlight {
            font-weight: bold;
            color: #c00000;
        }
    </style>
</head>
<body>

<h1>GPT-based Text Classification with LoRA</h1>

<p>
This project implements a <strong>binary text classification model</strong> (Positive / Negative) using a 
<strong>GPT-style Transformer</strong> with <strong>LoRA (Low-Rank Adaptation)</strong> for efficient fine-tuning. 
The implementation emphasizes correct <em>padding-aware pooling</em>, reliable training, and avoiding common pitfalls in GPT classifiers.
</p>

<hr>

<h2>🚀 Key Features</h2>
<ul>
    <li>GPT-based sequence classification</li>
    <li>Parameter-efficient fine-tuning via <strong>LoRA</strong></li>
    <li>Padding-aware pooling to prevent representation collapse</li>
    <li>Frozen GPT backbone with trainable adapters</li>
    <li>Clean PyTorch-based training and inference pipeline</li>
    <li>GPU-compatible (CUDA / Google Colab)</li>
</ul>

<hr>

<h2>🧠 Model Architecture</h2>

<pre>
Input Text
   ↓
Tokenizer (padding + truncation)
   ↓
Input IDs (batch_size × sequence_length)
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

<h2>📦 Requirements</h2>
<ul>
    <li>Python ≥ 3.9</li>
    <li>PyTorch ≥ 2.0</li>
    <li>Transformers or a custom GPT implementation</li>
    <li>CUDA-compatible GPU (optional but recommended)</li>
</ul>

<hr>

<h2>🔤 Tokenization</h2>

<p>
Efficient batching is ensured with fixed-length padding:
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
The <code>pad_token_id</code> is explicitly used in pooling to avoid PAD-token leakage.
</p>

<hr>

<h2>🧩 GPTForClassification Wrapper</h2>

<p>
This wrapper around the GPT model performs:
</p>
<ul>
    <li>Extraction of token embeddings</li>
    <li>Padding-aware pooling</li>
    <li>Forwarding pooled embeddings into a classification head</li>
</ul>

<h3>Padding-aware Pooling (CRITICAL FIX)</h3>

<p><span class="highlight">Incorrect approach (causes collapse):</span></p>
<pre><code>
x = x[:, -1, :]
</code></pre>

<p><span class="highlight">Correct approach:</span></p>
<pre><code>
attention_mask = (input_ids != pad_token_id)
last_token_idx = attention_mask.sum(dim=1) - 1
x = x[torch.arange(x.size(0)), last_token_idx]
</code></pre>

<p>This ensures the model always uses the <strong>last real token</strong> rather than PAD.</p>

<hr>

<h2>🪜 LoRA: Low-Rank Adaptation</h2>

<p>
LoRA enables parameter-efficient fine-tuning by injecting trainable adapters into the <strong>Q, K, V attention projections</strong>.
</p>

<h3>LoRA Formula</h3>
<pre><code>
output = W(x) + scaling * (x @ A @ B)
</code></pre>

<ul>
    <li>Original GPT weights remain frozen</li>
    <li>Only LoRA matrices <code>A</code> and <code>B</code> are updated</li>
    <li>Significantly reduces trainable parameters</li>
</ul>

<hr>

<h2>❄️ Parameter Freezing Strategy</h2>

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
for name, p in clf_model.named_parameters():
    if "lora_" in name or "classifier" in name:
        p.requires_grad = True
</code></pre>

<p>Trainable parameters include <code>lora_A</code>, <code>lora_B</code>, and the classifier weights and biases.</p>

<hr>

<h2>⚙️ Optimizer</h2>

<p>
Pass only trainable parameters to the optimizer:
</p>

<pre><code>
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, clf_model.parameters()),
    lr=3e-4
)
</code></pre>

<p>This prevents unintended updates to frozen GPT weights.</p>

<hr>

<h2>🏋️ Training Loop</h2>

<pre><code>
clf_model.train()
for epoch in range(num_epochs):
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
    <li>Loss function: <code>CrossEntropyLoss</code></li>
    <li>Short training to prevent overfitting</li>
    <li>Fully GPU-compatible</li>
</ul>

<hr>

<h2>🔍 Inference</h2>

<pre><code>
def predict(text):
    clf_model.eval()
    ...
    pred = torch.argmax(logits, dim=-1).item()
    return "Positive" if pred == 1 else "Negative"
</code></pre>

<p>Predictions properly handle padding and batching.</p>

<hr>

<h2>🛑 Common Pitfall Solved</h2>

<p><strong>Issue:</strong> Model predicted only “Positive”.</p>

<p><strong>Cause:</strong></p>
<ul>
    <li>Using last token embedding</li>
    <li>Last token often PAD</li>
    <li>PAD embeddings are constant → representation collapse</li>
</ul>

<p><strong>Solution:</strong> Apply <code>padding-aware pooling</code> using <code>pad_token_id</code>.</p>

<hr>

<h2>✅ Key Takeaways</h2>
<ul>
    <li>Handling padding correctly is critical for GPT classifiers</li>
    <li>LoRA allows efficient fine-tuning on limited hardware</li>
    <li>Always verify which parameters are trainable</li>
    <li>Silent bugs can completely hinder learning</li>
</ul>

<hr>

<h2>📌 Future Extensions</h2>
<ul>
    <li>Support multi-class classification</li>
    <li>Implement evaluation metrics (Accuracy, F1)</li>
    <li>Save and load LoRA adapters</li>
    <li>Integrate with HuggingFace ecosystem</li>
</ul>

<hr>

<h2>👤 Author</h2>
<p>
Developed by <strong>Vineeth Benakashetty</strong><br>
Focus: Practical and efficient NLP model design.
</p>

<hr>

<h2>📜 License</h2>
<p>MIT License</p>

</body>
</html>
