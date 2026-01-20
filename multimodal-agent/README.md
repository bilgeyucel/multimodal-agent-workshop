# 🖼️ Multimodal Haystack Agent
This demo shows how to build a multimodal agent that can interact with images using [Haystack](https://haystack.deepset.ai/) and [Hayhooks](https://github.com/deepset-ai/hayhooks).

## 🚀 Run the Demo

### 1️⃣ Clone the Repository
```sh
git clone git@github.com:bilgeyucel/multimodal-agent-workshop.git
cd multimodal-agent-workshop
```
### 2️⃣ Install Dependencies(Python>=3.10)
Create and activate a virtual environment (optional but recommended):
```sh
python3.11 -m venv .venv
source .venv/bin/activate
```
Install Haystack, Hayhooks and other required packages:
```sh
pip install -q "haystack-ai>=2.22.0" "sentence-transformers>=4.1.0" "haystack-experimental>=0.16.0" pypdf pypdfium2 markdown-it-py
```

### 3️⃣ Configure Environment Variables
Set API Keys:
```sh
export OPENAI_API_KEY=your_openai_api_key # For OpenAI models
```

🔗 **More details on configuration:** [Hayhooks Documentation](https://github.com/deepset-ai/hayhooks?tab=readme-ov-file#configuration)

### 4️⃣ Start the Hayhooks Server 
```sh
hayhooks run
```
Check if Hayhooks is running:
```sh
hayhooks status
```
Output:
```sh
╭─────────────────────────────────────────────────────────────╮
│ Hayhooks server is up and running at: http://localhost:1416 │
╰─────────────────────────────────────────────────────────────╯

No pipelines currently deployed
```

### 5️⃣ Deploy the Agent
Deploy the agent by giving a name and the path:

```sh
hayhooks pipeline deploy-files -n deploy-files -n multimodal_agent multimodal-agent
```
If deployment is successful, you'll see output like this when you run `hayhooks status` again:
```sh
╭─────────────────────────────────────────────────────────────╮
│ Hayhooks server is up and running at: http://localhost:1416 │
╰─────────────────────────────────────────────────────────────╯

           Deployed Pipelines           
╭───┬──────────────────────┬───────────╮
│ № │ Pipeline Name        │ Status    │
├───┼──────────────────────┼───────────┤
│ 1 │ multimodal_agent     │ 🟢 Active │
╰───┴──────────────────────┴───────────╯
```

After making changes on the `pipeline_wrapper.py` files, you can redeploy the same agent with the `--overwrite` command without restarting the Hayhooks server.
```sh
hayhooks pipeline deploy-files -n multimodal_agent --overwrite multimodal-agent
```

### 6️⃣ Test the API
Swagger docs: [http://localhost:1416/docs](http://localhost:1416/docs)

Test with `curl`:
```sh
curl -X 'POST' \
  'http://localhost:1416/multimodal_agent/run' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Can I reimburse this receipt from my social budget?",
  "image_path": "./files/receipt.jpeg"
}'
```