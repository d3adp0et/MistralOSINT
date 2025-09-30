# OSINT-Recon LLM Project Setup Guide

## System Requirements Check
- ✅ RTX 3050 (4GB VRAM) - Perfect for Mistral 7B quantized
- ✅ Intel i5 11th Gen - Sufficient for inference
- ✅ RAM: Ensure you have at least 16GB total

## Project Structure
```
osint-recon-llm/
├── models/
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── data/
│   ├── raw/
│   ├── enriched/
│   └── vectors/
├── tools/
│   ├── recon_tools.py
│   └── osint_wrapper.py
├── embeddings/
│   └── embedding_manager.py
├── llm/
│   └── llm_engine.py
├── config/
│   └── config.yaml
├── outputs/
│   └── reports/
├── requirements.txt
├── main.py
└── README.md
```

## Step 1: Environment Setup

### 1.1 Create Virtual Environment
```bash
python -m venv osint_env
# On Windows:
osint_env\Scripts\activate
# On Linux/Mac:
source osint_env/bin/activate
```

### 1.2 Install Dependencies
Create `requirements.txt`:
```txt
# Core LLM
llama-cpp-python==0.2.90
huggingface-hub==0.24.5

# Embeddings & Vector DB
sentence-transformers==3.0.1
faiss-cpu==1.8.0

# OSINT Tools Support
dnspython==2.6.1
requests==2.32.3
aiohttp==3.10.5

# Data Processing
pandas==2.2.2
pyyaml==6.0.1
python-whois==0.9.4

# CLI & Reports
rich==13.7.1
click==8.1.7
jinja2==3.1.4
weasyprint==62.3

# Utilities
python-dotenv==1.0.1
tqdm==4.66.4
```

Install everything:
```bash
pip install -r requirements.txt

# Install llama-cpp with CUDA support for your RTX 3050
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

## Step 2: Install OSINT Tools

### 2.1 Install Go (Required for tools)
```bash
# Windows: Download from https://golang.org/dl/
# Linux:
wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/bin/go/bin' >> ~/.bashrc
```

### 2.2 Install Reconnaissance Tools
```bash
# Subfinder
go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest

# httpx
go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest

# dnsx
go install -v github.com/projectdiscovery/dnsx/cmd/dnsx@latest

# Add Go bin to PATH
export PATH=$PATH:$HOME/go/bin
```

## Step 3: Download Models

### 3.1 Download Mistral 7B Model
```python
# download_model.py
from huggingface_hub import hf_hub_download

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

print("Downloading Mistral 7B model...")
model_path = hf_hub_download(
    repo_id=model_name,
    filename=model_file,
    local_dir="./models",
    local_dir_use_symlinks=False
)
print(f"Model downloaded to: {model_path}")
```

### 3.2 Initialize Embedding Model
```python
# init_embeddings.py
from sentence_transformers import SentenceTransformer

print("Downloading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./models/embeddings/')
print("Embedding model ready!")
```

## Step 4: Core Implementation

### 4.1 Configuration File
Create `config/config.yaml`:
```yaml
llm:
  model_path: "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
  n_gpu_layers: 20  # Adjust based on your VRAM
  n_ctx: 4096
  temperature: 0.7
  max_tokens: 512

embeddings:
  model_name: "all-MiniLM-L6-v2"
  model_path: "./models/embeddings/"
  dimension: 384

vector_db:
  index_path: "./data/vectors/faiss.index"
  
osint_tools:
  subfinder:
    threads: 10
    timeout: 30
  httpx:
    threads: 50
    timeout: 10
  dnsx:
    threads: 100
```

### 4.2 LLM Engine
Create `llm/llm_engine.py`:
```python
from llama_cpp import Llama
import yaml
from typing import Dict, Any

class LLMEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['llm']
        
        self.llm = Llama(
            model_path=self.config['model_path'],
            n_gpu_layers=self.config['n_gpu_layers'],
            n_ctx=self.config['n_ctx'],
            verbose=False
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        response = self.llm(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.config['max_tokens']),
            temperature=kwargs.get('temperature', self.config['temperature']),
            stop=kwargs.get('stop', ["\n\n", "Human:", "User:"])
        )
        return response['choices'][0]['text'].strip()
    
    def analyze_recon(self, data: Dict[str, Any], task: str) -> str:
        """Analyze reconnaissance data"""
        prompt = f"""[INST] You are an OSINT analyst. Analyze the following reconnaissance data and {task}.

Data:
{data}

Provide a concise, actionable analysis focusing on security implications.
[/INST]

Analysis:"""
        
        return self.generate(prompt)
```

### 4.3 Embedding Manager
Create `embeddings/embedding_manager.py`:
```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

class EmbeddingManager:
    def __init__(self, model_path: str = "./models/embeddings/"):
        self.model = SentenceTransformer(model_path)
        self.dimension = 384
        self.index = None
        self.metadata = []
        self.load_or_create_index()
    
    def load_or_create_index(self):
        index_path = "./data/vectors/faiss.index"
        metadata_path = "./data/vectors/metadata.pkl"
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
    
    def add_documents(self, texts: list, metadata: list = None):
        """Add documents to vector store"""
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"text": t} for t in texts])
        
        self.save_index()
    
    def search(self, query: str, k: int = 5):
        """Search similar documents"""
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    'content': self.metadata[idx],
                    'distance': distances[0][i]
                })
        return results
    
    def save_index(self):
        """Save index and metadata"""
        os.makedirs("./data/vectors", exist_ok=True)
        faiss.write_index(self.index, "./data/vectors/faiss.index")
        with open("./data/vectors/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
```

### 4.4 OSINT Tools Wrapper
Create `tools/osint_wrapper.py`:
```python
import subprocess
import json
import asyncio
from typing import List, Dict

class OSINTTools:
    def __init__(self):
        self.results = {}
    
    async def run_subfinder(self, domain: str) -> List[str]:
        """Run subfinder for subdomain enumeration"""
        cmd = f"subfinder -d {domain} -silent -json"
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        subdomains = []
        for line in stdout.decode().strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    subdomains.append(data.get('host', ''))
                except:
                    pass
        return subdomains
    
    async def run_httpx(self, domains: List[str]) -> List[Dict]:
        """Probe domains with httpx"""
        input_data = '\n'.join(domains)
        cmd = "httpx -silent -json -follow-redirects -status-code -title -tech-detect"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate(input_data.encode())
        
        results = []
        for line in stdout.decode().strip().split('\n'):
            if line:
                try:
                    results.append(json.loads(line))
                except:
                    pass
        return results
    
    async def run_dnsx(self, domains: List[str]) -> List[Dict]:
        """Resolve DNS records"""
        input_data = '\n'.join(domains)
        cmd = "dnsx -silent -json -a -aaaa -cname -mx -txt -ns"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate(input_data.encode())
        
        results = []
        for line in stdout.decode().strip().split('\n'):
            if line:
                try:
                    results.append(json.loads(line))
                except:
                    pass
        return results
    
    async def full_recon(self, domain: str):
        """Run full reconnaissance pipeline"""
        print(f"[*] Starting recon for {domain}")
        
        # Subdomain enumeration
        print("[*] Enumerating subdomains...")
        subdomains = await self.run_subfinder(domain)
        print(f"[+] Found {len(subdomains)} subdomains")
        
        # HTTP probing
        print("[*] Probing HTTP services...")
        http_results = await self.run_httpx(subdomains)
        print(f"[+] {len(http_results)} services responded")
        
        # DNS resolution
        print("[*] Resolving DNS records...")
        dns_results = await self.run_dnsx(subdomains)
        
        return {
            'domain': domain,
            'subdomains': subdomains,
            'http_services': http_results,
            'dns_records': dns_results
        }
```

### 4.5 Main Application
Create `main.py`:
```python
#!/usr/bin/env python3
import asyncio
import click
import json
from rich.console import Console
from rich.table import Table
from datetime import datetime
import os

from llm.llm_engine import LLMEngine
from embeddings.embedding_manager import EmbeddingManager
from tools.osint_wrapper import OSINTTools

console = Console()

class OSINTReconLLM:
    def __init__(self):
        console.print("[bold green]Initializing OSINT-Recon LLM System...[/bold green]")
        self.llm = LLMEngine()
        self.embeddings = EmbeddingManager()
        self.osint = OSINTTools()
    
    async def recon_domain(self, domain: str):
        """Run reconnaissance on a domain"""
        # Gather OSINT data
        recon_data = await self.osint.full_recon(domain)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/recon_{domain}_{timestamp}.json"
        os.makedirs("outputs", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(recon_data, f, indent=2)
        
        # Store in vector DB
        console.print("[bold yellow]Storing in vector database...[/bold yellow]")
        for subdomain in recon_data['subdomains']:
            self.embeddings.add_documents(
                [f"Subdomain: {subdomain} for {domain}"],
                [{'type': 'subdomain', 'domain': domain, 'value': subdomain}]
            )
        
        for service in recon_data['http_services']:
            text = f"Service: {service.get('url', '')} Status: {service.get('status_code', '')} Title: {service.get('title', '')}"
            self.embeddings.add_documents(
                [text],
                [{'type': 'http_service', 'domain': domain, 'data': service}]
            )
        
        # Analyze with LLM
        console.print("[bold cyan]Analyzing with LLM...[/bold cyan]")
        analysis = self.llm.analyze_recon(
            recon_data,
            "identify potential security risks and interesting findings"
        )
        
        # Display results
        self.display_results(recon_data, analysis)
        
        # Generate report
        self.generate_report(domain, recon_data, analysis, timestamp)
        
        return recon_data, analysis
    
    def display_results(self, data, analysis):
        """Display results in terminal"""
        # Subdomains table
        table = Table(title="Discovered Subdomains")
        table.add_column("Subdomain", style="cyan")
        table.add_column("Status", style="green")
        
        for service in data['http_services'][:10]:  # Show first 10
            table.add_row(
                service.get('url', 'N/A'),
                str(service.get('status_code', 'N/A'))
            )
        
        console.print(table)
        
        # LLM Analysis
        console.print("\n[bold magenta]LLM Analysis:[/bold magenta]")
        console.print(analysis)
    
    def generate_report(self, domain, data, analysis, timestamp):
        """Generate HTML report"""
        report_template = """
        <html>
        <head>
            <title>OSINT Report - {domain}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }}
                .subdomain {{ margin: 5px 0; padding: 5px; background: #fff; }}
                pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>OSINT Reconnaissance Report</h1>
            <p><strong>Target:</strong> {domain}</p>
            <p><strong>Date:</strong> {date}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Discovered {subdomain_count} subdomains</p>
                <p>Found {service_count} active HTTP services</p>
            </div>
            
            <div class="section">
                <h2>AI Analysis</h2>
                <pre>{analysis}</pre>
            </div>
            
            <div class="section">
                <h2>Subdomains</h2>
                {subdomain_list}
            </div>
        </body>
        </html>
        """
        
        subdomain_html = "".join([f'<div class="subdomain">{s}</div>' for s in data['subdomains'][:50]])
        
        html_content = report_template.format(
            domain=domain,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            subdomain_count=len(data['subdomains']),
            service_count=len(data['http_services']),
            analysis=analysis,
            subdomain_list=subdomain_html
        )
        
        report_path = f"outputs/report_{domain}_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        console.print(f"[bold green]Report saved to: {report_path}[/bold green]")
    
    def search_knowledge(self, query: str):
        """Search the vector database"""
        results = self.embeddings.search(query, k=5)
        
        console.print(f"\n[bold cyan]Search results for: {query}[/bold cyan]")
        for i, result in enumerate(results, 1):
            console.print(f"{i}. {result['content']} (distance: {result['distance']:.4f})")
        
        return results

@click.command()
@click.option('--domain', '-d', help='Target domain for reconnaissance')
@click.option('--search', '-s', help='Search existing knowledge base')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
async def main(domain, search, interactive):
    """OSINT-Recon LLM System"""
    app = OSINTReconLLM()
    
    if domain:
        await app.recon_domain(domain)
    
    elif search:
        app.search_knowledge(search)
    
    elif interactive:
        console.print("[bold yellow]Interactive Mode[/bold yellow]")
        while True:
            try:
                choice = console.input("\n[1] Recon domain\n[2] Search knowledge\n[3] Exit\nChoice: ")
                
                if choice == '1':
                    domain = console.input("Enter domain: ")
                    await app.recon_domain(domain)
                elif choice == '2':
                    query = console.input("Search query: ")
                    app.search_knowledge(query)
                elif choice == '3':
                    break
            except KeyboardInterrupt:
                break
    
    else:
        console.print("Use --help for usage information")

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Running the System

### 5.1 First Run Setup
```bash
# Create necessary directories
mkdir -p models data/raw data/enriched data/vectors outputs config

# Download models (run once)
python download_model.py
python init_embeddings.py
```

### 5.2 Basic Usage
```bash
# Single domain recon
python main.py -d example.com

# Search knowledge base
python main.py -s "wordpress vulnerabilities"

# Interactive mode
python main.py -i
```

## Step 6: Creating Air-Gapped Environment

Since you asked about air-gapping:

### Option 1: Virtual Machine Isolation
```bash
# 1. Install VirtualBox or VMware
# 2. Create a Linux VM (Ubuntu recommended)
# 3. Disable network adapter after setup
# 4. Transfer files via shared folders only
```

### Option 2: Docker Container
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Disable network for container
CMD ["python", "main.py", "-i"]
```

Run with network isolation:
```bash
docker build -t osint-recon .
docker run --rm -it --network none osint-recon
```

## Next Steps for Expansion

1. **Add More Tools**:
   - Amass for deeper subdomain enum
   - WhatWeb for tech fingerprinting
   - theHarvester for email gathering

2. **Enhanced LLM Capabilities**:
   - Fine-tune on OSINT-specific data
   - Add conversation memory
   - Implement RAG for better context

3. **GitHub Integration**:
   - Add GitHub secret scanning
   - Repository analysis
   - Commit history mining

4. **Social Media OSINT**:
   - LinkedIn scraping
   - Twitter analysis
   - Profile correlation

5. **Advanced Features**:
   - Multi-agent system with CrewAI
   - Automated vulnerability correlation
   - Attack path generation

## Troubleshooting

### GPU Memory Issues
If you encounter CUDA out of memory:
```python
# Reduce n_gpu_layers in config.yaml
n_gpu_layers: 15  # Lower this value
```

### Tool Installation Issues
Ensure Go binaries are in PATH:
```bash
echo $PATH
which subfinder
```

### Model Loading Issues
Verify model file integrity:
```bash
ls -lh models/*.gguf
# Should show ~4GB file
```

## Security Notes

- Never run reconnaissance against targets without authorization
- Use VPN/proxies for any active scanning
- Store all data encrypted at rest
- Regularly clean old reconnaissance data
- Follow responsible disclosure for any findings

This setup gives you a solid foundation for OSINT operations with LLM integration, optimized for your RTX 3050 hardware.
