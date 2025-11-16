## Installation

```bash
git clone https://github.com/Circuit-Overtime/moeJson.git
cd moeJson

pip install -r requirements.txt
```

# Download model
```bash
chmod +x model_installation.sh
./model_installation.sh
```

## Running

### Start Model Servers

```bash
python api/model_server.py 7002
python api/model_server.py 7003
```

### Start API

```bash
python api/app.py
```

## Usage

```bash
curl "http://localhost:9000/gen?prompt=Generate%20image%20of%20sunset"

# POST request
curl -X POST http://localhost:9000/gen \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Generate image of sunset"}'

curl http://localhost:9000/health
```

## Architecture

```mermaid
graph TD
        A[Client Request] -->|GET/POST| B[app.py Quart Server]
        B -->|validate prompt| C{Word Count < 100?}
        C -->|no| D[Error Response]
        C -->|yes| E[get_available_model]
        E -->|random load balance| F{Model Server Pool}
        F -->|port 7002| G[ModelServer 1]
        F -->|port 7003| H[ModelServer 2]
        G -->|IPC BaseManager| I[ModelManager]
        H -->|IPC BaseManager| J[ModelManager]
        I -->|tokenize| K[Llama Model]
        J -->|tokenize| L[Llama Model]
        K -->|inference| M[JSON Extraction]
        L -->|inference| M
        M -->|extract_json| N[Response to Client]
        B -->|/health| O[Connected Models Count]
```
