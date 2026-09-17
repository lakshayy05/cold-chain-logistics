# 🧊 Cold-Chain Logistics AI-Assistant

**An autonomous, conversational compliance agent for cold-chain fleet operations** — a Forward-Deployed Engineering (FDE) project that replaces static dashboards with a natural-language copilot that reasons over live telemetry, live weather, and versioned SOP documents, on top of a deliberately messy "legacy" enterprise database.

> Ask it a question like a dispatcher would ask a colleague — it plans, writes and runs the SQL itself, checks the rules, and answers in plain English. Every step it takes is logged.

---

## 📌 The Problem

Cold-chain operations lose money the moment a shipment drifts out of its temperature band. Catching that and reacting correctly today is a slow, manual chain of human effort:

1. A driver or IoT sensor detects a temperature drift.
2. A dispatcher manually queries multiple databases/dashboards.
3. They search a Standard Operating Procedure (SOP) document — which changes frequently — to figure out what the rule actually says.
4. Only then do they execute the correct escalation (Tier 1 self-resolve vs. Tier 2 hand-off to a Logistics Manager).

This is slow, error-prone, and depends on the dispatcher remembering the *current* version of the SOP. Dashboards show numbers — they don't tell you whether those numbers are a compliance breach.

**The goal:** collapse steps 2–4 into a single natural-language question, answered correctly, instantly, and with a full audit trail.

---

## 💡 The Solution

A dispatcher can type something like:

> *"Find any active shipments near Los Angeles (Latitude ~33.8, Longitude ~-118.1). Check the local weather there, and tell me if the current cargo temperature violates the SOP for fresh perishables."*

The agent:
- Writes and executes a **live SQL query** against the fleet database.
- Calls a **live weather API** for that corridor.
- Retrieves the **current SOP text** (temperature thresholds, breach definitions, escalation rules) from a vector knowledge base.
- **Reasons across all three** and gives a single, synthesized verdict — e.g. whether a shipment is in breach, and whether it's a Tier 1 or Tier 2 situation.

It also handles pure SOP/meta questions — e.g. *"I'm a new dispatcher on the night shift, what's the difference between a Tier 1 and Tier 2 escalation?"* — by retrieving from the SOP index directly, **without touching the database at all** (a deliberate "restraint" test: the agent should only call tools it actually needs).

---

## 🏗️ System Architecture

```
User (Dispatcher)
       │
       ▼
┌─────────────────────┐
│  Streamlit (ui.py)   │   ← Chat UI, session/thread state, audit-log writer
└─────────┬────────────┘
          ▼
┌──────────────────────────┐
│ LangGraph agent           │  ← 2-node ReAct graph: reasoner ⇄ tools
│ (orchestrator.py)         │
└─────────┬─────────────────┘
          ▼
┌─────────────────────┐
│   agent_tools.py      │  ← 3 tools bound to the LLM
└──┬────────┬──────────┬┘
   ▼        ▼          ▼
 MSSQL   Open-Meteo   Pinecone
(Docker,  Weather      (SOP
 EC2)     REST API     vectors)
```

**Component responsibilities**

| File | Responsibility |
|---|---|
| `src/ui.py` | Streamlit frontend. Renders chat, streams the agent's execution live (tool calls, tool outputs), writes every step to the SQL audit log, and hosts a separate admin-gated audit-log viewer. |
| `src/orchestrator.py` | Builds and compiles the LangGraph state machine (the "brain"). Configures which LLM backend to use, binds the tools, and runs the reasoning loop. |
| `src/agent_tools.py` | The three tools the agent can call: SQL query execution, live weather/corridor lookup, and SOP vector search. Also owns the embedding-model setup for the SOP retriever. |
| `src/prompts/system_prompt.txt` | The system prompt — strict rules the LLM must follow (never invent tool results, never fabricate escalation tiers, always compare against the SOP's *actual* returned thresholds, etc.). |
| `scripts/ingest_legacy_data.py` | One-time ETL: loads the Kaggle logistics dataset, maps it into a deliberately obscure "legacy enterprise" schema, and loads it into MSSQL. |
| `scripts/ingest_sop_pinecone.py` | Parses SOP documents (`.md`/`.txt`/`.pdf`/`.csv`/`.xlsx`) and upserts them into a Pinecone vector index, with hash-based incremental re-indexing. |
| `scripts/setup_security_and_view.sql` | Creates the clean semantic view + the least-privilege database role the agent runs as. |

---

## 🤖 Agent Orchestration — How the "Brain" Works

The orchestrator (`src/orchestrator.py`) is a **LangGraph** graph with exactly two nodes:

- **`reasoner`** — calls the LLM (with the three tools bound to it) and decides what to do next.
- **`tools`** — a `ToolNode` that actually executes whichever tool(s) the reasoner asked for.

```
START → reasoner → (tools_condition) → tools → reasoner → ... → END
```

`tools_condition` is LangGraph's built-in router: if the LLM's last message contains a tool call, it routes to `tools`; if the LLM produced a plain-text answer instead, the graph ends. This is what creates the ReAct-style "think → act → observe → think again" loop, entirely driven by the model's own decisions rather than a fixed script.

Conversation state is kept per-session using a `MemorySaver` checkpointer keyed on a `thread_id` — in the UI this is a fresh `uuid4` generated per browser session (reset via the "Purge Dispatch Workspace Session" button), so each dispatcher's conversation is isolated.

**The LLM backend is pluggable**, selected via the `Agent_llm` environment variable:
- `OLLAMA` (default) — runs a local model (`qwen2.5:3b` by default) via `ChatOllama`, so the whole thing can run with zero API cost.
- `OPENAI` — uses `ChatOpenAI` (`gpt-4o` by default).
- `DEEPSEEK` — uses DeepSeek's OpenAI-compatible API.

All three are configured with `temperature=0` for deterministic, repeatable reasoning.

### The system prompt does real work

`src/prompts/system_prompt.txt` is not a generic "be helpful" prompt — it's a fairly strict rulebook, for example:
- Never claim "no shipments exist" just because a query returned zero rows — distinguish *no records found* vs *telemetry unavailable* vs *tool/API failure*.
- Never declare cargo "SOP-compliant" without an actual temperature reading.
- Use only the temperature thresholds actually returned by the SOP search — never invent Fahrenheit values or made-up escalation tiers.
- For Tier 1/Tier 2 escalation questions, search the SOP first; if no escalation matrix is found, say so rather than guessing.

This is what stops the demo's two test questions from being answered by hallucination instead of by actually calling tools.

---

## 🛠️ How the Agent Actually Writes and Runs SQL

This is the core "agent does real work" piece of the project. The tool is defined like this (simplified):

```python
@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the FDE_VIEWS.VW_ACTIVE_FLEET view.
    Columns available: Timestamp, Latitude, Longitude, Current_Temperature_C,
    Cargo_Condition_Code, Risk_Classification, Delay_Probability,
    Port_Congestion_Level, Route_Risk_Index.
    """
```

The LLM is told exactly what columns exist on the view, and it is the one **writing the raw T-SQL text itself** — there's no hardcoded query template per question type. When a dispatcher asks "find shipments near Los Angeles," the model composes a `SELECT ... WHERE Latitude BETWEEN ... AND Longitude BETWEEN ...`-style query on its own and passes it as the tool's argument.

Because the agent gets to write arbitrary SQL text, safety comes from **three independent layers**, not from trusting the prompt:

1. **App-level guardrail** — the tool itself refuses to execute anything that doesn't start with `SELECT` (`SECURITY BLOCK: Only SELECT operations are authorized on this view.`), and the connection only ever fetches the first 10 rows.
2. **Schema-level guardrail** — the query is only ever pointed at `FDE_VIEWS.VW_ACTIVE_FLEET`, a curated view with a handful of flat, renamed columns — the agent has no idea the underlying legacy table (or its cryptic column names) even exists.
3. **Database-level guardrail (the one that actually matters)** — the agent connects as a SQL login, `USR_FDE_RO`, that is granted `SELECT` **only** on that one view and is explicitly `DENY`'d `SELECT` on the raw table and `DENY`'d `INSERT/UPDATE/DELETE/ALTER` on the entire schema. So even if a prompt-injection attempt convinced the model to try `DROP TABLE` or to read the raw legacy table directly, the database itself would refuse the command — the security boundary doesn't rely on the LLM behaving.

This is set up in `scripts/setup_security_and_view.sql`:

```sql
CREATE SCHEMA FDE_VIEWS;

CREATE VIEW FDE_VIEWS.VW_ACTIVE_FLEET AS
SELECT
    TS_UTC AS [Timestamp], V_LAT AS [Latitude], V_LON AS [Longitude],
    CAST(IOT_TEMP_VAL_C AS FLOAT) AS [Current_Temperature_C],
    CGO_COND_CD AS [Cargo_Condition_Code], RISK_CLS_TXT AS [Risk_Classification],
    DELAY_PROB_DEC AS [Delay_Probability], PRT_CNG_LVL AS [Port_Congestion_Level],
    RT_RSK_IDX AS [Route_Risk_Index]
FROM dbo.TBL_SC_FLEET_HIST_RAW;

CREATE LOGIN USR_FDE_RO WITH PASSWORD = 'AgentPassword2026!';
CREATE USER USR_FDE_RO FOR LOGIN USR_FDE_RO;

GRANT SELECT ON FDE_VIEWS.VW_ACTIVE_FLEET TO USR_FDE_RO;
DENY SELECT ON dbo.TBL_SC_FLEET_HIST_RAW TO USR_FDE_RO;
DENY INSERT, UPDATE, DELETE, ALTER ON SCHEMA::dbo TO USR_FDE_RO;
```

One deliberate exception is carved out later: `USR_FDE_RO` is additionally granted `INSERT` on exactly one table, `FDE_VIEWS.AgentAuditLog` — nothing else — so the same low-privilege agent connection can *write its own audit trail* without ever gaining read/write access to anything it shouldn't touch. That's the least-privilege model in one sentence: read-only everywhere, with one narrow, explicit, audited write exception.

---

## 🌦️ The Other Two Tools

**`fetch_corridor_conditions(latitude, longitude)`** — calls the free [Open-Meteo](https://open-meteo.com/) REST API for live temperature and wind speed at a given GPS point, then derives a simple corridor congestion index/status from wind speed (e.g. high wind → "High Transit Disruption"). If the API call fails, it returns an explicit failure string rather than fabricating a reading — matching the system prompt's "never invent tool results" rule.

**`search_compliance_sop(query)`** — a Pinecone-backed vector search (`retriever.invoke(query)`, top-`k=2`) over the ingested SOP documents. Each retrieved chunk carries its `source_file` and `file_format` in the response, which is what lets the UI show *which* SOP document and section backed a given answer (visible in the "View Raw Output" traces in the demo).

The SOP retriever's embedding model is also environment-driven (`Embeddings_model=OPENAI` vs the local default), with two separate, isolated Pinecone indexes (`fde-sop-index-openai`, 1536-dim vs `fde-sop-index-local`, 1024-dim using `BAAI/bge-m3`) so switching providers never mixes incompatible vector spaces.

---

## 🔐 Data & Security Model

Because an LLM agent is being given the ability to compose its own SQL, security is enforced at the **database layer itself**, not just in application code — the assumption is that a prompt-injection attempt should still fail even if it somehow gets past the agent's own logic.

1. **Least-privilege role** — `USR_FDE_RO` is read-only everywhere except one explicitly-granted `INSERT` into the audit table.
2. **View-based isolation** — the agent only ever sees `FDE_VIEWS.VW_ACTIVE_FLEET`, a clean, renamed, flattened view — it never sees, and is explicitly denied access to, the raw `dbo.TBL_SC_FLEET_HIST_RAW` table underneath.
3. **Immutable audit logging** — every reasoning step is written to an append-only table:

   ```sql
   CREATE TABLE FDE_VIEWS.AgentAuditLog (
       LogID INT IDENTITY(1,1) PRIMARY KEY,
       Timestamp DATETIME DEFAULT GETDATE(),
       SessionID VARCHAR(50),
       NodeExecuted VARCHAR(50),
       ToolName VARCHAR(100),
       Content NVARCHAR(MAX)
   );
   ```

   The Streamlit app writes a row here every time the reasoner requests a tool call, every time a tool returns output, and every time the reasoner produces a final answer — capturing `SessionID`, which graph node ran, which tool fired, and the raw payload. Failures to write the log are swallowed silently so a logging hiccup never breaks the dispatcher's experience.
4. **Separate admin boundary for reading the logs** — the "Security & Audit Logs" tab in the UI is gated behind its *own* login form, checked against a completely separate `SQL_ADMIN_USER` / `SQL_ADMIN_PASSWORD` set in `.env` — the everyday agent credentials (`USR_FDE_RO`) cannot be used to *read* the audit log, only to append to it. Reading the trail requires a genuine admin credential.

---

## ☁️ Deployment — Docker + AWS

- **Database:** MS SQL Server 2022 runs **inside a Docker container** (`mcr.microsoft.com/mssql/server:2022-latest`), with a named volume (`mssql_data`) so data survives container restarts:

  ```bash
  docker run -v mssql_data:/var/opt/mssql \
    -e "ACCEPT_EULA=Y" \
    -e "MSSQL_SA_PASSWORD=..." \
    -p 1433:1433 \
    --name legacy-mssql \
    -d mcr.microsoft.com/mssql/server:2022-latest
  ```

  This container runs on an **AWS EC2 instance** (`c7i-flex.large`, Ubuntu, 30 GB storage), with its Security Group restricting inbound access on port 1433.

- **App:** the Streamlit app (`src/ui.py`) runs on the same or a separate EC2 instance as a **systemd service**, so it survives reboots and restarts automatically:

  ```ini
  [Unit]
  Description=Streamlit Cold-Chain Dispatch Console
  After=network.target

  [Service]
  User=ubuntu
  WorkingDirectory=/home/ubuntu/cold-chain-logistics-FDE-Project
  ExecStart=/home/ubuntu/cold-chain-logistics-FDE-Project/venv/bin/streamlit run src/ui.py --server.port=8501 --server.address=0.0.0.0
  Restart=always

  [Install]
  WantedBy=multi-user.target
  ```

  The Microsoft ODBC Driver 18 for SQL Server (`msodbcsql18`) and `unixodbc-dev` are installed on this box so `pyodbc`/SQLAlchemy can talk to the SQL Server container.

- **CI/CD:** `.github/workflows/deploy.yml` defines a manually-triggered (`workflow_dispatch`) GitHub Actions job that SSHes into the EC2 app server (via `appleboy/ssh-action`, using `EC2_HOST` / `EC2_USER` / `EC2_SSH_KEY` repo secrets), pulls the latest code, and restarts the `streamlit` systemd service — a lightweight "push-button" deploy rather than a fully automated pipeline.

---

## 📥 Data Ingestion

### 1. Fleet telemetry (`scripts/ingest_legacy_data.py`)

The project intentionally simulates a **messy legacy enterprise system** to make the security/semantic-view story meaningful. It takes a public Kaggle dataset (`dynamic_supply_chain_logistics_dataset.csv`, source noted in `data/source/data.txt`) and remaps clean column names into cryptic legacy-style codes before loading it into SQL Server:

| Clean source column | Legacy column it becomes |
|---|---|
| `timestamp` | `TS_UTC` |
| `vehicle_gps_latitude` / `longitude` | `V_LAT` / `V_LON` |
| `iot_temperature` | `IOT_TEMP_VAL_C` |
| `cargo_condition_status` | `CGO_COND_CD` |
| `risk_classification` | `RISK_CLS_TXT` |
| `delay_probability` | `DELAY_PROB_DEC` |
| `port_congestion_level` | `PRT_CNG_LVL` |
| `route_risk_level` | `RT_RSK_IDX` |

This lands in `dbo.TBL_SC_FLEET_HIST_RAW` — the "legacy junk" table that `VW_ACTIVE_FLEET` later translates back into clean, human-readable column names for the agent.

### 2. SOP knowledge base (`scripts/ingest_sop_pinecone.py`)

SOP documents live in `data/policy/` (the repo ships two dated versions, `Cold_Chain_Incident_SOP_v2.md` and `_v3.md`, so the agent can be shown answering against whichever is currently "live"). The ingestion script:
- Supports **`.md`, `.txt`, `.pdf`, `.csv`, and `.xlsx`** inputs through a single polymorphic parser — Markdown is split by header (`#`/`##`/`###`) before chunking, PDFs are parsed page-by-page with `pypdf`, and tabular files are flattened row-by-row into text.
- Does **incremental, hash-based re-indexing** — an MD5 hash of each file's bytes is cached in `data/cache/ingestion_hash_cache.json`, so unchanged SOP files are skipped on every run, changed files are deleted and re-upserted, and files removed from disk are purged from the vector index automatically.
- Upserts into **Pinecone** (serverless, AWS `us-east-1`) in batches of 100, tagging every chunk with `source_file`, `file_format`, and `document_type` metadata — which is exactly what powers the "[Source: Cold_Chain_Incident_SOP_v3.md | Format: MD]" citations visible in the tool-output traces.

Because the SOP lives in a retrievable vector index rather than the model's own memory, **updating a compliance rule means editing the markdown file and re-running the ingestion script — not retraining or re-prompting the agent.**

---

## 🖥️ Demo Walkthrough

The screenshots below show the actual system running two live queries and inspecting its own audit trail.

**1. Multi-tool compliance query** — *"Find any active shipments near Los Angeles... check the local weather... does the cargo temperature violate the SOP for fresh perishables?"* The agent recognizes the intent, calls `search_compliance_sop` and `fetch_corridor_conditions`, and synthesizes a single answer: no active shipments in that radius, corridor temperature 21.3°C / wind 4.8 km/h (Risk Index 2.5/10, normal), and — critically — it compares that reading against the SOP's 0.0°C–4.0°C fresh-perishables threshold to correctly report no violation.

**2. SOP explainer, no database touched** — *"I'm a new dispatcher on the night shift, what's the difference between a Tier 1 and Tier 2 escalation?"* The agent answers directly from general reasoning/SOP context without needing to call `query_telemetry_db` at all — demonstrating that it doesn't reach for a tool it doesn't need.

**3. Enterprise Agent Audit Trail** — the admin-gated "Security & Audit Logs" view queries `AgentAuditLog` directly, showing every reasoning step and tool call (`fetch_corridor_conditions`, `search_compliance_sop`) with its raw payload/output, timestamp, and session token — proving the audit trail described above works end-to-end.

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Agent Orchestration | LangGraph (`StateGraph`, `ToolNode`, `tools_condition`, `MemorySaver`) |
| LLM Reasoning | Pluggable — local Ollama (`qwen2.5:3b`), OpenAI (`gpt-4o`), or DeepSeek |
| Structured Data | Microsoft SQL Server 2022 (containerized via Docker) |
| SOP / Unstructured Knowledge | Pinecone (serverless vector DB) + OpenAI or local HuggingFace (`BAAI/bge-m3`) embeddings |
| External Data | Open-Meteo weather REST API |
| Database Access | SQLAlchemy + `pyodbc` (ODBC Driver 18 for SQL Server) |
| Infrastructure | AWS EC2 (Ubuntu), systemd service |
| CI/CD | GitHub Actions (manual `workflow_dispatch` + SSH deploy) |
| Auditing | Append-only SQL table (`FDE_VIEWS.AgentAuditLog`) |

---

## 📂 Repository Structure

```
cold-logistic-
├── .github/workflows/deploy.yml       # Manual SSH deploy to EC2
├── .streamlit/config.toml             # Disables the file watcher (production setting)
├── Misc/Materials/                    # Business presentation + Technical Design Document (PDFs)
├── data/
│   ├── cache/ingestion_hash_cache.json
│   ├── policy/                        # SOP source docs (v2, v3)
│   ├── raw/dynamic_supply_chain_logistics_dataset.csv
│   └── source/data.txt                # Dataset provenance (Kaggle link)
├── docs/instrutions.md                # Full phase-by-phase build/run instructions
├── scripts/
│   ├── ingest_legacy_data.py          # CSV → messy "legacy" MSSQL table
│   ├── ingest_sop_pinecone.py         # SOP docs → Pinecone vector index
│   └── setup_security_and_view.sql    # View + least-privilege role + DENY rules
├── src/
│   ├── agent_tools.py                 # The 3 tools (SQL, weather, SOP search)
│   ├── orchestrator.py                # LangGraph agent definition + CLI chat loop
│   ├── ui.py                          # Streamlit dispatch console + audit log viewer
│   └── prompts/system_prompt.txt      # Agent rulebook
├── requirements.txt
└── LICENSE
```

---

## 🚀 Getting Started

Full step-by-step setup (EC2, Docker, Pinecone, security script, in that order) is documented in [`docs/instrutions.md`](./docs/instrutions.md). Short version:

```bash
# 1. Clone and install
git clone https://github.com/lakshayy05/cold-logistic-.git
cd cold-logistic-
pip install -r requirements.txt

# 2. Spin up MSSQL in Docker (locally or on an EC2 instance)
docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=<your-sa-password>" \
   -p 1433:1433 --name legacy-mssql -d mcr.microsoft.com/mssql/server:2022-latest

# 3. Configure your .env (DB host/creds, PINECONE_API_KEY, LLM provider keys, etc.)

# 4. Load the "legacy" fleet data
python scripts/ingest_legacy_data.py

# 5. Lock down the database (creates VW_ACTIVE_FLEET + USR_FDE_RO)
#    Run scripts/setup_security_and_view.sql against your SQL Server instance

# 6. Ingest the SOP documents into Pinecone
python scripts/ingest_sop_pinecone.py

# 7. (One-time, via a SQL client) create FDE_VIEWS.AgentAuditLog and
#    GRANT INSERT on it to USR_FDE_RO — see docs/instrutions.md Phase 4

# 8. Run the agent
streamlit run src/ui.py
```

Required `.env` variables (non-exhaustive — see the scripts for the full list): `SQL_SERVER_HOST`, `SQL_SERVER_PORT`, `SQL_AGENT_USER`, `SQL_AGENT_PASSWORD`, `SQL_ADMIN_USER`, `SQL_ADMIN_PASSWORD`, `PINECONE_API_KEY`, `Embeddings_model`, `Agent_llm`, plus the relevant LLM provider key (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or an `OLLAMA_BASE_URL` for a local model).

---

## 📄 Further Reading

- **Business Presentation** — problem statement, operational bottleneck, and solution overview for non-technical stakeholders.
- **Technical Design Document (TDD)** — architecture (HLD/LLD), agent orchestration, data & security model, and deployment strategy.

Both are included in [`/Misc/Materials`](./Misc/Materials).

---

## 📃 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.
