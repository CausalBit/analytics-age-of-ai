# Steps 10–14 — Agent, Coding CLI, Genie, Unity AI Gateway

Steps 10–14 are largely no-code / UI-driven / CLI-driven, so they live here as an
agent definition plus a walkthrough rather than notebook cells.

> All commands/keys touching Omnigent, ucode, Lakebase, Genie, and Unity AI Gateway
> are marked `[VERIFY AGAINST CURRENT DOCS]` where the exact syntax may drift — these
> are Beta/Preview products in 2026. Sources are in `../knowledge-base/`.

---

## Step 10 — Omnigent: the churn-explainer agent
See [`churn_explainer.yaml`](churn_explainer.yaml). Omnigent is a meta-harness over
Claude Code, Codex, Cursor, Pi, and custom agents — **change the harness or model in
one line**, keeping tools/prompts/skills/policies constant. The agent reasons over the
served churn score (Step 6) + feedback retrieved from Vector Search (Step 7) to answer
"why is this customer at risk?"

Replaces the deprecated **Mosaic AI Agent Framework**. KB: `omnigent.md`.

---

## Step 11 — ucode: launch the coding-agent session
`ucode` (Unity AI Gateway Coding CLI) installs, authenticates, and configures coding
agents, routing all model traffic through Unity AI Gateway (no raw API keys). It
auto-registers Unity Catalog functions, the Vector Search index, and SQL warehouses as
MCP tools. KB: `ucode-coding-agent-integration.md`, `ucode-model-provider-services.md`,
`ucode-github-repo.md`.

```bash
# Install (Python 3.12+, uv)
uv tool install git+https://github.com/databricks/ucode

# Launch Claude Code as the harness for our agent, first-run prompts for workspace + auth
ucode claude

# Route through a governed model provider service instead of Databricks-hosted models
ucode claude --provider main.default.anthropic_prod   # [VERIFY AGAINST CURRENT DOCS]

# Register MCP tools (UC functions, Vector Search, SQL warehouses) for the session
ucode configure mcp

# Register reusable skills scoped to our project + UC location
ucode configure skills --location main.telco_churn --path .

# Check what's wired up + 7-day usage
ucode status
ucode usage
```

---

## Step 12 — Genie Spaces + Genie One (UI-driven)
KB: `genie-one-chat.md`.

1. **Create a Genie Space** over the churn tables (`main.telco_churn.customers`,
   `customer_features`). Define trusted metrics (e.g. *churn rate = churned / total*)
   and business rules / synonyms so answers are governed.
2. Publish it as a **Genie Agent**.
3. In **Genie One** (the unified full-screen chat), a business user asks:
   *"Which contract type has the highest churn?"* Genie One first searches Genie Agents
   (matches our Space), then falls back to dashboards/queries/metric views if unmatched.
4. Optional: connect external sources (Drive, Confluence, Slack) and turn a good
   conversation into a reusable Genie Agent.

Naming: **Genie → Genie One**; **Genie Space → Genie Agent-enabled Genie Space**.

---

## Step 13 — Genie Code (in-workspace assistant)
KB: `genie-code.md`. While building steps 1–12, developers get inline AI help directly
in notebooks, the SQL editor, Lakeflow Pipelines Editor, AI/BI dashboards, and MLflow:
chat with doc citations, inline autocomplete, automatic error fixes, `/` slash commands,
and natural-language data filtering. Respects Unity Catalog permissions. *Demo it live:*
trigger an error in a cell and use the "diagnose/fix" action.

---

## Step 14 — Unity AI Gateway: govern the AI services
KB: `unity-ai-gateway.md`, `unity-ai-gateway-summit-2026-blog.md`.
Replaces the deprecated **Mosaic AI Gateway**.

Register as governed AI services (Unity Catalog securables):
- the **Model Serving** endpoint (`telco-churn-endpoint`),
- the **Vector Search** index (`feedback_index`),
- the **Omnigent** coding agent / its model provider service.

Then configure (UI):
- **Rate limiting** per user/group on model + MCP services
- **Guardrails** via service policies (built-in + Contextual Service Policies Beta, e.g.
  approval workflow before code pushes / data modifications)
- **Inference-table logging** to Delta for usage/cost attribution by user, team, tool

---

## Deliverable E — Naming migration callout
| Old term | 2026 term |
|---|---|
| Mosaic AI Gateway | Unity AI Gateway |
| Mosaic AI Agent Framework | Omnigent |
| Genie | Genie One |
| Genie Space | Genie Agent-enabled Genie Space |
