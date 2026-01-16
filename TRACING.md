# OpenTelemetry Tracing for SWE-agent

This fork of SWE-agent includes OpenTelemetry instrumentation for tracing agent runs, LLM calls, and tool executions.

## Quick Start

### 1. Install OpenTelemetry dependencies

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

### 2. Enable tracing

Set environment variables:

```bash
export SWE_AGENT_ENABLE_TRACING=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

### 3. Start the OpenTelemetry Collector

```bash
# Download the collector
curl -LO https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.96.0/otelcol_0.96.0_darwin_arm64.tar.gz
tar -xzf otelcol_0.96.0_darwin_arm64.tar.gz

# Start with Parseable config
./otelcol --config otel-collector-config.yaml
```

### 4. Run SWE-agent

```bash
python -m sweagent run --config config/default.yaml --problem_statement "Fix the bug in main.py"
```

## What Gets Traced

### Agent Invocation Span (`invoke_agent`)

The root span for each agent run:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | `invoke_agent` |
| `gen_ai.agent.name` | `SWE-agent` |
| `gen_ai.conversation.id` | Unique session ID |
| `agent.problem_id` | Problem/issue being solved |
| `gen_ai.request.model` | Model being used |
| `gen_ai.usage.input_tokens` | Total input tokens |
| `gen_ai.usage.output_tokens` | Total output tokens |
| `agent.success` | Whether task completed |
| `agent.iterations` | Number of LLM calls |

### LLM Call Spans (`chat`)

Each call to the language model:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | LLM provider (e.g., `openai`) |
| `gen_ai.request.model` | Model name |
| `gen_ai.request.temperature` | Temperature setting |
| `gen_ai.usage.input_tokens` | Prompt tokens |
| `gen_ai.usage.output_tokens` | Completion tokens |
| `gen_ai.response.model` | Actual model used |
| `gen_ai.response.finish_reasons` | Why generation stopped |

### Tool Execution Spans (`execute_tool`)

Each command executed in the environment:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | `execute_tool` |
| `gen_ai.tool.name` | Command name (e.g., `edit`, `cat`) |
| `gen_ai.tool.call.id` | Unique call ID |
| `gen_ai.tool.call.arguments` | Command arguments |
| `gen_ai.tool.call.result` | Command output (truncated) |

## Trace Hierarchy

A typical trace looks like:

```
invoke_agent SWE-agent (45.2s)
├── chat gpt-4o (3.1s)
│   └── [tokens: 2100 in, 450 out]
├── execute_tool cat (0.1s)
│   └── [args: {"command": "cat main.py"}]
├── chat gpt-4o (2.8s)
│   └── [tokens: 3200 in, 380 out]
├── execute_tool edit (0.2s)
│   └── [args: {"command": "edit 15:20 ..."}]
├── chat gpt-4o (2.1s)
│   └── [tokens: 3800 in, 120 out]
├── execute_tool python (1.5s)
│   └── [args: {"command": "python test.py"}]
└── execute_tool submit (0.1s)
```

## Querying Traces in Parseable

### Agent runs with token usage

```sql
SELECT
  trace_id,
  "agent.problem_id" AS problem,
  "gen_ai.usage.input_tokens" AS input_tokens,
  "gen_ai.usage.output_tokens" AS output_tokens,
  "agent.iterations" AS iterations,
  "agent.success" AS success,
  span_duration_ms / 1000.0 AS duration_seconds
FROM "swe-agent-traces"
WHERE "gen_ai.operation.name" = 'invoke_agent'
ORDER BY p_timestamp DESC;
```

### LLM calls breakdown

```sql
SELECT
  trace_id,
  COUNT(*) AS llm_calls,
  SUM("gen_ai.usage.input_tokens") AS total_input_tokens,
  SUM("gen_ai.usage.output_tokens") AS total_output_tokens,
  AVG(span_duration_ms) AS avg_latency_ms
FROM "swe-agent-traces"
WHERE "gen_ai.operation.name" = 'chat'
GROUP BY trace_id;
```

### Tool usage

```sql
SELECT
  "gen_ai.tool.name" AS tool,
  COUNT(*) AS call_count,
  AVG(span_duration_ms) AS avg_duration_ms
FROM "swe-agent-traces"
WHERE "gen_ai.operation.name" = 'execute_tool'
GROUP BY tool
ORDER BY call_count DESC;
```

### Cost estimation

```sql
SELECT
  DATE_TRUNC('day', p_timestamp) AS day,
  COUNT(DISTINCT trace_id) AS agent_runs,
  SUM("gen_ai.usage.input_tokens") AS input_tokens,
  SUM("gen_ai.usage.output_tokens") AS output_tokens,
  ROUND(
    SUM("gen_ai.usage.input_tokens") * 0.0000025 +
    SUM("gen_ai.usage.output_tokens") * 0.00001,
    4
  ) AS estimated_cost_usd
FROM "swe-agent-traces"
WHERE "gen_ai.operation.name" = 'chat'
GROUP BY day
ORDER BY day DESC;
```

## Privacy

By default, the instrumentation captures:
- Model names and parameters
- Token counts
- Command names and truncated arguments/results

To disable tracing entirely:

```bash
export SWE_AGENT_ENABLE_TRACING=false
```
