"""OpenTelemetry instrumentation for SWE-agent.

This module provides tracing for agent invocations, LLM calls, and tool executions
following OpenTelemetry GenAI semantic conventions.

To enable tracing, set the following environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    SWE_AGENT_ENABLE_TRACING=true
"""

import json
import os
import uuid
from functools import wraps
from typing import Any, Callable

# Check if tracing is enabled before importing OpenTelemetry
TRACING_ENABLED = os.getenv("SWE_AGENT_ENABLE_TRACING", "false").lower() == "true"

if TRACING_ENABLED:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, SpanKind
else:
    trace = None
    SpanKind = None
    Status = None
    StatusCode = None


_tracer = None
_conversation_id = None


def setup_telemetry(service_name: str = "swe-agent") -> None:
    """Initialize OpenTelemetry with OTLP exporter."""
    global _tracer, _conversation_id
    
    if not TRACING_ENABLED:
        return
    
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })
    
    provider = TracerProvider(resource=resource)
    
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    
    exporter = OTLPSpanExporter(
        endpoint=f"{otlp_endpoint}/v1/traces",
    )
    
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    
    _tracer = trace.get_tracer(__name__)
    _conversation_id = f"conv_{uuid.uuid4().hex[:12]}"


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None and TRACING_ENABLED:
        setup_telemetry()
    return _tracer


def get_conversation_id() -> str:
    """Get the current conversation ID."""
    global _conversation_id
    if _conversation_id is None:
        _conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    return _conversation_id


def reset_conversation_id() -> str:
    """Reset the conversation ID for a new agent run."""
    global _conversation_id
    _conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    return _conversation_id


def trace_llm_call(
    model_name: str,
    provider: str = "litellm",
    temperature: float = 0.0,
    capture_content: bool = False,
):
    """Decorator to trace LLM calls following GenAI semantic conventions.
    
    Args:
        model_name: The model being called
        provider: The LLM provider name
        temperature: The temperature setting
        capture_content: Whether to capture input/output messages (privacy sensitive)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(
                f"chat {model_name}",
                kind=SpanKind.CLIENT
            ) as span:
                # Set GenAI semantic convention attributes
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.provider.name", provider)
                span.set_attribute("gen_ai.request.model", model_name)
                span.set_attribute("gen_ai.request.temperature", temperature)
                span.set_attribute("gen_ai.conversation.id", get_conversation_id())
                
                # Capture input messages if enabled
                if capture_content and len(args) > 1:
                    messages = args[1] if len(args) > 1 else kwargs.get("messages", [])
                    if messages:
                        span.set_attribute("gen_ai.input.messages", json.dumps(messages[:3]))  # Truncate
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Extract token usage if available
                    if isinstance(result, list) and len(result) > 0:
                        span.set_attribute("gen_ai.response.finish_reasons", '["stop"]')
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_agent_step(agent_name: str = "SWE-agent"):
    """Decorator to trace agent step execution.
    
    Args:
        agent_name: Name of the agent
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                return func(self, *args, **kwargs)
            
            with tracer.start_as_current_span(
                f"agent_step {agent_name}",
                kind=SpanKind.INTERNAL
            ) as span:
                span.set_attribute("gen_ai.operation.name", "agent_step")
                span.set_attribute("gen_ai.agent.name", agent_name)
                span.set_attribute("gen_ai.conversation.id", get_conversation_id())
                
                try:
                    result = func(self, *args, **kwargs)
                    
                    # Capture step output attributes
                    if hasattr(result, 'action'):
                        span.set_attribute("agent.action", str(result.action)[:500])
                    if hasattr(result, 'thought'):
                        span.set_attribute("agent.thought", str(result.thought)[:500])
                    if hasattr(result, 'done'):
                        span.set_attribute("agent.done", result.done)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_tool_execution(tool_name: str, tool_call_id: str = None, arguments: dict = None):
    """Context manager to trace tool execution.
    
    Args:
        tool_name: Name of the tool being executed
        tool_call_id: Unique ID for this tool call
        arguments: Tool arguments (will be JSON serialized)
    """
    class ToolTraceContext:
        def __init__(self):
            self.span = None
            
        def __enter__(self):
            tracer = get_tracer()
            if tracer is None:
                return self
            
            self.span = tracer.start_span(
                f"execute_tool {tool_name}",
                kind=SpanKind.INTERNAL
            )
            self.span.set_attribute("gen_ai.operation.name", "execute_tool")
            self.span.set_attribute("gen_ai.tool.name", tool_name)
            self.span.set_attribute("gen_ai.tool.type", "function")
            
            if tool_call_id:
                self.span.set_attribute("gen_ai.tool.call.id", tool_call_id)
            if arguments:
                self.span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments)[:2000])
            
            return self
        
        def set_result(self, result: Any):
            if self.span:
                self.span.set_attribute("gen_ai.tool.call.result", json.dumps(result)[:2000] if result else "")
        
        def set_error(self, error: Exception):
            if self.span:
                self.span.set_attribute("error.type", type(error).__name__)
                self.span.set_status(Status(StatusCode.ERROR, str(error)))
                self.span.record_exception(error)
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.span:
                if exc_type is None:
                    self.span.set_status(Status(StatusCode.OK))
                else:
                    self.set_error(exc_val)
                self.span.end()
            return False
    
    return ToolTraceContext()


class AgentRunTrace:
    """Context manager for tracing an entire agent run."""
    
    def __init__(
        self,
        agent_name: str = "SWE-agent",
        problem_id: str = None,
        model_name: str = None,
    ):
        self.agent_name = agent_name
        self.problem_id = problem_id
        self.model_name = model_name
        self.span = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.iterations = 0
    
    def __enter__(self):
        tracer = get_tracer()
        if tracer is None:
            return self
        
        # Reset conversation ID for new run
        reset_conversation_id()
        
        self.span = tracer.start_span(
            f"invoke_agent {self.agent_name}",
            kind=SpanKind.INTERNAL
        )
        self.span.set_attribute("gen_ai.operation.name", "invoke_agent")
        self.span.set_attribute("gen_ai.agent.name", self.agent_name)
        self.span.set_attribute("gen_ai.agent.description", "Software Engineering Agent")
        self.span.set_attribute("gen_ai.conversation.id", get_conversation_id())
        
        if self.problem_id:
            self.span.set_attribute("agent.problem_id", self.problem_id)
        if self.model_name:
            self.span.set_attribute("gen_ai.request.model", self.model_name)
        
        return self
    
    def add_tokens(self, input_tokens: int, output_tokens: int):
        """Add token counts from an LLM call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.iterations += 1
    
    def set_result(self, success: bool, submission: str = None):
        """Set the final result of the agent run."""
        if self.span:
            self.span.set_attribute("agent.success", success)
            self.span.set_attribute("agent.iterations", self.iterations)
            self.span.set_attribute("gen_ai.usage.input_tokens", self.total_input_tokens)
            self.span.set_attribute("gen_ai.usage.output_tokens", self.total_output_tokens)
            if submission:
                self.span.set_attribute("agent.has_submission", True)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type is None:
                self.span.set_status(Status(StatusCode.OK))
            else:
                self.span.set_attribute("error.type", exc_type.__name__)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            self.span.end()
        return False


# Global agent run trace for tracking across the codebase
_current_agent_run: AgentRunTrace = None


def start_agent_run(agent_name: str = "SWE-agent", problem_id: str = None, model_name: str = None) -> AgentRunTrace:
    """Start tracing an agent run."""
    global _current_agent_run
    _current_agent_run = AgentRunTrace(agent_name, problem_id, model_name)
    _current_agent_run.__enter__()
    return _current_agent_run


def end_agent_run(success: bool = False, submission: str = None):
    """End the current agent run trace."""
    global _current_agent_run
    if _current_agent_run:
        _current_agent_run.set_result(success, submission)
        _current_agent_run.__exit__(None, None, None)
        _current_agent_run = None


def get_current_agent_run() -> AgentRunTrace:
    """Get the current agent run trace."""
    return _current_agent_run
