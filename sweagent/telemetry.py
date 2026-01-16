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
    from opentelemetry import trace, context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, SpanKind, set_span_in_context
else:
    trace = None
    context = None
    SpanKind = None
    Status = None
    StatusCode = None
    set_span_in_context = None


_tracer = None
_conversation_id = None
_agent_run_context = None  # Store the context with the agent run span
_agent_run_span = None  # Store the agent run span for explicit parent linking


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


def get_agent_run_span():
    """Get the current agent run span for explicit parent linking."""
    return _agent_run_span


# Global step context for nesting tool spans under step spans
_current_step_context = None
_current_step_span = None


def get_current_step_context():
    """Get the current step context for creating child spans."""
    global _current_step_context
    return _current_step_context


class StepTrace:
    """Context manager for tracing a single agent step (LLM call + tool execution)."""
    
    def __init__(self, step_number: int):
        self.step_number = step_number
        self.span = None
        self.ctx = None
        self.token = None
    
    def __enter__(self):
        global _current_step_context, _current_step_span
        tracer = get_tracer()
        if tracer is None:
            return self
        
        # Get parent context (agent run)
        parent_context = _agent_run_context if _agent_run_context else None
        
        self.span = tracer.start_span(
            f"agent_step {self.step_number}",
            kind=SpanKind.INTERNAL,
            context=parent_context
        )
        self.span.set_attribute("gen_ai.operation.name", "agent_step")
        self.span.set_attribute("agent.step.number", self.step_number)
        self.span.set_attribute("gen_ai.conversation.id", get_conversation_id())
        
        # Set this span as current context for child spans
        self.ctx = set_span_in_context(self.span)
        self.token = context.attach(self.ctx)
        _current_step_context = self.ctx
        _current_step_span = self.span
        
        return self
    
    def set_thought(self, thought: str):
        if self.span and thought:
            self.span.set_attribute("agent.step.thought", thought[:1000])
    
    def set_action(self, action: str):
        if self.span and action:
            self.span.set_attribute("agent.step.action", action[:500])
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_step_context, _current_step_span
        if self.span:
            if exc_type is None:
                self.span.set_status(Status(StatusCode.OK))
            else:
                self.span.set_attribute("error.type", type(exc_val).__name__)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            
            if self.token is not None:
                context.detach(self.token)
            _current_step_context = None
            _current_step_span = None
            self.span.end()
        return False


def trace_agent_step(step_number: int):
    """Create a step trace context manager."""
    return StepTrace(step_number)


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
            self.token = None
            
        def __enter__(self):
            tracer = get_tracer()
            if tracer is None:
                return self
            
            # Get parent context - prefer step context, fallback to agent run context
            parent_context = _current_step_context if _current_step_context else (_agent_run_context if _agent_run_context else None)
            
            # Start span as child of step or agent run context
            self.span = tracer.start_span(
                f"execute_tool {tool_name}",
                kind=SpanKind.INTERNAL,
                context=parent_context
            )
            self.span.set_attribute("gen_ai.operation.name", "execute_tool")
            self.span.set_attribute("gen_ai.tool.name", tool_name)
            self.span.set_attribute("gen_ai.tool.type", "function")
            self.span.set_attribute("gen_ai.conversation.id", get_conversation_id())
            
            if tool_call_id:
                self.span.set_attribute("gen_ai.tool.call.id", tool_call_id)
            if arguments:
                self.span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments)[:2000])
            
            # Set this span as current so any nested operations are children
            ctx = set_span_in_context(self.span)
            self.token = context.attach(ctx)
            
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
                # Detach context before ending span
                if self.token is not None:
                    context.detach(self.token)
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
        self.ctx = None
        self.token = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.iterations = 0
    
    def __enter__(self):
        global _agent_run_context, _agent_run_span
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
        
        # Set this span as the current context so child spans are nested
        self.ctx = set_span_in_context(self.span)
        self.token = context.attach(self.ctx)
        _agent_run_context = self.ctx
        _agent_run_span = self.span  # Store the span globally for child span creation
        
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
        global _agent_run_context, _agent_run_span
        if self.span:
            if exc_type is None:
                self.span.set_status(Status(StatusCode.OK))
            else:
                self.span.set_attribute("error.type", exc_type.__name__)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            
            # Detach the context before ending the span
            if self.token is not None:
                context.detach(self.token)
            _agent_run_context = None
            _agent_run_span = None
            self.span.end()
        return False


# Global agent run trace for tracking across the codebase
_current_agent_run: AgentRunTrace = None


def get_agent_run_context():
    """Get the current agent run context for creating child spans."""
    global _agent_run_context
    return _agent_run_context


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
