I'll give you a comprehensive explanation of how tracing is implemented in Pipecat. Let me first explore the tracing folder structure and then dive deep into the implementation.
Ran tool
Now let me read the key files to understand the tracing architecture:
Read file: src/pipecat/utils/tracing/__init__.py
Read file: src/pipecat/utils/tracing/setup.py
Read file: src/pipecat/utils/tracing/turn_context_provider.py
Read file: src/pipecat/utils/tracing/turn_trace_observer.py
Read file: src/pipecat/utils/tracing/class_decorators.py
Read file: src/pipecat/utils/tracing/service_attributes.py
Now let me get some examples of how these tracing decorators are used in the codebase:
Search files...
Search files...
Now I have a comprehensive understanding of the tracing implementation in Pipecat. Let me create a detailed explanation with visual diagrams:

# Comprehensive Guide to Tracing in Pipecat

## Overview

Pipecat implements a sophisticated **OpenTelemetry-based distributed tracing system** that provides deep observability into conversational AI pipelines. The tracing architecture captures detailed information about service interactions, performance metrics, and conversation flows across the entire system.

## Architecture Overview
Ran tool
## 1. Core Components Breakdown

### 1.1 Setup Module (`setup.py`)

```7:85:src/pipecat/utils/tracing/setup.py
"""Core OpenTelemetry tracing utilities and setup for Pipecat."""

import os

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


def is_tracing_available() -> bool:
    """Returns True if OpenTelemetry tracing is available and configured.

    Returns:
        bool: True if tracing is available, False otherwise.
    """
    return OPENTELEMETRY_AVAILABLE


def setup_tracing(
    service_name: str = "pipecat",
    exporter=None,  # User-provided exporter
    console_export: bool = False,
) -> bool:
    """Set up OpenTelemetry tracing with a user-provided exporter.

    Args:
        service_name: The name of the service for traces
        exporter: A pre-configured OpenTelemetry span exporter instance.
                  If None, only console export will be available if enabled.
        console_export: Whether to also export traces to console (useful for debugging)

    Returns:
        bool: True if setup was successful, False otherwise

    Example:
        # With OTLP exporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
        setup_tracing("my-service", exporter=exporter)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return False

    try:
        # Create a resource with service info
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.instance.id": os.getenv("HOSTNAME", "unknown"),
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Set up the tracer provider with the resource
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Add console exporter if requested (good for debugging)
        if console_export:
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Add user-provided exporter if available
        if exporter:
            tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        return True
    except Exception as e:
        print(f"Error setting up tracing: {e}")
        return False
```

**Key Features:**
- **Graceful degradation**: If OpenTelemetry isn't available, tracing becomes a no-op
- **Flexible setup**: Supports any OpenTelemetry exporter (Jaeger, OTLP, etc.)
- **Environment awareness**: Automatically captures service metadata from environment

## 2. Turn Tracing - The Heart of Pipecat Observability

### What is Turn Tracing?

**Turn tracing** is Pipecat's core observability concept that tracks **conversational turns** - the fundamental unit of interaction in voice AI applications. A turn represents:

1. **User speaks** (STT processes audio → text)
2. **LLM processes** (generates response based on context)
3. **System responds** (TTS converts text → audio)
Ran tool
Ran tool
### 2.1 Turn Context Provider

```25:72:src/pipecat/utils/tracing/turn_context_provider.py
class TurnContextProvider:
    """Provides access to the current turn's tracing context.

    This is a singleton that services can use to get the current turn's
    span context to create child spans.
    """

    _instance = None
    _current_turn_context: Optional["Context"] = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = TurnContextProvider()
        return cls._instance

    def set_current_turn_context(self, span_context: Optional["SpanContext"]):
        """Set the current turn context.

        Args:
            span_context: The span context for the current turn or None to clear it.
        """
        if not is_tracing_available():
            return

        if span_context:
            # Create a non-recording span from the span context
            non_recording_span = NonRecordingSpan(span_context)
            self._current_turn_context = set_span_in_context(non_recording_span)
        else:
            self._current_turn_context = None

    def get_current_turn_context(self) -> Optional["Context"]:
        """Get the OpenTelemetry context for the current turn.

        Returns:
            The current turn context or None if not available.
        """
        return self._current_turn_context


# Create a simple helper function to get the current turn context
def get_current_turn_context() -> Optional["Context"]:
    """Get the OpenTelemetry context for the current turn.

    Returns:
        The current turn context or None if not available.
    """
    provider = TurnContextProvider.get_instance()
    return provider.get_current_turn_context()
```

**Key Features:**
- **Singleton pattern**: Ensures global access to turn context
- **Thread-safe context management**: Uses OpenTelemetry's context propagation
- **Non-recording spans**: Efficiently propagates context without creating duplicate spans

### 2.2 Turn Trace Observer

```30:150:src/pipecat/utils/tracing/turn_trace_observer.py
class TurnTraceObserver(BaseObserver):
    """Observer that creates trace spans for each conversation turn.

    This observer uses TurnTrackingObserver to track turns and creates
    OpenTelemetry spans for each turn. Service spans (STT, LLM, TTS)
    become children of the turn spans.

    If conversation tracing is enabled, turns become children of a
    conversation span that encapsulates the entire session.
    """

    def __init__(
        self,
        turn_tracker: TurnTrackingObserver,
        conversation_id: Optional[str] = None,
        additional_span_attributes: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._turn_tracker = turn_tracker
        self._current_span: Optional["Span"] = None
        self._current_turn_number: int = 0
        self._trace_context_map: Dict[int, "SpanContext"] = {}
        self._tracer = trace.get_tracer("pipecat.turn") if is_tracing_available() else None

        # Conversation tracking properties
        self._conversation_span: Optional["Span"] = None
        self._conversation_id = conversation_id
        self._additional_span_attributes = additional_span_attributes or {}

        if turn_tracker:

            @turn_tracker.event_handler("on_turn_started")
            async def on_turn_started(tracker, turn_number):
                await self._handle_turn_started(turn_number)

            @turn_tracker.event_handler("on_turn_ended")
            async def on_turn_ended(tracker, turn_number, duration, was_interrupted):
                await self._handle_turn_ended(turn_number, duration, was_interrupted)

    async def on_push_frame(self, data: FramePushed):
        """Process a frame without modifying it.

        This observer doesn't need to process individual frames as it
        relies on turn start/end events from the turn tracker.
        """
        pass

    def start_conversation_tracing(self, conversation_id: Optional[str] = None):
        """Start a new conversation span.

        Args:
            conversation_id: Optional custom ID for the conversation. If None, a UUID will be generated.
        """
        if not is_tracing_available() or not self._tracer:
            return

        # Generate a conversation ID if not provided
        context_provider = ConversationContextProvider.get_instance()
        if conversation_id is None:
            conversation_id = context_provider.generate_conversation_id()
            logger.debug(f"Generated new conversation ID: {conversation_id}")

        self._conversation_id = conversation_id

        # Create a new span for this conversation
        self._conversation_span = self._tracer.start_span("conversation")

        # Set span attributes
        self._conversation_span.set_attribute("conversation.id", conversation_id)
        self._conversation_span.set_attribute("conversation.type", "voice")
        # Set custom otel attributes if provided
        for k, v in (self._additional_span_attributes or {}).items():
            self._conversation_span.set_attribute(k, v)

        # Update the conversation context provider
        context_provider.set_current_conversation_context(
            self._conversation_span.get_span_context(), conversation_id
        )

        logger.debug(f"Started tracing for Conversation {conversation_id}")

    def end_conversation_tracing(self):
        """End the current conversation span and ensure the last turn is closed."""
        if not is_tracing_available():
            return

        # First, ensure any active turn is closed properly
        if self._current_span:
            # If we have an active turn span, end it with a standard duration
            logger.debug(f"Ending Turn {self._current_turn_number} due to conversation end")
            self._current_span.set_attribute("turn.was_interrupted", True)
            self._current_span.set_attribute("turn.ended_by_conversation_end", True)
            self._current_span.end()
            self._current_span = None

            # Clear the turn context provider
            context_provider = TurnContextProvider.get_instance()
            context_provider.set_current_turn_context(None)

        # Now end the conversation span if it exists
        if self._conversation_span:
            # End the span
            self._conversation_span.end()
            self._conversation_span = None

            # Clear the context provider
            context_provider = ConversationContextProvider.get_instance()
            context_provider.set_current_conversation_context(None)

            logger.debug(f"Ended tracing for Conversation {self._conversation_id}")
            self._conversation_id = None

    async def _handle_turn_started(self, turn_number: int):
        """Handle a turn start event by creating a new span."""
        if not is_tracing_available() or not self._tracer:
            return

        # If this is the first turn and no conversation span exists yet,
        # start the conversation tracing (will generate ID if needed)
        if turn_number == 1 and not self._conversation_span:
            self.start_conversation_tracing(self._conversation_id)

        # Get the parent context - conversation if available, otherwise use root context
        parent_context = None
        if self._conversation_span:
            context_provider = ConversationContextProvider.get_instance()
            parent_context = context_provider.get_current_conversation_context()

        # Create a new span for this turn
        self._current_span = self._tracer.start_span("turn", context=parent_context)
        self._current_turn_number = turn_number

        # Set span attributes
        self._current_span.set_attribute("turn.number", turn_number)
        self._current_span.set_attribute("turn.type", "conversation")

        # Add conversation ID attribute if available
        if self._conversation_id:
            self._current_span.set_attribute("conversation.id", self._conversation_id)

        # Store the span context so services can become children of this span
        self._trace_context_map[turn_number] = self._current_span.get_span_context()

        # Update the context provider so services can access this span
        context_provider = TurnContextProvider.get_instance()
        context_provider.set_current_turn_context(self._current_span.get_span_context())

        logger.debug(f"Started tracing for Turn {turn_number}")
```

**Key Features:**
- **Event-driven**: Listens to turn tracking events
- **Hierarchical spans**: Conversation → Turn → Service operations
- **Context propagation**: Makes turn context available to all services
- **Graceful cleanup**: Handles interrupted turns and conversation end

## 3. Service Decorators - Automatic Instrumentation

The service decorators provide **automatic, zero-configuration tracing** for AI services. Each decorator captures service-specific information while maintaining the turn hierarchy.

### 3.1 `@traced_llm` Decorator

```277:481:src/pipecat/utils/tracing/service_decorators.py
def traced_llm(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Traces LLM service methods with LLM-specific attributes.

    Automatically captures and records:
    - Service name and model information
    - Context content and messages
    - Tool configurations
    - Token usage metrics
    - Performance metrics like TTFB
    - Aggregated output text

    Args:
        func: The LLM method to trace.
        name: Custom span name. Defaults to service type and class name.

    Returns:
        Wrapped method with LLM-specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, context, *args, **kwargs):
            try:
                if not is_tracing_available():
                    return await f(self, context, *args, **kwargs)

                service_class_name = self.__class__.__name__
                span_name = "llm"

                # Get the parent context - turn context if available, otherwise service context
                turn_context = get_current_turn_context()
                parent_context = turn_context or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Store original method and output aggregator
                        original_push_frame = self.push_frame
                        output_text = ""  # Simple string accumulation

                        async def traced_push_frame(frame, direction=None):
                            nonlocal output_text
                            # Capture text from LLMTextFrame during streaming
                            if (
                                hasattr(frame, "__class__")
                                and frame.__class__.__name__ == "LLMTextFrame"
                                and hasattr(frame, "text")
                            ):
                                output_text += frame.text

                            # Call original
                            if direction is not None:
                                return await original_push_frame(frame, direction)
                            else:
                                return await original_push_frame(frame)

                        # For token usage monitoring
                        original_start_llm_usage_metrics = None
                        if hasattr(self, "start_llm_usage_metrics"):
                            original_start_llm_usage_metrics = self.start_llm_usage_metrics

                            # Override the method to capture token usage
                            @functools.wraps(original_start_llm_usage_metrics)
                            async def wrapped_start_llm_usage_metrics(tokens):
                                # Call the original method
                                await original_start_llm_usage_metrics(tokens)

                                # Add token usage to the current span
                                _add_token_usage_to_span(current_span, tokens)

                            # Replace the method temporarily
                            self.start_llm_usage_metrics = wrapped_start_llm_usage_metrics

                        try:
                            # Replace push_frame to capture output
                            self.push_frame = traced_push_frame

                            # Detect if we're using Google's service
                            is_google_service = "google" in service_class_name.lower()

                            # Try to get messages based on service type
                            messages = None
                            serialized_messages = None

                            # TODO: Revisit once we unify the messages across services
                            if is_google_service:
                                # Handle Google service specifically
                                if hasattr(context, "get_messages_for_logging"):
                                    messages = context.get_messages_for_logging()
                            else:
                                # Handle other services like OpenAI
                                if hasattr(context, "get_messages"):
                                    messages = context.get_messages()
                                elif hasattr(context, "messages"):
                                    messages = context.messages

                            # Serialize messages if available
                            if messages:
                                try:
                                    serialized_messages = json.dumps(messages)
                                except Exception as e:
                                    serialized_messages = f"Error serializing messages: {str(e)}"

                            # Get tools, system message, etc. based on the service type
                            tools = getattr(context, "tools", None)
                            serialized_tools = None
                            tool_count = 0

                            if tools:
                                try:
                                    serialized_tools = json.dumps(tools)
                                    tool_count = len(tools) if isinstance(tools, list) else 1
                                except Exception as e:
                                    serialized_tools = f"Error serializing tools: {str(e)}"

                            # Handle system message for different services
                            system_message = None
                            if hasattr(context, "system"):
                                system_message = context.system
                            elif hasattr(context, "system_message"):
                                system_message = context.system_message
                            elif hasattr(self, "_system_instruction"):
                                system_message = self._system_instruction

                            # Get settings from the service
                            params = {}
                            if hasattr(self, "_settings"):
                                for key, value in self._settings.items():
                                    if key == "extra":
                                        continue
                                    # Add value directly if it's a basic type
                                    if isinstance(value, (int, float, bool, str)):
                                        params[key] = value
                                    elif value is None or (
                                        hasattr(value, "__name__") and value.__name__ == "NOT_GIVEN"
                                    ):
                                        params[key] = "NOT_GIVEN"

                            # Add all available attributes to the span
                            attribute_kwargs = {
                                "service_name": service_class_name,
                                "model": getattr(self, "model_name", "unknown"),
                                "stream": True,  # Most LLM services use streaming
                                "parameters": params,
                            }

                            # Add optional attributes only if they exist
                            if serialized_messages:
                                attribute_kwargs["messages"] = serialized_messages
                            if serialized_tools:
                                attribute_kwargs["tools"] = serialized_tools
                                attribute_kwargs["tool_count"] = tool_count
                            if system_message:
                                attribute_kwargs["system"] = system_message

                            # Add all gathered attributes to the span
                            add_llm_span_attributes(span=current_span, **attribute_kwargs)

                        except Exception as e:
                            logging.warning(f"Error setting up LLM tracing: {e}")
                            # Don't raise - let the function execute anyway

                        # Run function with modified push_frame to capture the output
                        result = await f(self, context, *args, **kwargs)

                        # Add aggregated output after function completes, if available
                        if output_text:
                            current_span.set_attribute("output", output_text)

                        return result

                    finally:
                        # Always restore the original methods
                        self.push_frame = original_push_frame

                        if (
                            "original_start_llm_usage_metrics" in locals()
                            and original_start_llm_usage_metrics
                        ):
                            self.start_llm_usage_metrics = original_start_llm_usage_metrics

                        # Update TTFB metric
                        ttfb: Optional[float] = getattr(
                            getattr(self, "_metrics", None), "ttfb", None
                        )
                        if ttfb is not None:
                            current_span.set_attribute("metrics.ttfb", ttfb)
            except Exception as e:
                logging.error(f"Error in LLM tracing (continuing without tracing): {e}")
                # If tracing fails, fall back to the original function
                return await f(self, context, *args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
```

**Advanced Features:**
- **Method interception**: Captures streaming output by intercepting `push_frame`
- **Token usage tracking**: Automatically captures and reports token metrics
- **Context serialization**: Safely serializes complex context objects
- **Multi-provider support**: Handles different LLM service APIs (OpenAI, Google, etc.)

### 3.2 Usage Example

Here's how services use these decorators:

```500:650:src/pipecat/services/google/llm.py
class GoogleLLMService(LLMService):
    # ... service initialization code ...

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        grounding_metadata = None
        search_result = ""

        try:
            logger.debug(
                # f"{self}: Generating chat [{self._system_instruction}] | [{context.get_messages_for_logging()}]"
                f"{self}: Generating chat [{context.get_messages_for_logging()}]"
            )

            messages = context.messages
            if context.system_message and self._system_instruction != context.system_message:
                logger.debug(f"System instruction changed: {context.system_message}")
                self._system_instruction = context.system_message

            tools = []
            if context.tools:
                tools = context.tools
            elif self._tools:
                tools = self._tools
            tool_config = None
            if self._tool_config:
                tool_config = self._tool_config

            # Filter out None values and create GenerationContentConfig
            generation_params = {
                k: v
                for k, v in {
                    "system_instruction": self._system_instruction,
                    "temperature": self._settings["temperature"],
                    "top_p": self._settings["top_p"],
                    "top_k": self._settings["top_k"],
                    "max_output_tokens": self._settings["max_tokens"],
                    "tools": tools,
                    "tool_config": tool_config,
                }.items()
                if v is not None
            }

            generation_config = (
                GenerateContentConfig(**generation_params) if generation_params else None
            )

            await self.start_ttfb_metrics()
            response = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=messages,
                config=generation_config,
            )

            function_calls = []
            async for chunk in response:
                # Stop TTFB metrics after the first chunk
                await self.stop_ttfb_metrics()
                if chunk.usage_metadata:
                    prompt_tokens += chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens += chunk.usage_metadata.candidates_token_count or 0
                    total_tokens += chunk.usage_metadata.total_token_count or 0

                if not chunk.candidates:
                    continue

                for candidate in chunk.candidates:
                    # ... process response chunks ...

            await self.run_function_calls(function_calls)
        except DeadlineExceeded:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            if grounding_metadata and isinstance(grounding_metadata, dict):
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=search_result,
                    origins=grounding_metadata["origins"],
                    rendered_content=grounding_metadata["rendered_content"],
                )
                await self.push_frame(llm_search_frame)

            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())
```

**What Gets Captured:**
- Service name: `GoogleLLMService`
- Model: `gemini-pro`
- Input messages: Full conversation history
- System instructions
- Tool configurations
- Token usage: prompt/completion/total tokens
- TTFB (Time to First Byte)
- Complete response text
- Performance metrics

## 4. Service Attributes System

The service attributes module standardizes how different services report their telemetry:
Ran tool

### Key Standardization Features:

```23:85:src/pipecat/utils/tracing/service_attributes.py
def _get_gen_ai_system_from_service_name(service_name: str) -> str:
    """Extract the standardized gen_ai.system value from a service class name.

    Source:
    https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system

    Uses standard OTel names where possible, with special case mappings for
    service names that don't follow the pattern.
    """
    SPECIAL_CASE_MAPPINGS = {
        # AWS
        "AWSBedrockLLMService": "aws.bedrock",
        # Azure
        "AzureLLMService": "az.ai.openai",
        # Google
        "GoogleLLMService": "gcp.gemini",
        "GoogleLLMOpenAIBetaService": "gcp.gemini",
        "GoogleVertexLLMService": "gcp.vertex_ai",
        # Others
        "GrokLLMService": "xai",
    }

    if service_name in SPECIAL_CASE_MAPPINGS:
        return SPECIAL_CASE_MAPPINGS[service_name]

    if service_name.endswith("LLMService"):
        provider = service_name[:-10].lower()
    else:
        provider = service_name.lower()

    return provider
```

**Compliance with OpenTelemetry Standards:**
- Follows [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/) for AI/ML
- Maps service names to standard `gen_ai.system` values
- Ensures consistent attribute naming across providers

## 5. Advanced Features

### 5.1 Specialized Service Decorators

Pipecat includes specialized decorators for advanced AI services:

**Gemini Live Decorator:**
```484:550:src/pipecat/utils/tracing/service_decorators.py
def traced_gemini_live(operation: str) -> Callable:
    """Traces Gemini Live service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:
    - llm_setup: Configuration, tools definitions, and system instructions
    - llm_tool_call: Function call information
    - llm_tool_result: Function execution results
    - llm_response: Complete LLM response with usage and output

    Args:
        operation: The operation name (matches the event type being handled)

    Returns:
        Wrapped method with Gemini Live specific tracing.
    """
```

**OpenAI Realtime Decorator:**
```785:850:src/pipecat/utils/tracing/service_decorators.py
def traced_openai_realtime(operation: str) -> Callable:
    """Traces OpenAI Realtime service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:
    - llm_setup: Session configuration and tools
    - llm_request: Context and input messages
    - llm_response: Usage metadata, output, and function calls

    Args:
        operation: The operation name (matches the event type being handled)

    Returns:
        Wrapped method with OpenAI Realtime specific tracing.
    """
```

### 5.2 Class-level Tracing

For more granular control, Pipecat provides class-level decorators:

```160:220:src/pipecat/utils/tracing/class_decorators.py
def traced(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    attachment_strategy: AttachmentStrategy = AttachmentStrategy.CHILD,
) -> Callable:
    """Adds tracing to an async function in a Traceable class.

    Args:
        func: The async function to trace.
        name: Custom span name. Defaults to function name.
        attachment_strategy: How to attach this span (CHILD, LINK, NONE).

    Returns:
        Wrapped async function with tracing.

    Raises:
        RuntimeError: If used in a class not inheriting from Traceable.
        ValueError: If used on a non-async function.
    """

def traceable(cls: C) -> C:
    """Makes a class traceable for OpenTelemetry.

    Creates a new class that inherits from both the original class
    and Traceable, enabling tracing for class methods.
```

## 6. Real-World Example: Complete Flow

Let me show you what a complete tracing flow looks like in practice:
Ran tool
## 7. Configuration and Setup

### Basic Setup Example:

```python
from pipecat.utils.tracing.setup import setup_tracing
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Setup with Jaeger
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

setup_tracing(
    service_name="my-voice-bot",
    exporter=jaeger_exporter,
    console_export=True  # Also print to console for debugging
)
```

### Integration with Turn Tracking:

```python
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver

# Create observers
turn_tracker = TurnTrackingObserver()
trace_observer = TurnTraceObserver(
    turn_tracker=turn_tracker,
    conversation_id="user-session-123",
    additional_span_attributes={
        "user.id": "user-123",
        "application.version": "1.0.0"
    }
)

# Add to pipeline
pipeline = Pipeline([
    # ... your services ...
], observers=[turn_tracker, trace_observer])
```

## 8. Benefits and Use Cases

### 8.1 Performance Monitoring

**TTFB (Time to First Byte) Tracking:**
- STT: How quickly speech recognition starts producing results
- LLM: Latency from prompt to first token
- TTS: Time to start audio generation

**Token Usage Monitoring:**
- Track costs across different models
- Monitor usage patterns and optimization opportunities
- Real-time budget tracking

### 8.2 Debugging and Troubleshooting

**Conversation Flow Visualization:**
- See the complete flow from user input to system response
- Identify bottlenecks in the pipeline
- Debug interrupted or failed turns

**Error Correlation:**
- Link errors to specific conversation turns
- Track error rates across different services
- Understand failure patterns

### 8.3 Business Intelligence

**User Interaction Analytics:**
- Average turn duration
- Most common conversation patterns
- Service usage distribution

**Quality Metrics:**
- STT accuracy tracking
- LLM response quality patterns
- TTS voice preference analytics

## 9. Performance Considerations

### Minimal Overhead Design:

1. **Conditional Imports**: OpenTelemetry is only imported if available
2. **No-op Fallbacks**: If tracing fails, services continue normally
3. **Efficient Context Propagation**: Uses OpenTelemetry's built-in context system
4. **Lazy Evaluation**: Expensive operations only happen when tracing is enabled

### Resource Management:

- **Automatic Cleanup**: Spans are automatically ended even if services crash
- **Memory Efficient**: Uses span contexts instead of full span objects for propagation
- **Batched Exports**: Uses OpenTelemetry's batch processors for efficient export

The tracing system in Pipecat represents a **sophisticated, production-ready observability solution** specifically designed for conversational AI workloads. It provides the deep insights needed to build, debug, and optimize voice AI applications at scale while maintaining excellent performance characteristics.