Search files...
Here’s an in-depth explanation of how the current tool/function call flow and registration work in Pipecat, based on the codebase and the search results:

---

## 1. **Tool/Function Registration**

- **Where:** Tool registration is handled in the LLM service layer (see `src/pipecat/services/llm_service.py` and `src/pipecat/services/mcp_service.py`).
- **How:**
  - Tools (functions) are registered with the LLM service using a method like `register_function(function_name, handler)`.
  - For external tool providers (like MCP), the service queries the available tools (via `_list_tools`), converts their schemas, and registers a wrapper handler for each tool.
  - The handler is responsible for executing the tool and returning the result via a callback.

**Example:**
```python
llm.register_function(tool_name, mcp_tool_wrapper)
```
- The `mcp_tool_wrapper` is a coroutine that knows how to call the tool and handle the result.

---

## 2. **Tool Call Initiation (Function Call from LLM)**

- **How:**
  - When the LLM generates a response that includes a function/tool call (e.g., OpenAI function calling), the LLM service parses the function call from the LLM output.
  - It creates a `FunctionCallFromLLM` object, which includes the function name, tool call ID, arguments, and context.
  - These are collected and passed to `run_function_calls`.

**Example:**
```python
function_calls = [FunctionCallFromLLM(...)]
await self.run_function_calls(function_calls)
```

---

## 3. **Function Call Execution**

- **How:**
  - The LLM service maintains a registry of function handlers.
  - For each function call, it creates a `FunctionCallRunnerItem` and invokes the registered handler.
  - Before execution, it emits a `FunctionCallInProgressFrame` downstream and upstream to signal that a function call is in progress.
  - The handler executes the function/tool and, upon completion, calls a callback with the result.

**Example:**
```python
progress_frame = FunctionCallInProgressFrame(...)
await self.push_frame(progress_frame, FrameDirection.DOWNSTREAM)
```

---

## 4. **Result Handling**

- **How:**
  - When the tool/function completes, the result is wrapped in a `FunctionCallResultFrame`.
  - This frame is pushed both downstream and upstream.
  - The context aggregator (e.g., `LLMAssistantContextAggregator`) receives the result frame, updates the conversation context, and may trigger further LLM inference if needed (e.g., to generate a final response using the tool result).

**Example:**
```python
result_frame = FunctionCallResultFrame(...)
await self.push_frame(result_frame, FrameDirection.DOWNSTREAM)
```

---

## 5. **Context Aggregator Role**

- **Tracks** which function calls are in progress.
- **Updates** the context with the result when a `FunctionCallResultFrame` is received.
- **Triggers** further LLM inference if the tool result requires it (e.g., if `run_llm` is set).

---

## 6. **Summary of Frame Types Involved**

- `FunctionCallFromLLM`: Represents a function call requested by the LLM.
- `FunctionCallInProgressFrame`: Signals that a function call is being executed.
- `FunctionCallResultFrame`: Contains the result of the function call.
- `FunctionCallCancelFrame`: Signals cancellation of a function call.

---

## 7. **How Tool Calls Are Correlated**

- Each tool call has a unique `tool_call_id`.
- This ID is used to match the in-progress frame, the result frame, and any context updates.

---

## 8. **Test Coverage**

- There are tests (see `tests/test_context_aggregators.py`) that simulate the function call flow, including in-progress, result, and context update handling.

---

## 9. **Extensibility**

- The system is designed to support multiple tool providers and dynamic tool registration.
- The function call flow is abstracted so that new tool types or providers can be integrated by registering new handlers.

---

### **In summary:**
- Tools are registered with the LLM service.
- When the LLM requests a tool call, a function call frame is created and executed via the registered handler.
- Progress and result frames are pushed through the pipeline.
- The context aggregator updates the conversation state and may trigger further LLM actions.

---

**If you want to implement a client-side tool call, you would follow a similar pattern:**
- Register client tools.
- When a tool call is needed, emit a frame (e.g., `ClientToolCallFrame`).
- Wait for the client to respond with a result frame.
- Update context and continue the pipeline as with server-side tools.

---

**If you have any specific questions about any part of this flow, or want to see how a particular frame or handler works, let me know!**