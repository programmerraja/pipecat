Here are the incoming (server → client) and outgoing (client → server) WebSocket message types from ElevenLabs' Conversational AI, along with sample data:

---

## 🔄 Client → Server Events

1. **conversation\_initiation\_client\_data**

```json
{
  "type": "conversation_initiation_client_data",
  "conversation_config_override": {
    "agent": {
      "prompt": { "prompt": "You are a helpful customer support agent named Alexis." },
      "first_message": "Hi, I'm Alexis from ElevenLabs support. How can I help you today?",
      "language": "en"
    },
    "tts": { "voice_id": "21m00Tcm4TlvDq8ikWAM" }
  },
  "custom_llm_extra_body": {"temperature":0.7,"max_tokens":150},
  "dynamic_variables": {"user_name":"John","account_type":"premium"}
}
```

2. **user\_audio\_chunk** (streaming audio from mic)

```json
{ "user_audio_chunk": "base64EncodedAudioData==" }
```

3. **pong** (response to ping)

```json
{ "type": "pong", "event_id": 12345 }
```

4. **client\_tool\_result**

```json
{
  "type": "client_tool_result",
  "tool_call_id": "tool_call_123",
  "result": "Account is active and in good standing",
  "is_error": false
}
```

5. **contextual\_update**

```json
{ "type": "contextual_update", "text": "User is viewing the pricing page" }
```

6. **user\_message** (text input)

```json
{ "type": "user_message", "text": "I would like to upgrade my account" }
```

7. **user\_activity** (heartbeat or activity signal)

```json
{ "type": "user_activity" }
```

---

## 🛎️ Server → Client Events

1. **conversation\_initiation\_metadata**

```json
{
  "type": "conversation_initiation_metadata",
  "conversation_initiation_metadata_event": {
    "conversation_id": "conv_123456789",
    "agent_output_audio_format": "pcm_16000",
    "user_input_audio_format": "pcm_16000"
  }
}
```

2. **vad\_score**

```json
{ "type": "vad_score", "vad_score_event": { "vad_score": 0.95 } }
```

3. **user\_transcript**

```json
{
  "type": "user_transcript",
  "user_transcription_event": {
    "user_transcript": "I need help with my voice cloning project."
  }
}
```

4. **internal\_tentative\_agent\_response**

```json
{
  "type": "internal_tentative_agent_response",
  "tentative_agent_response_internal_event": {
    "tentative_agent_response": "I'd be happy to help with your voice cloning project..."
  }
}
```

5. **agent\_response**

```json
{
  "type": "agent_response",
  "agent_response_event": {
    "agent_response": "I'd be happy to help with your voice cloning project. Could you tell me what specific aspects you need assistance with?"
  }
}
```

6. **audio** (agent’s audio stream)

```json
{
  "type": "audio",
  "audio_event": {
    "audio_base_64": "base64EncodedAudioResponse==",
    "event_id": 1
  }
}
```

7. **ping** (heartbeat from server)

```json
{ "type": "ping", "ping_event": { "event_id": 12345, "ping_ms": 50 } }
```

8. **interruption**

```json
{
  "type": "interruption",
  "interruption_event": { "reason": "user_interrupt" }
}
```

9. **client\_tool\_call** (server asking client to call a tool)

```json
{
  "type": "client_tool_call",
  "client_tool_call": {
    "tool_name": "check_account_status",
    "tool_call_id": "tool_call_123",
    "parameters": {"user_id":"user_123"}
  }
}
```

---

### ✅ Summary Table

| Direction         | Type                                   | Description                          |
| ----------------- | -------------------------------------- | ------------------------------------ |
| **Client→Server** | conversation\_initiation\_client\_data | Start convo, set TTS params          |
|                   | user\_audio\_chunk                     | Stream of user’s audio               |
|                   | pong                                   | Reply to server ping                 |
|                   | client\_tool\_result                   | Return tool call result              |
|                   | contextual\_update                     | Provide contextual info              |
|                   | user\_message                          | Send text instead of voice           |
|                   | user\_activity                         | Heartbeat/activity signal            |
| **Server→Client** | conversation\_initiation\_metadata     | Server confirms conversation start   |
|                   | vad\_score                             | Voice activity detection score       |
|                   | user\_transcript                       | Transcript of user speech            |
|                   | internal\_tentative\_agent\_response   | Intermediate partial text from agent |
|                   | agent\_response                        | Final agent response in text         |
|                   | audio                                  | Agent’s audio stream                 |
|                   | ping                                   | Server heartbeat                     |
|                   | interruption                           | Notification of user interruption    |
|                   | client\_tool\_call                     | Server instructs client to run tool  |

---

Let me know if you'd like sample code or guidance on how to handle or test any of these messages in your own app.

---

**Twisted quiz for you**:
Which event would you send *only* when you want to inject non-urgent context (without interrupting the conversation)?
