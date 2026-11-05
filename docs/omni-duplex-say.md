# Forced speech in full-duplex sessions (`session.say`)

`session.say` makes a full-duplex session speak a given text verbatim, as the assistant, and
optionally stay silent afterwards. The text goes through the normal LLM → TTS → token2wav path
and is written into the conversation (KV cache) exactly as if the model had said it.

## Why

The fit-capture demo (MiniCPM_app, `static/fit-capture/`) coaches a user through a short
video recording. Two problems came from leaving everything to the model:

- **Unreliable start.** The page asks the model, through the system prompt, to greet the user and
  give the first instruction. In duplex mode the model decides every second whether to speak or
  listen. It often waited for the user to say something first, so the session sometimes opened
  with silence. Lowering `listen_prob_scale` helps but makes it chatty later.
- **Talking after the end.** Once the recording was saved the page wanted one closing sentence.
  The model kept generating and speaking until the page closed the socket, and closing too early
  cut the sentence off.

Both need the app to decide *what* is said and *when*. That is what `session.say` provides.

## How it works

Each duplex step is `stream_prefill` (one second of audio, optionally a frame), then
`duplex_do_decode`, which samples one unit. A speaking unit, as the model writes it, is:

```
<|speak|> text tokens… [<|turn_eos|>] <|chunk_eos|> </unit>
```

The Talker (TTS) receives each text token together with the LLM hidden state right after that
token was evaluated. While a say is pending, `duplex_do_decode` calls `duplex_do_forced_speak`
instead of sampling. It:

1. takes the next ~`chars_per_step` characters of the queued tokens, never splitting a word
   (default 12, about one second of speech; the duplex TTS generates at most 26 audio tokens,
   about 1 s, per non-final unit, so longer units drop words);
2. evaluates `<|speak|> + text (+ <|turn_eos|> on the last unit)` with `eval_tokens_with_hidden`,
   which gives the same (token, hidden state) pairs as sampling would;
3. evaluates `<|chunk_eos|>` and `</unit>`, then pushes the text to the text stream and an
   `LLMOut` to the TTS queue (`is_end_of_turn` on the last unit, so the TTS flushes);
4. does the same end-of-step bookkeeping as the sampled path (`duplex_finish_decode`:
   `__END_OF_TURN__`, sliding-window response unit, round boundary).

A pending say also pre-empts the opening `force_listen_count` steps. The user's audio and video
are still prefilled every step, so the model hears what happens while it is "talking". When the
text runs out, sampling resumes. With `mute_after`, the session instead emits
`<|listen|></unit>` every step until the next say or `session.reset`.

Text is tokenized with `parse_special=false`, so client text cannot inject control tokens. A new
say replaces any unfinished one. `session.reset` and session init clear the say and mute state.

Only the async duplex path (`duplex_llm_thread_func`, used by `llama-omni-server`) supports this.
The legacy synchronous `stream_decode` duplex branch ignores the queue.

## Protocol

Client → server (full_duplex only; other modes fail with `say_unsupported_mode`):

```json
{"type": "session.say",
 "payload": {"text": "Hi! Please step back until I can see all of you.",
             "chars_per_step": 12,
             "mute_after": false}}
```

| field | type | default | meaning |
|---|---|---|---|
| `text` | string | required | spoken verbatim; empty text just applies `mute_after` |
| `chars_per_step` | int | 12 | characters per one-second unit |
| `mute_after` | bool | false | after the last word, listen every step until the next say or reset |

Nothing is sent back right away. The forced units arrive as ordinary `response.output.delta` (text and audio)
and `response.done` events. After the step that decoded the last unit, the server sends:

```json
{"type": "session.say.done", "session_id": "…", "muted": true, "server_send_ts": …}
```

At that point the last audio may still be streaming to the client, so wait for playback to
finish before closing.

## Code

- `tools/omni/omni.h`: `duplex_say_*`, `duplex_muted`; `omni_duplex_say()`, `omni_duplex_say_clear()`
- `tools/omni/omni.cpp`: `duplex_say_take_unit`, `duplex_do_forced_speak`, the hook at the top
  of `duplex_do_decode`, `duplex_finish_decode`
- `tools/server/ws_handler.cpp`: the `session.say` message, `session.say.done` after each step
- `tools/server/protocol.cpp`: `make_session_say_done`
