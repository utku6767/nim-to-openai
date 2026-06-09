// server.js - OpenAI → NVIDIA NIM Proxy (Kimi K2.6 HARDENED)
// Fixes: tool DSL leaks, streaming corruption, second-turn tool bleed

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ---------------- CONFIG ----------------
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const MODEL = 'moonshotai/kimi-k2.6';

if (!NIM_API_KEY) {
  console.warn('⚠️ Missing NIM_API_KEY');
}

// ---------------- MIDDLEWARE ----------------
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// ---------------- TOOL FIREWALL ----------------
function stripToolTokens(text = '') {
  if (!text) return text;

  return text
    // full tool block
    .replace(/<\|tool_calls_section_begin\|>[\s\S]*?<\|tool_calls_section_end\|>/g, '')

    // single tool calls
    .replace(/<\|tool_call_begin\|>[\s\S]*?<\|tool_call_end\|>/g, '')

    // leftovers
    .replace(/<\|tool_calls_section_begin\|>/g, '')
    .replace(/<\|tool_calls_section_end\|>/g, '')
    .replace(/<\|tool_call_begin\|>/g, '')
    .replace(/<\|tool_call_end\|>/g, '')

    // whitespace cleanup
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ---------------- MESSAGE SANITIZER ----------------
function sanitizeMessage(msg) {
  if (!msg) return msg;

  // strip tool DSL leaks from content
  if (typeof msg.content === 'string') {
    msg.content = stripToolTokens(msg.content);
  }

  // tool call safety: never mix content + tool_calls
  if (msg.tool_calls?.length) {
    msg.content = null;
  }

  // optional: strip reasoning leaks
  delete msg.reasoning_content;

  return msg;
}

// ---------------- REQUEST MESSAGE CLEANER ----------------
function sanitizeMessages(messages = []) {
  return messages.map(m => {
    const cleaned = { ...m };

    if (cleaned.role === 'assistant' && cleaned.tool_calls) {
      cleaned.content = null;
    }

    // DO NOT forward raw tool messages into model context
    if (cleaned.role === 'tool') {
      return {
        role: 'tool',
        content: stripToolTokens(cleaned.content || ''),
        tool_call_id: cleaned.tool_call_id
      };
    }

    if (typeof cleaned.content === 'string') {
      cleaned.content = stripToolTokens(cleaned.content);
    }

    return cleaned;
  });
}

// ---------------- HEALTH ----------------
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: MODEL
  });
});

// ---------------- MODELS ----------------
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [
      {
        id: 'kimi-k2.6',
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'moonshotai'
      }
    ]
  });
});

// ---------------- CHAT COMPLETIONS ----------------
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const {
      messages,
      temperature = 0.6,
      max_tokens = 16384,
      stream = false,
      tools,
      tool_choice
    } = req.body;

    if (!messages) {
      return res.status(400).json({ error: { message: 'messages required' } });
    }

    const nimRequest = {
      model: MODEL,
      messages: sanitizeMessages(messages),
      temperature,
      top_p: 1.0,
      max_tokens,
      stream,
      ...(tools && { tools }),
      ...(tool_choice && { tool_choice })
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          Accept: stream ? 'text/event-stream' : 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120000
      }
    );

    // ---------------- STREAM MODE ----------------
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';

      response.data.on('data', chunk => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;

          const payload = line.replace('data: ', '').trim();

          if (payload === '[DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const json = JSON.parse(payload);

            const delta = json?.choices?.[0]?.delta;

            if (delta) {
              sanitizeMessage(delta);

              if (typeof delta.content === 'string') {
                delta.content = stripToolTokens(delta.content);
              }

              if (delta.tool_calls?.length) {
                delta.content = null;
              }
            }

            res.write(`data: ${JSON.stringify(json)}\n\n`);
          } catch {
            // last resort cleanup
            res.write(`${stripToolTokens(line)}\n\n`);
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => {
        console.error('Stream error:', err);
        res.end();
      });

      return;
    }

    // ---------------- NORMAL MODE ----------------
    const choices = (response.data.choices || []).map(c => {
      const msg = sanitizeMessage({
        role: c.message.role,
        content: c.message.content,
        tool_calls: c.message.tool_calls
      });

      return {
        index: c.index,
        message: msg,
        finish_reason: c.finish_reason
      };
    });

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: MODEL,
      choices,
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });

  } catch (err) {
    console.error('Proxy error:', err.message);

    res.status(err.response?.status || 500).json({
      error: {
        message: err.message,
        type: 'proxy_error',
        code: err.response?.status || 500
      }
    });
  }
});

// ---------------- 404 ----------------
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Not found: ${req.path}`, code: 404 }
  });
});

// ---------------- START ----------------
app.listen(PORT, () => {
  console.log(`🚀 Proxy running on ${PORT}`);
  console.log(`🧠 Model: ${MODEL}`);
});
