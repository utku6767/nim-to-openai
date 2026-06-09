// server.js - OpenAI to NVIDIA NIM API Proxy (Kimi K2.6 Edition)
// Features: Vision (multimodal), Tool Use (function calling)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware — 50mb limit for base64 image uploads
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Model mapping — all routes lead to Kimi K2.6
const MODEL_MAPPING = {
  'gpt-3.5-turbo':   'moonshotai/kimi-k2.6',
  'gpt-4':           'moonshotai/kimi-k2.6',
  'gpt-4-turbo':     'moonshotai/kimi-k2.6',
  'gpt-4o':          'moonshotai/kimi-k2.6',
  'claude-3-opus':   'moonshotai/kimi-k2.6',
  'claude-3-sonnet': 'moonshotai/kimi-k2.6',
  'gemini-pro':      'moonshotai/kimi-k2.6',
  'kimi-k2.6':       'moonshotai/kimi-k2.6'
};

const DEFAULT_MODEL = 'moonshotai/kimi-k2.6';

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy — Kimi K2.6',
    default_model: DEFAULT_MODEL,
    capabilities: ['vision', 'tool_use']
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;

    const nimModel = MODEL_MAPPING[model] || DEFAULT_MODEL;

    // VISION: Normalize OpenAI multimodal message format for NIM
    const formattedMessages = messages.map(msg => {
      if (Array.isArray(msg.content)) {
        return {
          role: msg.role,
          content: msg.content.map(part => {
            if (part.type === 'image_url') {
              return {
                type: 'image_url',
                image_url: { url: part.image_url.url }
              };
            }
            return part;
          })
        };
      }
      return msg;
    });

    // Build NIM request
    const nimRequest = {
      model: nimModel,
      messages: formattedMessages,
      temperature: temperature !== undefined ? temperature : 1.0,
      top_p: 1.0,
      max_tokens: max_tokens || 16384,
      stream: stream || false,
      // TOOL USE: pass through function definitions if provided
      ...(tools && { tools }),
      ...(tool_choice && { tool_choice })
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          'Accept': stream ? 'text/event-stream' : 'application/json'
        },
        responseType: stream ? 'stream' : 'json'
      }
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                // Strip reasoning content — clean output only
                delete delta.reasoning_content;
                // Tool call chunks pass through untouched
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      const choices = response.data.choices.map(choice => {
        const message = {
          role: choice.message.role,
          content: choice.message.content || ''
          // reasoning_content intentionally omitted
        };

        // TOOL USE: include tool_calls if Kimi returned them
        if (choice.message.tool_calls) {
          message.tool_calls = choice.message.tool_calls;
        }

        return {
          index: choice.index,
          message,
          finish_reason: choice.finish_reason
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model || 'kimi-k2.6',
        choices,
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    if (error.response?.data) {
      console.error('NIM error body:', JSON.stringify(error.response.data));
    }
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI → NVIDIA NIM Proxy (Kimi K2.6) running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
});
