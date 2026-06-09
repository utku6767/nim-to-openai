// server.js - Advanced OpenAI to NVIDIA NIM API Proxy (Kimi K2.6 Edition)
// Supports: Vision (Multimodal), Deep Reasoning, and Native Tool Use (Function Calling)

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - Max limit increased to 50mb to support base64 vision/image uploads smoothly
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 GLOBAL TOGGLES FOR ADVANCED FEATURES
const SHOW_REASONING = true;        // Set true to wrap Kimi's internal thinking steps inside <think> tags
const FORCE_THINKING_MODE = true;   // Injects chat_template_kwargs parameter for heavy reasoning tasks

const DEFAULT_MODEL = 'moonshotai/kimi-k2.6';
const MODEL_MAPPING = {
  'gpt-3.5-turbo':   DEFAULT_MODEL,
  'gpt-4':           DEFAULT_MODEL,
  'gpt-4-turbo':     DEFAULT_MODEL,
  'gpt-4o':          DEFAULT_MODEL,
  'claude-3-opus':   DEFAULT_MODEL,
  'claude-3-sonnet': DEFAULT_MODEL,
  'gemini-pro':      DEFAULT_MODEL,
  'kimi-k2.6':       DEFAULT_MODEL
};

// Health Check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'Kimi K2.6 Advanced Proxy',
    capabilities: ['vision', 'reasoning', 'tool_use'],
    reasoning_display: SHOW_REASONING,
    thinking_mode: FORCE_THINKING_MODE
  });
});

// OpenAI compatible model listing
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Main Chat Completions Endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;
    const nimModel = MODEL_MAPPING[model] || DEFAULT_MODEL;

    // 1. VISION SUPPORT: Process OpenAI multimodal message format natively
    const formattedMessages = messages.map(msg => {
      if (Array.isArray(msg.content)) {
        return {
          role: msg.role,
          content: msg.content.map(part => {
            if (part.type === 'image_url') {
              // Standardizes OpenAI image format to standard multimodal spec
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

    // 2. REASONING & TOOL USE SUPPORT: Construct payload for NVIDIA NIM
    const nimRequest = {
      model: nimModel,
      messages: formattedMessages,
      temperature: temperature !== undefined ? temperature : 0.7,
      top_p: 1.0,
      max_tokens: max_tokens || 16384,
      stream: stream || false,
      // Pass client tools (Cursor/Chatbox agents) straight to Kimi's native tool tracker
      tools: tools || undefined,
      tool_choice: tool_choice || undefined,
      // Inject native thinking parameter if toggled or needed by agent
      extra_body: FORCE_THINKING_MODE 
        ? { chat_template_kwargs: { thinking: true } } 
        : undefined
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

    // 3. STREAMING OUTPUT HANDLER
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

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
                const reasoning = delta.reasoning_content;
                const content = delta.content;

                // Dynamically intercept reasoning content chunks and format them into <think> tags
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  if (combinedContent) {
                    delta.content = combinedContent;
                  }
                }
                
                // Keep tool calls intact during stream chunks for agents like Cursor / Aider
                if (delta.tool_calls) {
                  delta.tool_calls = delta.tool_calls;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => { res.end(); });

    } else {
      // 4. NON-STREAMING (JSON) OUTPUT HANDLER
      const choice = response.data.choices[0];
      let fullContent = choice.message?.content || '';

      // Format non-streaming reasoning blocks
      if (SHOW_REASONING && choice.message?.reasoning_content) {
        fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${fullContent}`;
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model || 'gpt-4o',
        choices: [{
          index: 0,
          message: {
            role: choice.message.role,
            content: fullContent,
            // Return tools safely to client if Kimi executed a tool invocation step
            tool_calls: choice.message.tool_calls || undefined
          },
          finish_reason: choice.finish_reason
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal proxy server error',
        type: 'proxy_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

