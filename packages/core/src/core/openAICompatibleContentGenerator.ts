/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  FunctionCall,
  GenerateContentParameters,
  GenerateContentResponse
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

// Helper class to wrap the OpenAI API response
class OpenAIGenerateContentResponse implements GenerateContentResponse {
  data: any;

  constructor(response: any) {
    this.data = response;
  }

  get candidates() {
    return this.data.choices.map((choice: any) => ({
      index: choice.index,
      content: {
        role: 'model',
        parts: [{ text: choice.message.content }],
      },
      finishReason: choice.finish_reason,
      citationMetadata: {
        citationSources: [],
      },
    }));
  }

  get promptFeedback() {
    return {
      blockReason: undefined,
      safetyRatings: [],
    };
  }

  get text(): string {
    const choice = this.data.choices[0];
    return choice.message.content;
  }

  get functionCalls(): FunctionCall[] {
    return [];
  }

  get executableCode(): string {
    return '';
  }

  get codeExecutionResult(): string {
    return '';
  }

  [Symbol.iterator]() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    let i = 0;
    return {
      next() {
        if (i++ === 0) {
          return { value: self, done: false };
        }
        return { value: self, done: true };
      },
    };
  }
}

export class OpenAICompatibleContentGenerator implements ContentGenerator {
  private endpoint: string;
  private model: string;
  private apiKey: string;
  userTier?: any;

  constructor(config: { endpoint: string; model: string; apiKey: string }) {
    this.endpoint = config.endpoint;
    this.model = config.model;
    this.apiKey = config.apiKey;
  }

  private contentsToParts(contents: GenerateContentParameters['contents']): Content[] {
    if (typeof contents === 'string') {
      return [{ role: 'user', parts: [{ text: contents }] }];
    }
    return contents as Content[];
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.convertToOpenAIMessages(
      this.contentsToParts(request.contents),
    );
    const generationConfig = (request as any).generationConfig;
    const response = await fetch(`${this.endpoint}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages,
        temperature: generationConfig?.temperature,
        max_tokens: generationConfig?.maxOutputTokens,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch from local LLM: ${response.status} ${
          response.statusText
        } ${await response.text()}`,
      );
    }
    const data = await response.json();
    return new OpenAIGenerateContentResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.convertToOpenAIMessages(
      this.contentsToParts(request.contents),
    );
    const generationConfig = (request as any).generationConfig;
    const response = await fetch(`${this.endpoint}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages,
        temperature: generationConfig?.temperature,
        max_tokens: generationConfig?.maxOutputTokens,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch from local LLM: ${response.status} ${
          response.statusText
        } ${await response.text()}`,
      );
    }

    const stream = response.body;
    if (!stream) {
      throw new Error('No response body');
    }

    const generator = async function* () {
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const json = line.substring('data: '.length);
            if (json.trim() === '[DONE]') {
              return;
            }
            const data = JSON.parse(json);
            yield new OpenAIGenerateContentResponse({
              ...data,
              // We need to remap the delta to a message
              choices: data.choices.map((choice: any) => ({
                ...choice,
                message: choice.delta,
              })),
            });
          }
        }
      }
    };

    return generator();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const contents = this.contentsToParts(request.contents);
    const text = contents
      .map((c) =>
        (c.parts ?? []).map((p) => (p as { text: string }).text).join(''),
      )
      .join('');

    return {
      totalTokens: Math.ceil(text.length / 4),
    };
  }

  embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('embedContent not supported');
  }

  private convertToOpenAIMessages(
    contents: Content[],
  ): { role: 'user' | 'assistant'; content: string }[] {
    const messages: { role: 'user' | 'assistant'; content: string }[] = [];
    for (const content of contents) {
      const role = content.role === 'model' ? 'assistant' : 'user';
      const text = (content.parts ?? [])
        .map((part) => (part as { text: string }).text)
        .join('');
      messages.push({ role, content: text });
    }
    return messages;
  }
}