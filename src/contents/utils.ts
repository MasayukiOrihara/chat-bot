import { ChatAnthropic } from '@langchain/anthropic';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { ChatOpenAI } from '@langchain/openai';
import { readFile } from 'fs/promises';
import fs from 'fs/promises';
import path from 'path';

import { MessageJson } from './type';
import { LangChainAdapter } from 'ai';

/**
 * JSONファイル読み込み用関数
 * @param relativePath JSONファイルパス
 * @returns オブジェクト
 */
export async function loadJsonFile<T = unknown>(relativePath: string): Promise<
  | { success: true; data: T }
  | { success: false; error: string }
> {
  const filePath = path.join(process.cwd(), relativePath);

    try {
        const data = await readFile(filePath, 'utf-8');
        const json = JSON.parse(data);
        return { success: true, data: json };
    } catch (error) {
        return {
            success: false,
            error:
            error instanceof Error
            ? error.message
            : 'Unknown error while reading JSON file',
        };
    }
}

/**
 * 全文取得 & クローン
 * stream時にストリーム配信する＆全文をJSONに保存する関数
 * @param input 
 * @param stream 
 * @param path 
 * @returns 
 */
export async function wrapStreamWithSave(
  input: string,
  stream: ReadableStream,
  path: string
): Promise<ReadableStream> {
  const chunks: string[] = [];
  const response = LangChainAdapter.toDataStreamResponse(stream); 
  const reader = response.body?.getReader();

  const wrappedStream = new ReadableStream({
      async start(controller) {
        if (!reader) {
          controller.error(new Error(`ReadableStream reader not available`));
          return;
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          if (value instanceof Uint8Array) {
            const text = new TextDecoder().decode(value, { stream: true });

            const cleaned = text
              .split('\n')
              .map(line => {
                const match = line.match(/^\d+:"(.*)"$/);
                return match ? match[1] : '';
              })
              .join('');

            chunks.push(cleaned); // 保存用
            controller.enqueue(value); // クライアント用
          }
        }
        controller.close();

        // ここで全文保存
        await saveAsJson(input, chunks.join(''), path);
      },
    });

  return wrappedStream;
}

/**
 * 保存ユーティリティ
 * @param output 
 */
export async function saveAsJson(inputText: string,outputText: string, relativePath: string) {
  const filePath = path.join(process.cwd(), relativePath);
  let existingData: MessageJson[] = [];

  try {
    const fileData = await readFile(filePath, 'utf-8');
    existingData = JSON.parse(fileData);
  } catch (error) {
    console.log(error + ' :Create a new file.')
  }

  existingData.push({
    input: inputText,
    output: outputText,
    createdAt: getJstIsoString(),
  });

  await fs.writeFile(filePath, JSON.stringify(existingData, null, 2), 'utf-8');
}

/**
 * JST表記に変える関数
 * @returns 
 */
export function getJstIsoString() {
  const now = new Date();
  now.setHours(now.getHours() + 9);

  return now.toISOString().replace('Z', '+09:00');
}

/**
 * 型安全オブジェクト判定
 * @param val 
 * @returns 
 */
export function isObject(val: unknown): val is Record<string, unknown> {
    return typeof val === 'object' && val !== null;
}

/**
 * 名前から言語モデルを追加する
 * @param modelName 
 * @returns 
 */
export function getModel(modelName: string){
  let model;

  switch (modelName) {
    case 'gpt-4o':
      model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o',
      temperature: 0.7, // ランダム度（高いほど創造的）
      });
    break;
    case 'gpt-4o-mini':
      model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o-mini',
      temperature: 0.7, // ランダム度（高いほど創造的）
      });
    break;
    case 'claude-haiku':
      model = new ChatAnthropic({
        model: 'claude-3-5-haiku-20241022',
        temperature: 0.7, // ランダム度（高いほど創造的）
      });
    break;
    default:
      model = new FakeListChatModel({
        responses: [
          "（応答結果）",
        ],
      });
  }
  
  return model;
}

/**
 * キーワード以降の文字を抜き出す関数
 * @param input 
 * @param keyword 
 * @returns 
 */
export function cutKeyword(input: string, keyword: string): string {
  const index = input.indexOf(keyword);
  if (index === -1) return input;

  const endIndex = input.indexOf('\n', index);
  if (endIndex === -1) return input.slice(index + keyword.length);
  
  return input.slice(index + keyword.length, endIndex);
}