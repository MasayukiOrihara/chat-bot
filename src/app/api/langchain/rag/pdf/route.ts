import { PromptTemplateJson } from '@/contents/type';
import { getModel, isObject, loadJsonFile } from '@/contents/utils';
import { PromptTemplate } from '@langchain/core/prompts';
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
import fs from 'fs/promises';

import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from "@langchain/pinecone";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// チャット形式
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

/**
 * チャット応答AI（記憶・モデル変更対応済み）
 * @param req 
 * @returns 
 */
export async function POST(req: Request) {
  try{
    const body = await req.json();
    const messages = body.messages ?? [];
    const modelName = body.model ?? 'fake-llm';

    // 過去の履歴{chat_history}
    const formattedPreviousMessages = messages
      .slice(0, -1)
      .map(formatMessage)

    // メッセージ{input}
    const currentMessageContent = messages[messages.length - 1].content;

    // text取得
    const readTextFile = async (filePath: string): Promise<string> => {
      try {
        // ファイルを文字列として読み込む
        const text = await fs.readFile(filePath, 'utf-8');
        return text;
      } catch (error) {
        console.error('ファイル読み込みエラー:', error);
        throw error;
      }
    };
    
    // ドキュメントを分割 & ベクトル化
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 100,
      chunkOverlap: 50,
    });
  
    // 埋め込みモデル
    const embeddings = new OpenAIEmbeddings({
      modelName: "text-embedding-3-large",
      apiKey: process.env.OPENAI_API_KEY
    });

    // テキストを分割
    const text = await readTextFile("C:\\localgit\\chat-bot\\src\\data\\text\\稼働日、稼働時間に関するルール.txt");
    const splitText = await splitter.splitText(text);

    console.log(splitText.length);

    /** Pineconeを試してみる */
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!
    });

    // インデックスを取得
    const index = pinecone.index(process.env.PINECONE_INDEX_TEXT!);

    // PineconeStore を作成し、ドキュメントを追加
    const metadatas: object[] = splitText.map(() => ({}));
    const vectorStore = await PineconeStore.fromTexts(splitText, metadatas, embeddings, {
      pineconeIndex: index,
      namespace: "default", // 必要に応じて名前空間を設定
    });

            
    
    // vectore storeからinputで検索
    const response = await vectorStore.similaritySearchWithScore(currentMessageContent, 2);
      
    
    // 抽出
    const context = response.map(([doc, score]) => ({
      content: doc.pageContent,
      score: score,
    }));
    const data = context.map(item => item.content);
    console.log(response);

    //プロンプトテンプレートの作成
    const template = await loadJsonFile<PromptTemplateJson[]>('src/data/prompt-template.json');
    if (!template.success) {
      return new Response(JSON.stringify({ error: template.error }),{
        status: 500,
        headers: { 'Content-type' : 'application/json' },
      });
    }
    // プロンプトテンプレートの抽出
    const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-chat-aikato');
    if (!found) {
      throw new Error('テンプレートが見つかりませんでした');
    }

    // モデルの指定
    const model = getModel(modelName);
        
    const prompt = PromptTemplate.fromTemplate(found.template);
    const chain = prompt.pipe(model);

    const stream = await chain.stream({
      history: formattedPreviousMessages.join('\n'),
      input: currentMessageContent,
    });

    return LangChainAdapter.toDataStreamResponse(stream);
  } catch (error) {
    if (error instanceof Error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }
 
    return new Response(
      JSON.stringify({ error: 'Unknown error occurred' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    );
  }
}