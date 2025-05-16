import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
import { JSONLoader } from 'langchain/document_loaders/fs/json';
import { formatDocumentsAsString } from 'langchain/util/document';

// JSONを読み込む形式を作成
const loader = new JSONLoader('src/data/city.json', [
    '/city',
    '/slug',
    '/han',
    '/mayor',
    '/area_code',
    '/population',
    '/city_emblem_image',
]);

export const runtime = 'nodejs';
 
// メッセージの整形
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};
 
// プロンプトテンプレート
const TEMPLATE = `次のデータベースに基づいてユーザの質問に答えてください。もしその答えがデータベースに含まれていない場合は、情報がないことを正直に返答してください。
==============================
Context: {context}
==============================
Current conversation: {chat_history}
 
user: {question}
assistant:`;
 
export async function POST(req: Request) {
  try {
    const body = await req.json();
    // nullの場合は空の配列を、通常は受け取ったリクエスト
    const messages = body.messages ?? [];
 
    // ひとつ前ののメッセージ
    const formattedPreviousMessages = messages
      .slice(0, -1)
      .map(formatMessage);
 
    // 直近の要素を取得
    const currentMessageContent = messages[messages.length - 1].content;
 
    // JSONをここでロード
    let docs = [];
    try {
      docs = await loader.load();
    } catch (error) {
      console.error('Error loading JSON data:', error);
      return new Response(
        JSON.stringify({ error: 'Error loading data from JSON file.' }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        },
      );
    }
 
    const prompt = PromptTemplate.fromTemplate(TEMPLATE);
 
    const model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o',
      temperature: 0.8,
    });
 
    // チェインを作成
    const chain = RunnableSequence.from([
      {
        question: (input) => input.question,
        chat_history: (input) => input.chat_history,
        context: () => formatDocumentsAsString(docs),
      },
      prompt,
      model,
    ]);
 
    // ストリームを作成（最終出力）
    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join('\n'),
      question: currentMessageContent,
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