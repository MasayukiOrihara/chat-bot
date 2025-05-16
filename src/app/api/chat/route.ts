import { PromptTemplateJson } from '@/contents/type';
import { getModel, isObject, loadJsonFile } from '@/contents/utils';
import { PromptTemplate } from '@langchain/core/prompts';
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
 
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