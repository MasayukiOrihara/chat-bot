import { PromptTemplate } from '@langchain/core/prompts';
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
import { StringOutputParser } from '@langchain/core/output_parsers';

import { cutKeyword, getModel, isObject, loadJsonFile } from '@/contents/utils';
import { PromptTemplateJson } from '@/contents/type';
 
// チャット形式
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};
 
/**
 * チャットボット(折原AI)
 * 報告に対するビジネスマナーの指摘
 * 【　4月末リファクタリング済み　】
 * @param req 
 * @returns 
 */
export async function POST(req: Request) {
  try {
    // チャット履歴と選択したモデル
    const body = await req.json();
    const messages = body.messages ?? [];
    const modelName = body.model ?? 'fake-llm';
 
    // 過去の履歴 {chat_history}用
    const formattedPreviousMessages = messages
      .slice(0, -1)
      .map(formatMessage);
    //現在の履歴 {input}用 
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
    const foundCharacter = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot1-character');
    const foundScore = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot1-score-and-point');
    if (!foundCharacter || !foundScore) {
      throw new Error('テンプレートが見つかりませんでした');
    }
    // プロンプトの準備
    const characterPrompt = PromptTemplate.fromTemplate(foundCharacter.template);
    const scorePrompt = PromptTemplate.fromTemplate(foundScore.template);

    // 出力形式の指定
    const outputParser = new StringOutputParser();
 
    // モデルの指定
    const model = getModel(modelName);

    // プロンプトとモデルをつなぐ
    const firstChain = scorePrompt.pipe(model).pipe(outputParser);
    const secondChain = characterPrompt.pipe(model).pipe(outputParser);

    // 1回目の質問
    const getScore = await firstChain.invoke({
      input: currentMessageContent,
    });
    console.log("score: " + getScore);

    // 文字列の切り出し
    const score = cutKeyword(getScore, "総合点: ");
    const checkPoint = cutKeyword(score, "指摘ポイント: ");
 
    // 2回目の質問
    const stream = await secondChain.stream({
      chat_history: formattedPreviousMessages.join('\n'),
      input: currentMessageContent,
      score: score,
      prompt1_output: checkPoint,
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