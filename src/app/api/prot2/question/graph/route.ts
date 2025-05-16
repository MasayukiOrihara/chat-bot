import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Annotation, messagesStateReducer, StateGraph } from "@langchain/langgraph";
import { MessagesAnnotation } from "@langchain/langgraph";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
import { isObject, loadJsonFile } from "@/contents/utils";
import { PromptTemplateJson, RiddleJson } from "@/contents/type";

// モデルのセット（OPENAI固定）
const model = new ChatOpenAI({ apiKey: process.env.OPENAI_API_KEY, model: "gpt-4o", temperature: 0.3 });
// なぞなぞモードのフラグ状態保存
let wantRiddleState = false;
let riddleIdState = 0;
// なぞなぞ問題の読み込み
const questions = await loadJsonFile<RiddleJson[]>('src/data/riddle.json');
// プロンプトの読み込み
const template = await loadJsonFile<PromptTemplateJson[]>('src/data/prompt-template.json');
// プロンプトチェック用関数
function createErrorResponse(message: string, statusCode: number = 500): Response {
  return new Response(JSON.stringify({ error: message }), {
    status: statusCode,
    headers: { 'Content-Type': 'application/json' },
  });
}
// メッセージ形式
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

/**
 * なぞなぞを始めるかどうかの問い
 * @param param 
 * @returns 
 */
async function detectIntent({ messages }: typeof StateAnnotation.State) {
  // プロンプトテンプレートの抽出
  if (!template.success) return createErrorResponse(template.error);
  const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph-detectIntent');
  const foundYes = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph-detectIntent-yes');
  const foundNo = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph-detectIntent-no');
  if (!found || !foundYes || !foundNo) throw new Error('テンプレートが見つかりませんでした');

  // 最初にLLMに聞いてみる
  const detectIntentPrompt = PromptTemplate.fromTemplate(found.template);
  const detectIntentChain = detectIntentPrompt.pipe(model);
  const userMsg = messages[messages.length - 1].content;
  const response = await detectIntentChain.invoke({ input: userMsg });

  // 答え次第では始める
  const answerText =
    typeof response.content === "string"
      ? response.content
      : response.content.map((c: any) => c.text ?? "").join("");

  const want = answerText.trim().toUpperCase().includes("YES");
  return {
    messages: [...messages, new AIMessage(`${want ? foundYes.template : "世間話"}`)],
    wantRiddle: want,
  };
}


/**
 * 問題の提示と選定
 */
async function selectAndPresentQuestion({ messages, riddleId}: typeof StateAnnotation.State) {
  // なぞなぞがちゃんと読み込まれているかの確認
  if (!questions.success) return createErrorResponse(questions.error);
  // テンプレートチェック
  if (!template.success) return createErrorResponse(template.error);
  const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph-selectAndPresentQuestion');
  if (!found) throw new Error('テンプレートが見つかりませんでした');

  // ランダムに問題を選択
  const decidedRiddleId = Math.floor(Math.random() * questions.data.length);
  const randomQuestion = questions.data[decidedRiddleId];
  messages[messages.length -1].content += found.template.replace("{question}", randomQuestion.pageContent);

  return { 
    messages: [...messages],
    riddleId: decidedRiddleId
  };
}


/**
 * ヒント計算ノード
 */
async function giveHint({ messages, riddleId }: typeof StateAnnotation.State)  {
  const userMessage = messages[messages.length - 1];
  const userAnswer = typeof userMessage.content === "string"
    ? userMessage.content
    : userMessage.content.map((c: any) => c.text ?? "").join("");
  
  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-3-large"
  });

  // なぞなぞがちゃんと読み込まれているかの確認
  if (!questions.success) return createErrorResponse(questions.error);
  // テンプレートチェック
  if (!template.success) return createErrorResponse(template.error);
  const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph-giveHint');
  if (!found) throw new Error('テンプレートが見つかりませんでした');

  const riddleAnswer = [questions.data[riddleId].metadata.answer];
  const RiddleMetadata = questions.data[riddleId].metadata;
  
  const vectorStore = await MemoryVectorStore.fromTexts(riddleAnswer, RiddleMetadata, embeddings);
  const result = await vectorStore.similaritySearchWithScore(userAnswer, 1);
  const [bestMatch, score] = result[0];
  console.log("score: " + score);

  // ユーザーの回答、ヒントを使ってヒントを与える
  const hint = found
    .template
    .replace("{userAnswer}", userAnswer)
    .replace("{hint}", questions.data[riddleId].metadata.hint);
 
  //  正解（閾値は順次変える）
  if (score >= 0.75) return {clear: true};

  return {
    messages: [...messages, new AIMessage(hint)],
  };
}


/** 
 * ユーザーの回答前の状態保存ノード
 */ 
async function receiveAnswerExit({ wantRiddle, riddleId }: typeof StateAnnotation.State) {
  wantRiddleState = wantRiddle;
  riddleIdState = riddleId;
}

/** 
 * 前のターンの終わり状態を次ターンに渡す状態保存ノード
 */ 
async function startState({ wantRiddle, riddleId }: typeof StateAnnotation.State) {
  return { 
    wantRiddle: wantRiddleState,
    riddleId: riddleIdState
  };
}

/**
 * 終わりノード
 */
async function exit({ wantRiddle }: typeof StateAnnotation.State) {
  wantRiddleState = false;
  riddleIdState = 0;
}

/**
 * 正解終わりノード
 */
async function correcrAnswer({ messages }: typeof MessagesAnnotation.State) {
  // テンプレートチェック
  if (!template.success) return createErrorResponse(template.error);
  const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph-correcrAnswer');
  if (!found) throw new Error('テンプレートが見つかりませんでした');

  return {
    messages: [...messages, new AIMessage(found.template)],
  };
}


/**
 * グラフ定義
 */
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  wantRiddle: Annotation<boolean>({
    value: (state: boolean = false, action: boolean) => action,
    default: () => wantRiddleState,
  }),
  clear: Annotation<boolean>({
    value: (state: boolean = false, action: boolean) => action,
    default: () => false,
  }),
  riddleId: Annotation<number>({
    value: (state: number = 0, action: number) => action,
    default: () => 0,
  }),
});

const graph = new StateGraph(StateAnnotation)
  .addNode("start", startState)
  .addNode("detect", detectIntent)
  .addNode("question", selectAndPresentQuestion)
  .addNode("recExit", receiveAnswerExit)
  .addNode("hint", giveHint)
  .addNode("correcr", correcrAnswer)
  .addNode("exit", exit)

  .addEdge("__start__", "start")
  .addConditionalEdges("start", (state) => state.wantRiddle ? "hint" : "detect")
  .addConditionalEdges("hint" , (state) => state.clear ? "correcr" : "recExit")
  .addEdge("correcr", "exit")
  .addConditionalEdges("detect", (state) => state.wantRiddle ? "question" : "exit")
  .addEdge("question", "recExit")
  .compile();


/**
 * ストリーミング
 */
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];

    // 直近のメッセージを取得
    const userMessage = messages.at(-1).content;
    if (!userMessage) {
      return new Response(JSON.stringify({ error: 'No message provided' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

    // ユーザーのメッセージを受け取る
    const result = await graph.invoke({
      messages: [new HumanMessage(userMessage)],
    });
    
    console.log("AI: " + result.messages[1].content);

    // テンプレートチェック
    if (!template.success) return createErrorResponse(template.error);
    const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-question-graph');
    if (!found) throw new Error('テンプレートが見つかりませんでした');

    // ストリーミング応答を取得
    const prompt = PromptTemplate.fromTemplate(found.template);
    const chain = prompt.pipe(model);
    const stream = await chain.stream({ 
      userMessage: userMessage,
      aiMessage: result.messages[1].content,
      chatHistory: formattedPreviousMessages
    });

  // LangChainのストリーム出力を Webレスポンス用のストリーム形式に変換する
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