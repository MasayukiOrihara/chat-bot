import { PromptTemplate } from "@langchain/core/prompts";
import { PromptTemplateJson } from "@/contents/type";
import { Message as VercelChatMessage, LangChainAdapter } from "ai";
import {
  Annotation,
  MemorySaver,
  MessageGraph,
  messagesStateReducer,
  StateGraph,
  MessagesAnnotation,
  END,
} from "@langchain/langgraph";

import { getModel, isObject, loadJsonFile } from "@/contents/utils";
import {
  BaseMessage,
  HumanMessage,
  RemoveMessage,
  SystemMessage,
  trimMessages,
} from "@langchain/core/messages";

// チャット形式
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

/**
 * 過去履歴を切り捨てるもの
 * メッセージ数で履歴削除: ConversationBufferWindowMemory
 * トークン数での履歴削除: ConversationTokenBufferMemory
 */
const trimmer = trimMessages({
  maxTokens: 5, // 残す最大トークン数
  strategy: "last",
  tokenCounter: (messages: BaseMessage[]) => messages.length, // メッセージ数でカウント
  includeSystem: true, // システムを含めるか
  allowPartial: false,
  startOn: "human",
});

/**
 * グラフサンプル
 */
/** LLM を呼ぶ関数 */
async function callModel(state: typeof GraphAnnotation.State) {
  console.log("🧠 call model");
  const model = getModel("gpt-4o-mini");

  // 会話履歴削除
  // const trimmed = await trimmer.invoke(state.messages);

  // 会話履歴要約
  let messages;
  const summary = state.summary;
  // 要約が存在する場合システムメッセージとして追加
  if (summary) {
    const systemMessage = `前回の会話の要約: ${summary}`;
    messages = [new SystemMessage(systemMessage), ...state.messages];
  } else {
    messages = state.messages;
  }
  const response = await model.invoke(messages);
  console.log(response.content);

  return { messages: response };
}

/** 会話を行うか要約するかの判断処理 */
async function shouldContenue(state: typeof GraphAnnotation.State) {
  console.log("❓ should contenue");
  const messages = state.messages;

  if (messages.length > 6) return "summarize";
  return END;
}

/** 会話の要約処理 */
async function summarizeConversation(state: typeof GraphAnnotation.State) {
  console.log("📃 summarize conversation");
  const model = getModel("gpt-4o-mini");
  const summary = state.summary;

  let summaryMessage;
  if (summary) {
    summaryMessage = `これまでの会話の要約: ${summary}\n\n上記の新しいメッセージを考慮して要約を拡張してください。: `;
  } else {
    summaryMessage =
      "上記の会話の要約を会話の流れを重視して箇条書きで作成してください: ";
  }

  // 要約処理
  const messages = [...state.messages, new SystemMessage(summaryMessage)];
  const response = await model.invoke(messages);
  console.log(response.content);

  const deleteMessages = messages
    .slice(0, -2)
    .map((m) => new RemoveMessage({ id: m.id! }));
  return { summary: response.content, messages: deleteMessages };
}

// アノテーションの追加
const GraphAnnotation = Annotation.Root({
  summary: Annotation<string>(),
  ...MessagesAnnotation.spec,
});

// グラフ
const workflow = new StateGraph(GraphAnnotation)
  // ノード追加
  .addNode("conversation", callModel)
  .addNode("summarize", summarizeConversation)

  // エッジ追加
  .addEdge("__start__", "conversation")
  .addConditionalEdges("conversation", shouldContenue)
  .addEdge("summarize", END);

// 記憶の追加
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });

/**
 * チャット応答AI（記憶・モデル変更対応済み）
 * @param req
 * @returns
 */
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const modelName = body.model ?? "fake-llm";

    // 過去の履歴{chat_history}
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

    // メッセージ{input}
    const currentMessageContent = messages[messages.length - 1].content;

    //プロンプトテンプレートの作成
    const template = await loadJsonFile<PromptTemplateJson[]>(
      "src/data/prompt-template.json"
    );
    if (!template.success) {
      return new Response(JSON.stringify({ error: template.error }), {
        status: 500,
        headers: { "Content-type": "application/json" },
      });
    }
    // プロンプトテンプレートの抽出
    const found = template.data.find(
      (obj) => isObject(obj) && obj["name"] === "api-chat-aikato"
    );
    if (!found) {
      throw new Error("テンプレートが見つかりませんでした");
    }

    /**
     * ConversationBufferMemoryにあたるもの
     * 履歴全部保有
     */
    const config = { configurable: { thread_id: "abc123" } }; // ID で会話履歴を参照

    // 既存メッセージの取得
    const state = await app.getState(config);
    const existingMessages = state?.values?.messages || [];
    // console.log(existingMessages);

    // 履歴を含ませる
    const graphResult = await app.invoke(
      {
        messages: currentMessageContent,
      },
      config
    );

    // console.log("langgraph: " + graphResult.messages[graphResult.messages.length - 1].content);

    // モデルの指定
    const model = getModel(modelName);

    const prompt = PromptTemplate.fromTemplate(found.template);
    const chain = prompt.pipe(model);

    const stream = await chain.stream({
      history: formattedPreviousMessages.join("\n"),
      input: currentMessageContent,
    });

    return LangChainAdapter.toDataStreamResponse(stream);
  } catch (error) {
    if (error instanceof Error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({ error: "Unknown error occurred" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
