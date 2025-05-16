import { Flags, PromptTemplateJson, States } from '@/contents/type';
import { isObject, loadJsonFile } from '@/contents/utils';
import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { PromptTemplate } from '@langchain/core/prompts';
import { Annotation, messagesStateReducer, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Message as VercelChatMessage, LangChainAdapter } from 'ai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';


// モデルのセット（OPENAI固定）
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o",
  temperature: 0.3
});
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-3-large"
  });
// プロンプトの読み込み
const template = await loadJsonFile<PromptTemplateJson[]>('src/data/prompt-template.json');
// プロンプトチェック用関数
function createErrorResponse(message: string, statusCode: number = 500): Response {
  return new Response(JSON.stringify({ error: message }), {
    status: statusCode,
    headers: { 'Content-Type': 'application/json' },
  });
}
// チャット形式
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

// フラグ管理
const transitionStates = {
  isStarted: false,
  isTarget: false,
  isReason: false,
  checkTarget: false,
  checkReason: false,
}
const reasonFlags = {
  deadline: false,
  function: false,
  quality: false,
}


/** 初めの状態変更ノード */
async function setState({ messages }: typeof StateAnnotation.State) {
  console.log("setState:");

  // 現在の状態
  return {
    transition: {...transitionStates},
  }
}

/** 報連相ワークは始まっているかの状態確認ノード */
async function isProcessStarted({ messages, transition }: typeof StateAnnotation.State) {
  console.log("isProcessStarted: " + transition.isStarted);

  transitionStates.isStarted = true;
}

/** 開発の仕事を想像させるノード */
async function explainEngineeringTopics({ messages }: typeof StateAnnotation.State) {
  console.log("explainEngineeringTopics");

  // AIに開発についての話をさせる
  return {
    messages: [...messages, new AIMessage("開発の仕事について語ってください。\n")],
  };
}

/** 1つ目の問題「誰のため」は正解したかの状態確認ノード */
async function checkTargetMatch({ messages, transition }: typeof StateAnnotation.State) {
  console.log("checkTargetMatch: " + transition.checkTarget);

  const userMessage = messages[messages.length - 1];
  const userAnswer = typeof userMessage.content === "string"
    ? userMessage.content
    : userMessage.content.map((c: any) => c.text ?? "").join("");
  
  const targetAnswer = ["リーダー", "上司"];
  const targetMetadata = [
    { id: "1", quwstion_id: "1", question: "報連相は誰のためか"},
    { id: "2", quwstion_id: "1", question: "報連相は誰のためか"}
  ];

  const vectorStore = await MemoryVectorStore.fromTexts(targetAnswer, targetMetadata, embeddings);
  const result = await vectorStore.similaritySearchWithScore(userAnswer, 1);
  const [bestMatch, score] = result[0];
    console.log("score: " + score + ", match: "+ bestMatch.pageContent);

  // 正解パターン
  if (score >= 0.8) {
    transitionStates.checkTarget = true;
    return {
      transition: {...transitionStates}, 
    };
  }
}

/** 1つ目の問題「誰のため」を聞くノード */
async function questionTarget({ messages }: typeof StateAnnotation.State) {
  console.log("questionTarget");

  // 問題を出してもらう
  messages[messages.length -1].content += "上記について話したのち、生徒に下記の質問をしてください。\n開発の仕事を想定し、報連相は誰のためのものか";
  return {
    messages: [...messages]
  };
}

/** 1つ目の問題「誰のため」ヒントノード */
async function giveTargetHint({ messages }: typeof StateAnnotation.State) {
  console.log("giveTargetHint");

  // 答えに対してヒントを与える
  return {
    messages: [...messages, new AIMessage("答えを外したのであなたはユーザーを諫め、[ヒント]をあげてください。\n")],
  };
}

/** 1つ目の問題「誰のため」をクリアしたか状態確認ノード */
async function isTargetCleared({ messages, transition }: typeof StateAnnotation.State) {
  console.log("isTargetCleared: " + transition.isTarget);

  transitionStates.isTarget = true;
}

/** 1つ目の問題「誰のため」を正解したことをほめるノード */
async function praiseTargetCleared({ messages }: typeof StateAnnotation.State) {
  console.log("praiseTargetCleared");

  return {
    messages: [...messages, new AIMessage("問題に正解したのであなたはユーザーを褒めてください。\n")],
  };
}

/** 2つ目の問題「なぜリーダーのためなのか」は正解したかの状態確認ノード */
async function checkReasonMatch({ messages, transition, hit }: typeof StateAnnotation.State) {
  console.log("checkReasonMatch: " + transition.checkReason);

  const userMessage = messages[messages.length - 1];
  const userAnswer = typeof userMessage.content === "string"
    ? userMessage.content
    : userMessage.content.map((c: any) => c.text ?? "").join("");
  
  const targetAnswer = ["納期や期限を守る", "機能に過不足がない", "品質が良く不具合がない"];
  const targetMetadata = [
    { id: "1", quwstion_id: "2", question: "報連相はなぜリーダーのためなのか"},
    { id: "2", quwstion_id: "2", question: "報連相はなぜリーダーのためなのか"},
    { id: "3", quwstion_id: "2", question: "報連相はなぜリーダーのためなのか"}
  ];

  const vectorStore = await MemoryVectorStore.fromTexts(targetAnswer, targetMetadata, embeddings);
  const result = await vectorStore.similaritySearchWithScore(userAnswer, 3);

  // 上位３件を確認
  for(const [bestMatch, score] of result){
    console.log("score: " + score + ", match: "+ bestMatch.pageContent);

    // スコアが閾値以上の場合3つのそれぞれのフラグを上げる
    if (score >= 0.6){
      if(bestMatch.pageContent === targetAnswer[0]){
        reasonFlags.deadline = true;
      }
      if(bestMatch.pageContent === targetAnswer[1]){
        reasonFlags.function = true;
      }
      if(bestMatch.pageContent === targetAnswer[2]){
        reasonFlags.quality = true;
      }
      hit = true;
    }
  }
  console.log("納期: " + reasonFlags.deadline);
  console.log("機能: " + reasonFlags.function);
  console.log("品質: " + reasonFlags.quality);

  // 全正解
  if ( Object.values(reasonFlags).every(Boolean) ) {
    transitionStates.checkReason = true;
    return {
      transition: {...transitionStates}
    };
  }

  return {
    hit,
  }
}

/** 2つ目の問題「なぜリーダーのためなのか」を聞くノード */
async function questionReason({ messages }: typeof StateAnnotation.State) {
  console.log("questionReason");

  // 問題を出してもらう
  messages[messages.length -1].content += "上記を実施したのち、[報連相はリーダーのため]ということを前提に下記の質問をしてください。\n[報連相はなぜリーダーのためのものなのか]";
  return {
    messages: [...messages]
  };
}

/** 2つ目の問題「なぜリーダーのためなのか」ヒントノード */
async function giveReasonHint({ messages }: typeof StateAnnotation.State) {
  console.log("giveReasonHint");

  // 答えに対してヒントを与える
  return {
    messages: [...messages, new AIMessage("ユーザーは答えを外したのであなたはユーザーを諫め、[ヒント]をあげてください。\n")],
  };
}

/** 2つ目の問題「なぜリーダーのためなのか」をクリアしたか状態確認ノード */
async function isReasonCleared({ messages, transition }: typeof StateAnnotation.State) {
  console.log("isReasonCleared:" + transition.isReason);

  transitionStates.isReason = true;
}

/** 2つ目の問題「なぜリーダーのためなのか」を正解したことをほめるノード */
async function praiseReasonCleared({ messages }: typeof StateAnnotation.State) {
  console.log("praiseReasonCleared");

  const userMessage = messages[messages.length - 1];
  const userAnswer = typeof userMessage.content === "string"
    ? userMessage.content
    : userMessage.content.map((c: any) => c.text ?? "").join("");

  return {
    messages: [...messages, new AIMessage(`あなたはユーザーに「${userAnswer}」が正解だったことを報告してください。\n`)],
  };
}

/** 2つ目の問題「なぜリーダーのためなのか」を全問正解したことをほめるノード */
async function praiseReasonAllCleared({ messages }: typeof StateAnnotation.State) {
  console.log("praiseReasonAllCleared");

  return {
    messages: [...messages, new AIMessage(`問題に正解したのであなたはユーザーを褒めてください。\n`)],
  };
}

/** なぜ報連相が必要になるのかを解説するノード */
async function explainNewsletter({ messages }: typeof StateAnnotation.State) {
  console.log("explainNewsletter");


  messages[messages.length -1].content += "今までの会話の流れを受けてなぜ報連相が必要なのか解説してください。また解説の後ユーザーにこの講習を終えての所感を聞いてください。\n";
  return {
    messages: [...messages]
  };
}

/** 終了前のノード */
async function exit({ messages }: typeof StateAnnotation.State) {
  console.log("exit");
}

/** 結果を保存して終了ノード */
async function isProcessEnd({ messages }: typeof StateAnnotation.State) {
  console.log("isProccessEnd");

  return {
    messages: [...messages, new AIMessage("報連相の講習が終了したことを伝えてください。\n")],
  };
}

/**
 * グラフ定義
 * messages: 今までのメッセージを保存しているもの
 */
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  transition: Annotation<States>({
    value: (state: States = {
      isStarted: false,
      isTarget: false,
      isReason: false,
      checkTarget: false,
      checkReason: false,
    }, action: Partial<States>) => ({
      ...state,
      ...action,
    }),
  }),
  flags: Annotation<Flags>({
    value: (state: Flags = {
      deadline: false,
      function: false,
      quality: false,
    }, action: Partial<Flags>) => ({
      ...state,
      ...action,
    }),
  }),
  hit: Annotation<boolean>({
    value: (state: boolean = false, action: boolean) => action,
  }),
});

const graph = new StateGraph(StateAnnotation)
  .addNode("set", setState)
  .addNode("is1", isProcessStarted)
  .addNode("is2", isTargetCleared)
  .addNode("is3", isReasonCleared)
  .addNode("is4", isProcessEnd)
  .addNode("check1", checkTargetMatch)
  .addNode("check2", checkReasonMatch)
  .addNode("exit", exit)
  .addNode("praise1", praiseTargetCleared)
  .addNode("praise2", praiseReasonCleared)
  .addNode("praise3", praiseReasonAllCleared)
  .addNode("explainStart", explainEngineeringTopics)
  .addNode("explainEnd", explainNewsletter)
  .addNode("question1", questionTarget)
  .addNode("question2", questionReason)
  .addNode("hint1", giveTargetHint)
  .addNode("hint2", giveReasonHint)

  .addEdge("__start__", "set")
  .addConditionalEdges("set", (state) => {
    if (state.transition.isReason) return "is4";
    if (state.transition.isTarget) return "is2";
    return "is1";
  })
  .addConditionalEdges("is1", (state) => state.transition.isStarted ? "check1" : "explainStart")
  .addEdge("explainStart", "question1")
  .addEdge("question1", "exit")
  .addConditionalEdges("check1", (state) => state.transition.checkTarget ? "is2" : "hint1")
  .addEdge("hint1", "question1")
  .addConditionalEdges("is2", (state) => state.transition.isTarget ? "check2" : "praise1")
  .addEdge("praise1", "question2")
  .addConditionalEdges("check2", (state) => {
    if (state.transition.checkReason) return "is3";
    if (state.hit) return "praise2";
    return "hint2";
  })
  .addEdge("hint2", "question2")
  .addEdge("praise2", "question2")
  .addEdge("question2", "exit")
  .addEdge("is3", "praise3")
  .addEdge("praise3", "explainEnd")
  .addEdge("explainEnd", "exit")
  .addEdge("is4", "exit")
  .addEdge("exit", "__end__")
  .compile();



/**
 * 報連相ワークAI
 * 
 * @param req 
 * @returns 
 */
export async function POST(req: Request) {
  try{
    const body = await req.json();
    const messages = body.messages ?? [];

    // 過去の履歴
    const formattedPreviousMessages =  messages.slice(0, -1).map(formatMessage)
    // 直近のメッセージを取得
    const userMessage = messages.at(-1).content;
    if (!userMessage) {
      return new Response(JSON.stringify({ error: 'No message provided' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // langgraph
    const result = await graph.invoke({
      messages: [new HumanMessage(userMessage)],
    });

    console.log("langgraph: " + result.messages[1].content);
        
    // テンプレートチェック
        if (!template.success) return createErrorResponse(template.error);
        const found = template.data.find(obj => isObject(obj) && obj['name'] === 'api-prot2-aikato');
        if (!found) throw new Error('テンプレートが見つかりませんでした');
    
        // ストリーミング応答を取得
        const prompt = PromptTemplate.fromTemplate(found.template);
        const chain = prompt.pipe(model);
        const stream = await chain.stream({ 
          chatHistory: formattedPreviousMessages,
          userMessage: userMessage,
          aiMessage: result.messages[1].content,
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