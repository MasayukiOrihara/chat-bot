import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from '@langchain/openai';
import { LangChainAdapter } from 'ai';
import { TavilySearchAPIRetriever } from "@langchain/community/retrievers/tavily_search_api";
import { RunnableSequence } from '@langchain/core/runnables';
import { getModel } from '@/contents/utils';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";


/**
 * チャット応答AI（記憶なし・モデル変更対応済み）
 * 質問に対してweb検索を行い、結果をvectolにして保存
 * 保存したベクトルデータから質問文に類似したテキストを抜き出し質問に答える
 * 
 * @param req 
 * @returns 
 */
export async function POST(req: Request) {
  try{
    const body = await req.json();
    const messages = body.messages ?? [];
    const modelName = body.model ?? 'fake-llm';

    // メッセージ{input}
    const currentMessageContent = messages[messages.length - 1].content;

    /** 回答取得準備 */
    const model = getModel(modelName);
    const outputParser = new StringOutputParser();

    /** 質問文かどうかを判断させる */
    const desideTemplate = "以下の[発言]が「質問」という意図を含むかを「YES」または「NO」で答えてください。\n[発言]: {input}\n答え: ";
    const desidePrompt = PromptTemplate.fromTemplate(desideTemplate);
    const desideChain = desidePrompt.pipe(model).pipe(outputParser);

    const getDeside = await desideChain.invoke({
      input: currentMessageContent,
    });
    // console.log("こたえ: " + getDeside);

    /** ユーザーの質問を推論 */
    const imageTemplate = "ユーザーの[質問文]の[回答]を推論して一行でまとめてください。\n[質問文]: {input}\n[回答]: ";
    const imagePrompt = PromptTemplate.fromTemplate(imageTemplate);
    const imageChain = imagePrompt.pipe(model).pipe(outputParser);

    const getImage = await imageChain.invoke({
      input: currentMessageContent,
    });
    // console.log("推論: " + getImage);

    // 質問だった場合、推論された答えの方をクエリに採用する
    let query = `${currentMessageContent} site:docs.oracle.com`;
    if (getDeside === 'YES'){
      query = `${getImage} site:docs.oracle.com`;
    }
    console.log("クエリ: " + query);


    /** 文章検索の準備 */
    const tavily = new TavilySearchAPIRetriever({
      apiKey: process.env.TAVILY_API_KEY,
      k: 10,
      includeGeneratedAnswer: true,
    });

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 50,
    });
  
    const embeddings = new OpenAIEmbeddings({
      modelName: "text-embedding-3-large",
      apiKey: process.env.OPENAI_API_KEY
    });
        
    /** 検索の実行 */
    const docs = await tavily.invoke(query);
    const splitDocs = await splitter.splitDocuments(docs);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      splitDocs,
      embeddings
    );
    const response = await vectorStore.similaritySearchWithScore(query, 3);
        
    // 抽出
    const context = response.map(([doc, score]) => ({
      content: doc.pageContent,
      score: score,
    }));
    const data = context.map(item => item.content);
    
    const outputTemplate = "あなたは情報系の講師です。\n\n情報: {data}\nこれらの情報をもとにユーザーの質問に答えてください。\n\n質問: {input} \n答え: ";
    const outputPrompt = PromptTemplate.fromTemplate(outputTemplate);
    const outputChain = outputPrompt.pipe(model).pipe(outputParser);

    const stream = await outputChain.stream({
      data: data,
      input: query,
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