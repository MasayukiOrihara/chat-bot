import { FakeListChatModel } from "@langchain/core/utils/testing";
import { HumanMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";

/**
 * FakeListChatModel を使用すると、順序付けられた定義済みの応答をシミュレートできます。
 */

const chat = new FakeListChatModel({
  responses: ["後で折り返しご連絡します。", "You 'console' them!"],
});

const firstMessage = new HumanMessage("JavasSript のジョークを聞きたいですか?");
const secondMessage = new HumanMessage(
  "JavaScript 開発者を元気づけるにはどうすればよいでしょうか?"
);
const firstResponse = await chat.invoke([firstMessage]);
const secondResponse = await chat.invoke([secondMessage]);

// console.log({ firstResponse });
// console.log({ secondResponse });

/**
 * FakeListChatModel は、ストリーミングされた応答をシミュレートするためにも使用できます。
 */

const stream = await chat
  .pipe(new StringOutputParser())
  .stream(`JavasSript のジョークを聞きたいですか?`);
const chunks = [];
for await (const chunk of stream) {
  chunks.push(chunk);
}

// console.log(chunks.join(""));

/**
 * FakeListChatModel は、同期応答またはストリーミング応答のいずれかの遅延をシミュレートするためにも使用できます。
 */

const slowChat = new FakeListChatModel({
  responses: ["Because Oct 31 equals Dec 25", "You 'console' them!"],
  sleep: 1000,
});

const thirdMessage = new HumanMessage(
  "Why do programmers always mix up Halloween and Christmas?"
);
const slowResponse = await slowChat.invoke([thirdMessage]);
// console.log({ slowResponse });

const slowStream = await slowChat
  .pipe(new StringOutputParser())
  .stream("How do you cheer up a JavaScript developer?");
const slowChunks = [];
for await (const chunk of slowStream) {
  slowChunks.push(chunk);
}

// console.log(slowChunks.join(""));