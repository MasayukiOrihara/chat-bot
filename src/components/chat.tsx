'use client';
 
import { useChat } from '@ai-sdk/react';
import { SendHorizontalIcon } from 'lucide-react';
import Image from 'next/image';
import { useEffect, useRef } from 'react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';

import { Button } from './ui/button';
import { Input } from './ui/input';

type Model = 'gpt-4o' | 'claude-haiku' | 'fake-llm';

interface ChatProps {
  model: Model;
}
 
export const Chat: React.FC<ChatProps> = ({ model }) => {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    // APIの読み込み
    //api: 'api/prot1',
    // api: 'api/prot2/question/graph',
    api: 'api/prot2',
    // api: 'api/chat',
    // api: 'api/langchain/rag/web',
    //api: 'api/langchain',
    //api: 'api/chat/with/json',
    // api: 'api/langchain/memory',
    // api: 'api/langgraph/',
    //api: 'api/langgraph/reactAgent',
    body: {
      model,
    },
    onError: (e) => {
      toast.error('エラーが発生しました');
      console.log(e);
    },
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
 
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 回答が画像だった場合の処理を含む
  const getAnswer = (messageId: string, value: string, index: number) => {
    const imageMarkdownLinkRegex = /!\[([^\]]*)\]\((https?:\/\/[^\)]+)\)/g;
    const matches = [];
    let match;

    while ((match = imageMarkdownLinkRegex.exec(value)) !== null) {
      matches.push(match);
    }

    if (matches.length > 0) {
      const imageElements = matches.map((m, index) => (
        <div key={index}>
          <Image
            src={m[2]}
            alt={m[1] || 'image'}
            width={128}
            height={128}
            className="p-4"
          />
        </div>
      ));

      const nonImageText = value.replace(imageMarkdownLinkRegex, '');

      return (
        <div
          key={`${messageId}-${index}`}
          className="break-words overflow-hidden"
        >
          <p style={{ overflowWrap: 'anywhere' }}>{nonImageText}</p>
          {imageElements}
        </div>
      );
    }

    return (
      <div
        key={`${messageId}-${index}`}
        className="break-words overflow-hidden"
      >
        <p style={{ overflowWrap: 'anywhere' }}>{value}</p>
      </div>
    );
  };

  return (
    <div className="mt-2 flex flex-col w-full max-w-2xl h-full mx-auto gap-2 bg-zinc-900 overflow-hidden">
      <div className="flex flex-col flex-1 overflow-y-auto mb-18">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
                        'whitespace-pre-wrap px-5 py-3 rounded-lg mb-2 mx-8 flex gap-2',
                        message.role === 'user'
                        ? 'bg-zinc-800 text-neutral-500 self-start'
                        : 'text-gray-400 self-end',
                      )}
          >
            {message.role === 'assistant' && (
              <div className="h-8 px-3 py-2 font-bold text-xs rounded-lg bg-[#ff6467]/20 text-zinc-500 w-auto whitespace-nowrap">
                回答
              </div>
            )}
            {message.parts.map((part, i) => {
              switch (part.type) {
              case 'text':
                return getAnswer(message.id, part.text, i);
              }
            })}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
  
      <form
        onSubmit={handleSubmit}
        className="fixed bottom-2 w-full max-w-2xl p-4 bg-zinc-900"
      >
        <div className="flex w-full gap-4">
          <Input
          className="bg-zinc-800 w-full p-2 h-10 border border-zinc-700 rounded shadow-xl text-white placeholder:text-neutral-400"
          value={input}
          placeholder="質問をしてください..."
          onChange={handleInputChange}
          />

          <Button
          type="submit"
          className="w-18 h-10 bg-[#00bc7d] text-white p-2 rounded hover:bg-emerald-900 hover:cursor-pointer hover:text-white/40"
          >
          <SendHorizontalIcon />
          </Button>
        </div>
      </form>
    </div>
  );
};