import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage, ThinkingMessage } from "../../components/custom/message";
import { useScrollToBottom } from '@/components/custom/use-scroll-to-bottom';
import { useState } from "react";
import { message } from "../../interfaces/interfaces";
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import { v4 as uuidv4 } from 'uuid';

export function Chat() {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  async function handleSubmit(text?: string) {
    if (isLoading) return;
  
    const messageText = text || question;
    setIsLoading(true);
  
    const traceId = uuidv4();
    setMessages(prev => [...prev, { content: messageText, role: "user", id: traceId }]);
    setQuestion("");
  
    try {
      const response = await fetch("http://localhost:8090/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer "AIzaSyCjyVoiwVAzcGZ8-dtSfdEJnNADqSZnwMY"`,
        },
        body: JSON.stringify({ query: messageText }),
      });
  
      console.log("Response received:", response);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend error:", errorText);
        throw new Error(`HTTP error! Status: ${response.status} - ${errorText}`);
      }
  
      const data = await response.json();
      console.log("API response data:", data); // Add this log
  
      setIsLoading(false);
  
      const replyContent = data.reply || data.result;
      if (replyContent) {
        setMessages(prev => [
          ...prev,
          { content: replyContent, role: "assistant", id: uuidv4() }
        ]);
      } else {
        console.log("No reply or result in response");
      }
    } catch (error) {
      console.error("Error fetching response:", error);
      setIsLoading(false);
      setMessages(prev => [
        ...prev,
        { content: "Sorry, there was an error processing your request.", role: "assistant", id: uuidv4() }
      ]);
    }
  }
  
  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <Header />
      <div className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4" ref={messagesContainerRef}>
        {messages.length === 0 && <Overview />}
        {messages.map((message, index) => (
          <PreviewMessage key={index} message={message} />
        ))}
        {isLoading && <ThinkingMessage />}
        <div ref={messagesEndRef} className="shrink-0 min-w-[24px] min-h-[24px]" />
      </div>
      <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
