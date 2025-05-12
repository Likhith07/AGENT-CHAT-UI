import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext, StateType } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message, Thread as SDKThread } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";
import { GitHubSVG } from "../icons/github";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

function OpenGitHubRepo() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href="https://github.com/langchain-ai/agent-chat-ui"
            target="_blank"
            className="flex items-center justify-center"
          >
            <GitHubSVG
              width="24"
              height="24"
            />
          </a>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Open GitHub repo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadIdInternal] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages: Message[] = stream.messages ?? [];
  const isLoading = stream.isLoading;
  const apiUrl = stream.apiUrl;

  // Log message updates
  useEffect(() => {
    console.log("[Thread Component] stream.messages updated:", JSON.stringify(messages));
  }, [messages]);

  // Log isLoading state changes
  useEffect(() => {
    console.log("[Thread Component] stream.isLoading changed:", isLoading);
  }, [isLoading]);

  const lastError = useRef<string | undefined>(undefined);

  const setThreadIdInUrl = useCallback(
    (id: string | null) => {
      console.log("[Thread Component] setThreadIdInUrl called with:", id);
      _setThreadIdInternal(id, { shallow: false, history: "push" });
      closeArtifact();
      setArtifactContext({});
    },
    [_setThreadIdInternal, closeArtifact, setArtifactContext],
  );

  const createNewThreadBackend = useCallback(async (): Promise<string | null> => {
    if (!apiUrl) {
      toast.error("API URL is not available in context. Cannot create thread.");
      console.error("[Thread Component createNewThreadBackend] apiUrl from context is missing.");
      return null;
    }
    console.log("[Thread Component] Attempting to create new thread via backend POST /threads, using apiUrl from context:", apiUrl);
    try {
      const client = createClient(apiUrl, getApiKey() ?? undefined);
      const newThreadObject = await client.threads.create();
      
      const isValidThreadObject = (obj: any): obj is { id: string; [key: string]: any } => {
        return obj && typeof obj.id === 'string';
      };

      if (isValidThreadObject(newThreadObject)) {
        console.log("[Thread Component] Successfully created new thread via backend, ID:", newThreadObject.id);
        return newThreadObject.id;
      } else {
        console.error("[Thread Component] Failed to create new thread or received invalid/missing ID from backend.", newThreadObject);
        toast.error("Failed to create a new chat thread on the server (ID property missing or not a string).");
        return null;
      }
    } catch (error) {
      console.error("[Thread Component] Error creating new thread via backend:", error);
      toast.error("Error creating new chat thread: " + (error instanceof Error ? error.message : String(error)));
      return null;
    }
  }, [apiUrl]);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const errorObj = stream.error as any;
      const message = errorObj.message || JSON.stringify(errorObj);
      if (!message || lastError.current === message) return;
      lastError.current = message;
      toast.error("An error occurred.", {
        description: <p><strong>Details:</strong> <code>{message}</code></p>,
        richColors: true,
        closeButton: true,
      });
    } catch { /* no-op */ }
  }, [stream.error]);

  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (messages.length !== prevMessageLength.current && messages?.length && messages[messages.length - 1].type === "ai") {
      setFirstTokenReceived(true);
    }
    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    let currentThreadId = threadId;
    let isNewThread = false;

    if (!currentThreadId) {
      console.log("[Thread Component handleSubmit] No threadId, creating one before submit.");
      toast.info("Creating new chat thread...");
      const newId = await createNewThreadBackend();
      if (newId) {
        setThreadIdInUrl(newId);
        currentThreadId = newId;
        isNewThread = true;
        console.log(`[Thread Component handleSubmit] New thread ${currentThreadId} created and URL updated.`);
      } else {
        toast.error("Failed to create thread. Cannot send message.");
        console.error("[Thread Component handleSubmit] createNewThreadBackend failed.");
        return;
      }
    }

    if (!currentThreadId) {
        toast.error("Cannot send message without a valid chat thread ID.");
        console.error("[Thread Component handleSubmit] currentThreadId is still null after creation attempt.");
        return;
    }

    setFirstTokenReceived(false);
    const newHumanMessage: Message = { id: uuidv4(), type: "human", content: input };
    const toolMessages = ensureToolCallsHaveResponses(messages);
    const context = Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    console.log(`[Thread Component handleSubmit] Submitting message. Using threadId: ${currentThreadId}. Was new thread created just now: ${isNewThread}`);
    
    const payload = {
       messages: [...toolMessages, newHumanMessage],
       context
    };
    
    console.log("[Thread Component handleSubmit] Payload for stream.submit:", payload);

    stream.submit(
      payload,
      {
        streamMode: ["values"]
      },
    );
    setInput("");
  };

  const handleRegenerate = (parentCheckpoint: Checkpoint | null | undefined) => {
    if (!threadId) {
      toast.error("No active chat thread to regenerate.");
      return;
    }
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, { checkpoint: parentCheckpoint, streamMode: ["values"] });
  };

  const handleNewThreadClick = () => {
    console.log("[Thread Component] New Thread button clicked / Navigating to new thread.");
    setThreadIdInUrl(null);
  };

  const chatStarted = !!threadId || messages.length > 0;
  const hasNoAIOrToolMessages = !messages.find((m: Message) => m.type === "ai" || m.type === "tool");

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-white"
          style={{ width: 300 }}
          animate={isLargeScreen ? { x: chatHistoryOpen ? 0 : -300 } : { x: chatHistoryOpen ? 0 : -300 }}
          initial={{ x: -300 }}
          transition={isLargeScreen ? { type: "spring", stiffness: 300, damping: 30 } : { duration: 0 }}
        >
          <div className="relative h-full" style={{ width: 300 }}>
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen ? (isLargeScreen ? "calc(100% - 300px)" : "100%") : "100%",
          }}
          transition={isLargeScreen ? { type: "spring", stiffness: 300, damping: 30 } : { duration: 0 }}
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-2 pl-4">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? <PanelRightOpen className="size-5" /> : <PanelRightClose className="size-5" />}
                  </Button>
                )}
              </div>
              <div className="absolute top-2 right-4 flex items-center">
                <OpenGitHubRepo />
              </div>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2">
              <div className="relative flex items-center justify-start gap-2">
                <div className="absolute left-0 z-10">
                  {(!chatHistoryOpen || !isLargeScreen) && (
                    <Button
                      className="hover:bg-gray-100"
                      variant="ghost"
                      onClick={() => setChatHistoryOpen((p) => !p)}
                    >
                      {chatHistoryOpen ? <PanelRightOpen className="size-5" /> : <PanelRightClose className="size-5" />}
                    </Button>
                  )}
                </div>
                <motion.button
                  className="flex cursor-pointer items-center gap-2"
                  onClick={handleNewThreadClick}
                  animate={{ marginLeft: !chatHistoryOpen ? 48 : 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                >
                  <LangGraphLogoSVG width={32} height={32} />
                  <span className="text-xl font-semibold tracking-tight">Agent Chat</span>
                </motion.button>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center"><OpenGitHubRepo /></div>
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="New thread"
                  variant="ghost"
                  onClick={handleNewThreadClick}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>
              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                <>
                  {messages
                    .filter((m: Message) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message: Message, index: number) =>
                      message.type === "human" ? (
                        <HumanMessage key={message.id || `${message.type}-${index}`} message={message} isLoading={isLoading} />
                      ) : (
                        <AssistantMessage key={message.id || `${message.type}-${index}`} message={message} isLoading={isLoading} handleRegenerate={handleRegenerate} />
                      ),
                    )}
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
                    <AssistantMessage key="interrupt-msg" message={undefined} isLoading={isLoading} handleRegenerate={handleRegenerate} />
                  )}
                  {isLoading && !firstTokenReceived && <AssistantMessageLoading />}
                </>
              }
              footer={
                <div className="sticky bottom-0 flex flex-col items-center gap-8 bg-white">
                  {!chatStarted && (
                    <div className="flex items-center gap-3">
                      <LangGraphLogoSVG className="h-8 flex-shrink-0" />
                      <h1 className="text-2xl font-semibold tracking-tight">Agent Chat</h1>
                    </div>
                  )}
                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />
                  <div className="bg-muted relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl border shadow-xs">
                    <form onSubmit={handleSubmit} className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2">
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.nativeEvent.isComposing) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder="Type your message..."
                        className="field-sizing-content resize-none border-none bg-transparent p-3.5 pb-0 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />
                      <div className="flex items-center justify-between p-2 pt-4">
                        <div>
                          <div className="flex items-center space-x-2">
                            <Switch id="render-tool-calls" checked={hideToolCalls ?? false} onCheckedChange={setHideToolCalls} />
                            <Label htmlFor="render-tool-calls" className="text-sm text-gray-600">Hide Tool Calls</Label>
                          </div>
                        </div>
                        {stream.isLoading ? (
                          <Button key="stop" onClick={() => stream.stop()}>
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            Cancel
                          </Button>
                        ) : (
                          <Button type="submit" className="shadow-md transition-all" disabled={isLoading || !input.trim()}>
                            Send
                          </Button>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button onClick={closeArtifact} className="cursor-pointer">
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
    </div>
  );
}
