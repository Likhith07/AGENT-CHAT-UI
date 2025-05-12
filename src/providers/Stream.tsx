import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import {
  uiMessageReducer,
  isUIMessage,
  isRemoveUIMessage,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState } from "nuqs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { LangGraphLogoSVG } from "@/components/icons/langgraph";
import { Label } from "@/components/ui/label";
import { ArrowRight } from "lucide-react";
import { PasswordInput } from "@/components/ui/password-input";
import { getApiKey } from "@/lib/api-key";
import { useThreads } from "./Thread";
import { toast } from "sonner";

export type StateType = { messages: Message[]; ui?: UIMessage[] };

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
      context?: Record<string, unknown>;
    };
    CustomEventType: UIMessage | RemoveUIMessage;
  }
>;

// Extend StreamContextType to include apiUrl
type BaseStreamContextType = ReturnType<typeof useTypedStream>;
interface StreamContextType extends BaseStreamContextType {
  apiUrl?: string; // Add apiUrl here
}

const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 1000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
): Promise<boolean> {
  try {
    const res = await fetch(`${apiUrl}/info`, {
      ...(apiKey && {
        headers: {
          "X-Api-Key": apiKey,
        },
      }),
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl, // apiUrl is now a prop passed from StreamProvider
  assistantId,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string; // Ensure it's typed as string (guaranteed by StreamProvider logic)
  assistantId: string;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, setThreads } = useThreads();

  console.log("[StreamSession] Initializing/Re-rendering. Passed apiUrl:", apiUrl, "Initial threadId from URL (useQueryState):", threadId);

  const baseStreamValue = useTypedStream({
    apiUrl, // Use the passed apiUrl directly
    apiKey: apiKey ?? undefined,
    assistantId,
    threadId: threadId ?? null,
    onCustomEvent: (event, options) => {
      if (isUIMessage(event) || isRemoveUIMessage(event)) {
        options.mutate((prev) => {
          const ui = uiMessageReducer(prev.ui ?? [], event);
          return { ...prev, ui };
        });
      }
    },
    onThreadId: (newThreadIdFromSDK) => {
      console.log("[StreamSession onThreadId] Received newThreadIdFromSDK:", newThreadIdFromSDK);
      const currentUrlThreadId = threadId; // Get current threadId from useQueryState

      if (newThreadIdFromSDK && typeof newThreadIdFromSDK === 'string') {
        const cleanId = newThreadIdFromSDK.includes(' in ') ? newThreadIdFromSDK.split(' in ')[0] : newThreadIdFromSDK;
        if (cleanId !== currentUrlThreadId) {
          console.log("[StreamSession onThreadId] SDK provided a new, valid threadId. Updating URL and refetching threads. New:", cleanId, "Old:", currentUrlThreadId);
          setThreadId(cleanId, { shallow: false });
          sleep().then(() => {
            getThreads().then(setThreads).catch(console.error);
          });
        } else {
          console.log("[StreamSession onThreadId] SDK provided threadId which matches current URL threadId. No action.", cleanId);
        }
      } else if (newThreadIdFromSDK === null) {
        console.log("[StreamSession onThreadId] SDK explicitly provided null. Clearing URL threadId and refetching threads.");
        setThreadId(null, { shallow: false });
        sleep().then(() => {
          getThreads().then(setThreads).catch(console.error);
        });
      } else { // newThreadIdFromSDK is undefined or some other unexpected value
        console.warn(`[StreamSession onThreadId] SDK provided unexpected or undefined newThreadId ('${newThreadIdFromSDK}'). Current URL threadId ('${currentUrlThreadId}') will be maintained. No URL change or thread refetch based on this SDK event.`);
        // Explicitly DO NOTHING to the URL threadId if the SDK is unclear,
        // as our URL threadId is the source of truth for initializing useStream.
      }
    },
    onError: (error) => {
      console.error("[StreamSession useTypedStream onError] Stream error:", error);
      let errorMessage = "Unknown stream error";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      toast.error("Stream error occurred.", {
        description: errorMessage,
        richColors: true,
        closeButton: true,
      });
    }
  });

  // Combine baseStreamValue with apiUrl for the context
  const streamContextValue: StreamContextType = {
    ...baseStreamValue,
    apiUrl, // Add apiUrl to the context value
  };

  useEffect(() => {
    console.log("[StreamSession useEffect for threadId prop] threadId passed to useTypedStream is now:", threadId ?? null);
    // If baseStreamValue itself has a readable threadId property from the SDK, log it too.
    // This depends on the SDK's internals, it might not be directly exposed on baseStreamValue.
    // For example: console.log("[StreamSession useEffect for threadId prop] SDK's internal threadId (if available):", baseStreamValue.threadId);
  }, [threadId]);

  useEffect(() => {
    // This effect now mainly confirms apiUrl prop changes, if any.
    // The initial checkGraphStatus for the resolved apiUrl is done by StreamProvider before rendering StreamSession
    console.log("[StreamSession useEffect] apiUrl prop:", apiUrl, "apiKey prop:", apiKey);
    if (apiUrl) { // Only check status if apiUrl is actually set
        checkGraphStatus(apiUrl, apiKey).then((ok) => {
            if (!ok) {
                toast.error("Failed to connect to LangGraph server (from StreamSession)", {
                    description: () => (
                        <p>
                        Please ensure your graph is running at <code>{apiUrl}</code>.
                        </p>
                    ),
                    duration: 10000,
                    richColors: true,
                    closeButton: true,
                });
            } else {
                console.log("[StreamSession useEffect] Graph status check OK for apiUrl:", apiUrl);
            }
        });
    }
  }, [apiKey, apiUrl]);
  
  useEffect(() => {
    console.log("[StreamSession useEffect] threadId from useQueryState changed to:", threadId);
  }, [threadId]);

  return (
    <StreamContext.Provider value={streamContextValue}> {/* Use the combined value */}
      {children}
    </StreamContext.Provider>
  );
};

const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "agent";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined = process.env.NEXT_PUBLIC_ASSISTANT_ID;

  const [apiUrlParam, setApiUrlParam] = useQueryState("apiUrl", { // Renamed to avoid conflict with prop name
    defaultValue: envApiUrl || "", 
  });
  const [assistantIdParam, setAssistantIdParam] = useQueryState("assistantId", { // Renamed
    defaultValue: envAssistantId || "",
  });

  const [apiKey, _setApiKey] = useState(() => {
    const storedKey = getApiKey();
    return storedKey || "";
  });

  const setApiKey = (key: string) => {
    window.localStorage.setItem("lg:chat:apiKey", key);
    _setApiKey(key);
  };

  // These are the definitive values after considering URL params and env vars
  const finalApiUrl = apiUrlParam || envApiUrl;
  const finalAssistantId = assistantIdParam || envAssistantId;
  
  // Effect to check graph status once finalApiUrl is determined and before rendering StreamSession or form
  // This moves the initial check here to avoid StreamSession trying to check with a potentially empty apiUrl prop initially
  useEffect(() => {
    if (finalApiUrl) {
        console.log("[StreamProvider useEffect] Initial check for graph status with finalApiUrl:", finalApiUrl);
        checkGraphStatus(finalApiUrl, apiKey).then((ok) => {
            if (!ok) {
                toast.error("Failed to connect to LangGraph server (Initial Check)", {
                    description: () => (
                        <p>
                        Please ensure your graph is running at <code>{finalApiUrl}</code> and API key (if any) is correct.
                        </p>
                    ),
                    duration: 10000,
                    richColors: true,
                    closeButton: true,
                });
            }
        });
    }
  }, [finalApiUrl, apiKey]);

  if (!finalApiUrl || !finalAssistantId) {
    console.log("[StreamProvider] Rendering setup form. finalApiUrl:", finalApiUrl, "finalAssistantId:", finalAssistantId);
    return (
      <div className="flex min-h-screen w-full items-center justify-center p-4">
        <div className="animate-in fade-in-0 zoom-in-95 bg-background flex max-w-3xl flex-col rounded-lg border shadow-lg">
          <div className="mt-14 flex flex-col gap-2 border-b p-6">
            <div className="flex flex-col items-start gap-2">
              <LangGraphLogoSVG className="h-7" />
              <h1 className="text-xl font-semibold tracking-tight">Agent Chat</h1>
            </div>
            <p className="text-muted-foreground">
              Welcome to Agent Chat! Before you get started, you need to enter the URL of your deployment and the assistant / graph ID.
            </p>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const form = e.target as HTMLFormElement;
              const formData = new FormData(form);
              const apiUrlFromForm = formData.get("apiUrl") as string;
              const assistantIdFromForm = formData.get("assistantId") as string;
              const apiKeyFromForm = formData.get("apiKey") as string;

              // These will update the URL query parameters
              setApiUrlParam(apiUrlFromForm);
              setAssistantIdParam(assistantIdFromForm);
              setApiKey(apiKeyFromForm); // This updates localStorage and local state
            }}
            className="bg-muted/50 flex flex-col gap-6 p-6"
          >
            <div className="flex flex-col gap-2">
              <Label htmlFor="apiUrl">Deployment URL<span className="text-rose-500">*</span></Label>
              <p className="text-muted-foreground text-sm">URL of your LangGraph deployment.</p>
              <Input id="apiUrl" name="apiUrl" className="bg-background" defaultValue={finalApiUrl || DEFAULT_API_URL} required />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="assistantId">Assistant / Graph ID<span className="text-rose-500">*</span></Label>
              <p className="text-muted-foreground text-sm">ID of the graph or assistant.</p>
              <Input id="assistantId" name="assistantId" className="bg-background" defaultValue={finalAssistantId || DEFAULT_ASSISTANT_ID} required />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="apiKey">LangSmith API Key (Optional)</Label>
              <p className="text-muted-foreground text-sm">Only required for some deployments. Stored in local storage.</p>
              <PasswordInput id="apiKey" name="apiKey" defaultValue={apiKey ?? ""} className="bg-background" placeholder="lsv2_pt_..." />
            </div>
            <div className="mt-2 flex justify-end">
              <Button type="submit" size="lg">Continue <ArrowRight className="size-5" /></Button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  console.log("[StreamProvider] Rendering StreamSession with apiUrl:", finalApiUrl, "assistantId:", finalAssistantId);
  return (
    <StreamSession
      apiKey={apiKey}
      apiUrl={finalApiUrl} // Pass the resolved finalApiUrl
      assistantId={finalAssistantId} // Pass the resolved finalAssistantId
    >
      {children}
    </StreamSession>
  );
};

export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
