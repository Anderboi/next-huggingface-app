import { HfInference } from "@huggingface/inference";
import { HuggingFaceStream, StreamingTextResponse } from "ai";
import { experimental_buildOpenAssistantPrompt } from "ai/prompts";

// Create a new HuggingFace Inference instance
const Hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// IMPORTANT! Set the runtime to edge
export const runtime = "edge";

function buildPrompt(
  messages: { content: string; role: "system" | "user" | "assistant" }[]
) {
  return (
    messages
      .map(({ content, role }) => {
        if (role === "user") {
          return `<|prompter|>${content}<|endoftext|>`;
        } else {
          return `<|assistant|>${content}<|endoftext|>`;
        }
      })
      .join("") + "<|assistant|>"
  );
}

export async function POST(req: Request) {
  // Extract the `messages` from the body of the request
  let { messages } = await req.json();

  const promptEngeneering =
    "Act as a top interior designer and architect. You are an expert in interior design and know all latest trends and all the information in this field. If someone asks you something totally unrelated to interior design or architecture, you will respond with generic text saying you are only trained to respond to questions about interior design and architecture. But if it touches interior design or architecture, then you should respond. The message is this:";

  messages = messages.map(
    (message: { content: string; role: "system" | "user" | "assistant" }) => {
      if (message.role === "user") {
        return {
          ...message,
          content: `${promptEngeneering} ${message.content}`,
        };
      } else {
        return message;
      }
    }
  );

  const response = Hf.textGenerationStream({
    model: "meta-llama/Llama-2-70b-chat-hf",
    inputs: buildPrompt(messages),
    parameters: {
      max_new_tokens: 200,
      // @ts-ignore (this is a valid parameter specifically in OpenAssistant models)
      typical_p: 0.2,
      repetition_penalty: 1,
      truncate: 1000,
      return_full_text: false,
    },
  });

  // Convert the response into a friendly text-stream
  const stream = HuggingFaceStream(response);

  // Respond with the stream
  return new StreamingTextResponse(stream);
}
