const encoder = new TextEncoder();
const decoder = new TextDecoder();

// create a stdin buffer to stick the response in
let stdinBuffer = "";

async function readLine(): Promise<string> {
  // check for already complete lines
  const newlineIndex = stdinBuffer.indexOf("\n");
  if (newlineIndex !== -1) {
    const line = stdinBuffer.slice(0, newlineIndex);
    stdinBuffer = stdinBuffer.slice(newlineIndex + 1);
    return line;
  }

  // read additional data from stdin until we've read teh whole buffer
  const buffer = new Uint8Array(1024 * 64);
  while (true) {
    const n = await Deno.stdin.read(buffer);
    if (n === null) {
      const remaining = stdinBuffer;
      stdinBuffer = "";
      if (remaining.length > 0) {
        return remaining;
      }
      throw new Error("Unexpected EOF while waiting for tool response");
    }

    stdinBuffer += decoder.decode(buffer.subarray(0, n));

    const idx = stdinBuffer.indexOf("\n");
    if (idx !== -1) {
      const line = stdinBuffer.slice(0, idx);
      stdinBuffer = stdinBuffer.slice(idx + 1);
      return line;
    }
  }
}

export async function callTool<T>(
  tool: string,
  params: Record<string, unknown>,
): Promise<T> {
  const request = JSON.stringify({ __tool_call__: true, tool, params });
  await Deno.stdout.write(encoder.encode(request + "\n"));

  const line = await readLine();

  let response: { __tool_result__?: T; __tool_error__?: string };
  try {
    response = JSON.parse(line);
  } catch {
    throw new Error(`Invalid JSON response from Python: ${line}`);
  }

  if ("__tool_error__" in response && response.__tool_error__ !== undefined) {
    throw new Error(response.__tool_error__);
  }

  if (!("__tool_result__" in response)) {
    throw new Error(`Invalid response format: ${line}`);
  }

  return response.__tool_result__ as T;
}

export function output(value: unknown): void {
  const message = JSON.stringify({ __output__: value });
  Deno.stdout.writeSync(encoder.encode(message + "\n"));
}

export function debug(...args: unknown[]): void {
  const message = JSON.stringify({
    __debug__: args
      .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
      .join(" "),
  });
  Deno.stdout.writeSync(encoder.encode(message + "\n"));
}
