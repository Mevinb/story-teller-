export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function api<T = Record<string, unknown>>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, { cache: "no-store", ...options });
  const text = await response.text();
  let data: { error?: string; success?: boolean; status?: string };
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new ApiError(
      `The server returned an unreadable response (${response.status}).`,
      response.status,
    );
  }
  if (
    !response.ok ||
    data.success === false ||
    (data.error && (!data.status || data.status === "error"))
  ) {
    throw new ApiError(
      data.error || `Request failed (${response.status}).`,
      response.status,
    );
  }
  return data as T;
}
export const send = <T = Record<string, unknown>>(
  path: string,
  body?: unknown,
  method = "POST",
) =>
  api<T>(path, {
    method,
    ...(body === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
export const projectUrl = (project: string, suffix = "") =>
  `/api/project/${encodeURIComponent(project)}${suffix}`;
export const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export const words = (text: string) =>
  text.trim().split(/\s+/).filter(Boolean).length;
export const splitList = (text: string) =>
  text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
export function downloadText(text: string, filename: string) {
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/plain;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
