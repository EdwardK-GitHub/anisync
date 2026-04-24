export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(data.detail ?? "Request failed.")
  }

  return data as T
}

export function assetUrl(path?: string | null): string {
  if (!path) return ""
  if (path.startsWith("http://") || path.startsWith("https://")) return path
  return `${API_BASE}${path}`
}
