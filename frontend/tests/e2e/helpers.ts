import { APIRequestContext } from "@playwright/test";

export const API_BASE_URL = process.env.RAG_API_BASE_URL || "http://localhost:8000";

export async function isBackendUp(request: APIRequestContext): Promise<boolean> {
  try {
    const resp = await request.get(`${API_BASE_URL}/health`, { timeout: 5000 });
    return resp.ok();
  } catch {
    return false;
  }
}
