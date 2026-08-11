export class UMDAPIError extends Error {
  constructor(status, code, message, requestId, details = {}) {
    super(`${code}: ${message}`);
    this.name = "UMDAPIError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

export class UMDClient {
  constructor({ endpoint, apiKey, fetch: fetchImpl = globalThis.fetch }) {
    if (!fetchImpl) throw new Error("A Fetch-compatible implementation is required");
    this.endpoint = endpoint.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.fetch = fetchImpl;
  }

  async request(method, path, body, { idempotencyKey, signal } = {}) {
    const headers = {
      Authorization: `Bearer ${this.apiKey}`,
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-Request-ID": `sdk_${crypto.randomUUID().replaceAll("-", "")}`,
    };
    if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
    const response = await this.fetch(`${this.endpoint}${path}`, {
      method, headers, signal, body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
      const error = payload.error ?? {};
      throw new UMDAPIError(response.status, error.code ?? "http_error",
        error.message ?? response.statusText, error.request_id, error.details);
    }
    return payload;
  }

  async add(input, options = {}) {
    return (await this.request("POST", "/v1/memories", input,
      { idempotencyKey: options.idempotencyKey, signal: options.signal })).data;
  }

  async search(input, options = {}) {
    return (await this.request("POST", "/v1/search", input, { signal: options.signal })).data;
  }

  async searchCapsules(input, options = {}) {
    return (await this.request("POST", "/v1/search", { ...input, capsules: true },
      { signal: options.signal })).data;
  }

  async get(memoryId, options = {}) {
    return (await this.request("GET", `/v1/memories/${encodeURIComponent(memoryId)}`,
      undefined, { signal: options.signal })).data;
  }

  async list({ limit = 50, cursor, state, scope } = {}, options = {}) {
    const query = new URLSearchParams({ limit: String(limit) });
    if (cursor) query.set("cursor", cursor);
    if (state) query.set("state", state);
    if (scope) query.set("scope", scope);
    const result = await this.request("GET", `/v1/memories?${query}`, undefined,
      { signal: options.signal });
    return { memories: result.data, nextCursor: result.meta.next_cursor };
  }

  async feedback(input, options = {}) {
    return (await this.request("POST", "/v1/retrieval-feedback", input,
      { signal: options.signal })).data;
  }

  async entities(options = {}) {
    return (await this.request("GET", "/v1/entities", undefined, options)).data;
  }

  async graph(limit = 500, options = {}) {
    return (await this.request("GET", `/v1/graph?limit=${limit}`, undefined, options)).data;
  }

  async audit(limit = 100, options = {}) {
    return (await this.request("GET", `/v1/audit?limit=${limit}`, undefined, options)).data;
  }

  async system(options = {}) {
    return (await this.request("GET", "/v1/system", undefined, options)).data;
  }
}
