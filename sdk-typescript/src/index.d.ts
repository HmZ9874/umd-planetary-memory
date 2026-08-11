export interface UMDClientOptions {
  endpoint: string;
  apiKey: string;
  fetch?: typeof globalThis.fetch;
}

export interface RequestOptions { signal?: AbortSignal; idempotencyKey?: string; }
export interface WriteInput {
  text: string; scope?: string; tags?: string[]; classification?: string;
  importance?: number; entities?: string[]; fact_key?: string; pinned?: boolean;
}
export interface WriteReceipt { memory_id: string; state: string; reason: string; revision: number; }
export interface SearchInput {
  query: string; top_k?: number; scope?: string; budget_chars?: number; chronological?: boolean;
}
export interface SearchHit {
  memory_id: string; text: string; force: number; orbit_radius: number;
  potential_energy: number; components: Record<string, number>;
  explanation: Record<string, unknown>;
}
export interface CapsuleHit {
  capsule_id: string; kind: string; source_ids: string[]; text: string; force: number;
  components: Record<string, number>; explanation: Record<string, unknown>;
}
export interface MemoryDocument {
  id: string; text: string; state: string; source: string; scope?: string;
  kind: string; entities: string[]; tags: string[]; classification: string;
}

export class UMDAPIError extends Error {
  status: number; code: string; requestId?: string; details: Record<string, unknown>;
}

export class UMDClient {
  constructor(options: UMDClientOptions);
  add(input: WriteInput, options?: RequestOptions): Promise<WriteReceipt>;
  search(input: SearchInput, options?: RequestOptions): Promise<SearchHit[]>;
  searchCapsules(input: Omit<SearchInput, "chronological">, options?: RequestOptions): Promise<CapsuleHit[]>;
  get(memoryId: string, options?: RequestOptions): Promise<MemoryDocument>;
  list(filters?: { limit?: number; cursor?: string; state?: string; scope?: string }, options?: RequestOptions):
    Promise<{ memories: MemoryDocument[]; nextCursor?: string }>;
  feedback(input: { query: string; useful_memory_id: string; rejected_memory_ids?: string[] }, options?: RequestOptions):
    Promise<Record<string, number>>;
  entities(options?: RequestOptions): Promise<Record<string, unknown>[]>;
  graph(limit?: number, options?: RequestOptions): Promise<Record<string, unknown>>;
  audit(limit?: number, options?: RequestOptions): Promise<Record<string, unknown>[]>;
  system(options?: RequestOptions): Promise<Record<string, unknown>>;
}
