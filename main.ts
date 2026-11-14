// free-model-aggregator-01 — one server that fronts ten simulated free-tier
// providers behind the two client wires (OpenAI Chat Completions and Anthropic
// Messages), in JSON and in server-sent events.
//
// How a request flows:
//
//   1. The client body is parsed once into a canonical shape: the system
//      texts, the conversation as ordered text/image parts, the tool
//      definitions in order, the output cap, the stop sequences, the JSON-mode
//      flag, and the temperature. Nothing the provider echoes back is dropped
//      or rewritten on the way through: the system text, the last user text,
//      the tool names, max_tokens, the stops, the image bytes and the JSON
//      flag all survive the round trip, because the client compares the
//      provider's echo against what it originally sent.
//   2. A candidate list is built from the brief's quality table merged with
//      the providers' own catalog listings: only free models, only models
//      whose capabilities meet the request's requirements (the workload
//      manifest row for this request, unioned with what the body itself
//      shows), and only wires that can carry the request (an OpenAI JSON-mode
//      request never goes to an Anthropic-wire provider, whose wire cannot
//      echo JSON mode). Candidates run best rating first.
//   3. Providers are called one at a time, always asking for a plain JSON
//      answer, and the walk stops at the first success: exactly one provider
//      call may answer a request. A refusal (402/429/5xx) or a transport
//      failure moves on to the next candidate; it never crashes the trial.
//   4. The winning provider's answer text is returned byte for byte, with its
//      usage counts exactly as reported and its stop reason mapped into the
//      client's wire. When the client asked for a stream, the JSON answer is
//      re-emitted as that wire's event sequence.
//
// Everything is a pure function of the brief, the requests, and the
// providers' answers: no wall clock anywhere (fixed ids and timestamps), no
// randomness, and every tie-break comes from counters and brief order, so a
// replay reproduces the same bytes. The catalog this server registers is the
// same document `--list` prints: the providers' own listings, with the
// brief's quality table filling any gap a failed probe leaves.

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBool(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** A stop field as a list of strings: a lone string becomes one stop. */
function asStringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function log(message: string): void {
  process.stderr.write(`[agg] ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The provider's own base path (or absolute URL), exactly as listed. */
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
  error: string | null;
}

function wireOf(value: unknown): Wire | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function authOf(value: unknown): AuthScheme {
  if (value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

/** The index of the `}` that closes the object opening at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`. */
function documentsIn(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a standalone object: keep scanning.
    }
  }
  return found;
}

function directoryProviderOf(value: unknown): DirectoryProvider | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  const wire = wireOf(value.wire);
  if (id === '' || wire === null) return null;
  const base =
    asString(value.path) !== ''
      ? asString(value.path)
      : asString(value.base_path) !== ''
        ? asString(value.base_path)
        : asString(value.base_url) !== ''
          ? asString(value.base_url)
          : asString(value.base);
  return {
    id,
    name: asString(value.name) !== '' ? asString(value.name) : id,
    wire,
    auth: authOf(value.auth),
    base,
    key_env: asString(value.key_env),
    key: asString(value.key),
  };
}

function ratingRowOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = asString(value.model);
  const provider = asString(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(value.rating_milli) ?? 0,
    context_length: asNumber(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function manifestRowOf(value: unknown): ManifestRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const row: ManifestRow = {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {},
  };
  if (requires.tools === true) row.requires.tools = true;
  if (requires.vision === true) row.requires.vision = true;
  if (requires.json === true) row.requires.json = true;
  const minContext = asNumber(requires.min_context);
  if (minContext !== null) row.requires.min_context = minContext;
  return row;
}

/** Read the brief file and pull its three documents out by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read brief: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directoryDoc === undefined || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = directoryProviderOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingRowOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const m = manifestRowOf(v);
          return m === null ? [] : [m];
        })
      : [];
  return { providers, ratings, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probes: the providers' own listings are the truth the catalog is
// scored against, so every provider's catalog route is asked in its own wire
// before anything is registered. A probe that fails falls back to the brief's
// quality rows for that provider, so one refused connection never costs a
// whole provider.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/** One routable model: a catalog entry plus its rating and capabilities. */
interface ModelEntry {
  provider: DirectoryProvider;
  providerOrder: number;
  id: string;
  free: boolean;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** Join a directory base to the env URL; an absolute base is used as-is. */
function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return prefix;
  return `${prefix}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers that carry one provider's key, in its own auth scheme. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function probedModelOf(wire: Wire, value: unknown): ProbedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? asString(value.name).replace(/^models\//, '') : asString(value.id);
  if (id === '') return null;
  const context =
    asNumber(value.context_length) ?? asNumber(value.context_window) ?? asNumber(value.inputTokenLimit);
  let free = asBool(value.free);
  if (free === null && isRecord(value.pricing)) {
    const prices = Object.values(value.pricing).flatMap((x) => {
      if (typeof x === 'number') return [x];
      if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
      return [];
    });
    if (prices.length > 0) free = prices.every((n) => Number.isFinite(n) && n === 0);
  }
  return {
    id,
    free,
    context_length: context,
    tools: asBool(value.tools),
    vision: asBool(value.vision),
    json: asBool(value.json),
  };
}

function probedModelsOf(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = probedModelOf(wire, v);
    return m === null ? [] : [m];
  });
}

const PROBE_TIMEOUT_MS = 5000;

/** Ask one provider's catalog route; any failure yields an empty listing. */
async function probeProvider(envUrl: string, provider: DirectoryProvider): Promise<ProbedModel[]> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return [];
    const body: unknown = await res.json();
    return probedModelsOf(provider.wire, body);
  } catch {
    return [];
  }
}

function ratingRowFor(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Build the catalog and the routing index together, so both always agree.
 * Models come from the probe listing first; when a probe came back empty, the
 * brief's quality rows for that provider stand in. The quality row wins over
 * the listing for freeness and capabilities where it exists. Models are
 * sorted by id within each provider so the document is identical no matter
 * which source supplied it.
 */
function buildCatalogAndIndex(
  brief: Brief,
  probes: Map<string, ProbedModel[]>,
): { catalog: Catalog; entries: ModelEntry[] } {
  const catalog: Catalog = { providers: [] };
  const entries: ModelEntry[] = [];
  brief.providers.forEach((provider, providerOrder) => {
    const probed = probes.get(provider.id) ?? [];
    const seen = new Set<string>();
    const merged: ProbedModel[] = [];
    for (const model of probed) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      merged.push(model);
    }
    if (merged.length === 0) {
      // The probe said nothing: fall back to the brief's rows for this provider.
      for (const row of brief.ratings) {
        if (row.provider !== provider.id || seen.has(row.model)) continue;
        seen.add(row.model);
        merged.push({
          id: row.model,
          free: row.free,
          context_length: row.context_length,
          tools: row.tools,
          vision: row.vision,
          json: row.json,
        });
      }
    }
    if (merged.length === 0) return;
    merged.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    const models: CatalogModel[] = [];
    for (const model of merged) {
      const row = ratingRowFor(brief.ratings, provider.id, model.id);
      const free = row !== null ? row.free : (model.free ?? false);
      const context_length = row?.context_length ?? model.context_length ?? null;
      models.push({ id: model.id, free, context_length });
      entries.push({
        provider,
        providerOrder,
        id: model.id,
        free,
        rating_milli: row?.rating_milli ?? 0,
        context_length,
        tools: row?.tools ?? model.tools ?? null,
        vision: row?.vision ?? model.vision ?? null,
        json: row?.json ?? model.json ?? null,
      });
    }
    catalog.providers.push({
      id: provider.id,
      base_url: provider.base,
      key_env: provider.key_env,
      wire: provider.wire,
      models,
    });
  });
  return { catalog, entries };
}

/** Probe every provider in brief order, sequentially: no concurrent calls. */
async function probeAll(envUrl: string, brief: Brief): Promise<Map<string, ProbedModel[]>> {
  const probes = new Map<string, ProbedModel[]>();
  if (envUrl === '') return probes;
  for (const provider of brief.providers) {
    probes.set(provider.id, await probeProvider(envUrl, provider));
  }
  return probes;
}

// ---------------------------------------------------------------------------
// Canonical requests: one parsed shape for both client wires, rendered back
// into whichever wire the chosen provider speaks. The parts of a message stay
// in their original order, and an image keeps its exact bytes (base64 data)
// or its exact URL text, because the provider hashes one of those two things
// and the client compares the hash.
// ---------------------------------------------------------------------------

type ImageRef =
  | { kind: 'base64'; media: string; data: string }
  | { kind: 'url'; url: string };

type ContentPart = { kind: 'text'; text: string } | { kind: 'image'; image: ImageRef };

interface CanonMessage {
  role: string;
  parts: ContentPart[];
}

interface ToolDef {
  name: string;
  description: string;
  parameters: unknown;
}

interface CanonRequest {
  systemTexts: string[];
  messages: CanonMessage[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  /** JSON mode as the client body itself asked for it. */
  jsonMode: boolean;
  temperature: number | null;
  stream: boolean;
  toolChoice: unknown;
}

/** Split a data URL into its media type and base64 payload, or return null. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const match = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  const media = match[1] === '' ? 'image/png' : match[1];
  return { media, data: match[2] };
}

function imageRefOfUrl(url: string): ImageRef {
  const data = dataUrlOf(url);
  if (data !== null) return { kind: 'base64', media: data.media, data: data.data };
  return { kind: 'url', url };
}

/** The text an OpenAI/Anthropic content value carries: a string as-is, or its text blocks joined. */
function contentPartsOpenAI(content: unknown): ContentPart[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      parts.push({ kind: 'image', image: imageRefOfUrl(block.image_url.url) });
    }
  }
  return parts;
}

function contentPartsAnthropic(content: unknown): ContentPart[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({
          kind: 'image',
          image: {
            kind: 'base64',
            media: typeof source.media_type === 'string' ? source.media_type : 'image/png',
            data: source.data,
          },
        });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ kind: 'image', image: { kind: 'url', url: source.url } });
      }
    }
    // tool_use / tool_result blocks carry nothing the echo reads; they are
    // left out of the canonical shape rather than mistranslated.
  }
  return parts;
}

function textsJoined(parts: ContentPart[]): string {
  return parts.flatMap((p) => (p.kind === 'text' ? [p.text] : [])).join('\n');
}

function parseOpenAIRequest(body: Json): CanonRequest {
  const systemTexts: string[] = [];
  const messages: CanonMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    const parts = contentPartsOpenAI(raw.content);
    if (role === 'system') {
      const text = textsJoined(parts);
      if (text !== '') systemTexts.push(text);
    } else {
      messages.push({ role: role !== '' ? role : 'user', parts });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      const name = asString(raw.function.name);
      if (name === '') continue;
      tools.push({
        name,
        description: asString(raw.function.description),
        parameters: raw.function.parameters,
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemTexts,
    messages,
    tools,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    stops: asStringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: asNumber(body.temperature),
    stream: body.stream === true,
    toolChoice: body.tool_choice,
  };
}

function parseAnthropicRequest(body: Json): CanonRequest {
  const systemTexts: string[] = [];
  if (typeof body.system === 'string') {
    if (body.system !== '') systemTexts.push(body.system);
  } else if (Array.isArray(body.system)) {
    const text = contentPartsAnthropic(body.system)
      .flatMap((p) => (p.kind === 'text' ? [p.text] : []))
      .join('\n');
    if (text !== '') systemTexts.push(text);
  }
  const messages: CanonMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = asString(raw.role);
    messages.push({ role: role !== '' ? role : 'user', parts: contentPartsAnthropic(raw.content) });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      const name = asString(raw.name);
      if (name === '') continue;
      tools.push({
        name,
        description: asString(raw.description),
        parameters: raw.input_schema,
      });
    }
  }
  return {
    systemTexts,
    messages,
    tools,
    maxTokens: asNumber(body.max_tokens),
    stops: asStringList(body.stop_sequences),
    // The Anthropic wire has no JSON-mode marker of its own; a JSON
    // requirement on such a request arrives via the manifest, not the body.
    jsonMode: false,
    temperature: asNumber(body.temperature),
    stream: body.stream === true,
    toolChoice: body.tool_choice,
  };
}

function schemaOrEmpty(parameters: unknown): unknown {
  return isRecord(parameters) ? parameters : { type: 'object', properties: {} };
}

// --- Upstream renderers: canonical shape -> one provider wire -------------

function openAIContent(parts: ContentPart[]): unknown {
  if (parts.length === 1 && parts[0].kind === 'text') return parts[0].text;
  return parts.map((part) => {
    if (part.kind === 'text') return { type: 'text', text: part.text };
    const image = part.image;
    const url = image.kind === 'base64' ? `data:${image.media};base64,${image.data}` : image.url;
    return { type: 'image_url', image_url: { url } };
  });
}

function renderOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const text of canon.systemTexts) messages.push({ role: 'system', content: text });
  for (const message of canon.messages) {
    messages.push({ role: message.role, content: openAIContent(message.parts) });
  }
  const out: Json = { model, messages, stream: false };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: schemaOrEmpty(tool.parameters) },
    }));
  }
  if (typeof canon.toolChoice === 'string' || isRecord(canon.toolChoice)) {
    out.tool_choice = canon.toolChoice;
  }
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  return out;
}

function anthropicContent(parts: ContentPart[]): unknown {
  if (parts.length === 1 && parts[0].kind === 'text') return parts[0].text;
  return parts.map((part) => {
    if (part.kind === 'text') return { type: 'text', text: part.text };
    const image = part.image;
    if (image.kind === 'base64') {
      return { type: 'image', source: { type: 'base64', media_type: image.media, data: image.data } };
    }
    return { type: 'image', source: { type: 'url', url: image.url } };
  });
}

/** A tool choice carried across to the Anthropic wire, best effort. */
function anthropicToolChoice(choice: unknown): Json | null {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'none') return { type: 'none' };
  if (choice === 'required') return { type: 'any' };
  if (isRecord(choice)) {
    if (choice.type === 'auto' || choice.type === 'any' || choice.type === 'none') return { type: choice.type };
    if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'tool', name: choice.name };
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      return { type: 'tool', name: choice.function.name };
    }
  }
  return null;
}

function renderAnthropic(canon: CanonRequest, model: string): Json {
  const out: Json = { model, stream: false };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.systemTexts.length > 0) out.system = canon.systemTexts.join('\n');
  out.messages = canon.messages.map((message) => ({
    role: message.role,
    content: anthropicContent(message.parts),
  }));
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: schemaOrEmpty(tool.parameters),
    }));
    const choice = anthropicToolChoice(canon.toolChoice);
    if (choice !== null) out.tool_choice = choice;
  }
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  return out;
}

function renderGemini(canon: CanonRequest): Json {
  const out: Json = {};
  if (canon.systemTexts.length > 0) {
    out.systemInstruction = { parts: [{ text: canon.systemTexts.join('\n') }] };
  }
  out.contents = canon.messages.map((message) => {
    const parts: Json[] = message.parts.map((part) => {
      if (part.kind === 'text') return { text: part.text };
      const image = part.image;
      if (image.kind === 'base64') {
        return { inlineData: { mimeType: image.media, data: image.data } };
      }
      return { fileData: { fileUri: image.url } };
    });
    return { role: message.role === 'assistant' ? 'model' : 'user', parts: parts.length > 0 ? parts : [{ text: '' }] };
  });
  if (canon.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: canon.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: schemaOrEmpty(tool.parameters),
        })),
      },
    ];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// --- Requirements and candidate ordering -----------------------------------

/** What a request needs: its manifest row, plus whatever its body shows. */
function requirementsFor(row: ManifestRow | null, canon: CanonRequest): Requirement {
  const requires: Requirement = { ...(row?.requires ?? {}) };
  if (canon.tools.length > 0) requires.tools = true;
  if (canon.messages.some((m) => m.parts.some((p) => p.kind === 'image'))) requires.vision = true;
  if (canon.jsonMode) requires.json = true;
  return requires;
}

function meetsRequirements(entry: ModelEntry, requires: Requirement): boolean {
  if (requires.tools === true && entry.tools !== true) return false;
  if (requires.vision === true && entry.vision !== true) return false;
  if (requires.json === true && entry.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (entry.context_length === null || entry.context_length < requires.min_context)
  ) {
    return false;
  }
  return true;
}

/**
 * Candidates in the order to try them: best rating first, then the provider
 * called least so far, then the provider that has failed least, then brief
 * order, then model id. Every tie-break is a counter or a fixed order, so
 * the order is a deterministic function of what has happened so far.
 */
function orderCandidates(
  entries: ModelEntry[],
  requires: Requirement,
  jsonMode: boolean,
  calls: Map<string, number>,
  failures: Map<string, number>,
): ModelEntry[] {
  return entries
    .filter((entry) => {
      if (!entry.free) return false;
      if (!meetsRequirements(entry, requires)) return false;
      // An Anthropic-wire provider echoes json_mode false, always: it can
      // never carry an OpenAI JSON-mode request.
      if (jsonMode && entry.provider.wire === 'anthropic') return false;
      return true;
    })
    .sort((a, b) => {
      if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
      const callsA = calls.get(a.provider.id) ?? 0;
      const callsB = calls.get(b.provider.id) ?? 0;
      if (callsA !== callsB) return callsA - callsB;
      const failA = failures.get(a.provider.id) ?? 0;
      const failB = failures.get(b.provider.id) ?? 0;
      if (failA !== failB) return failA - failB;
      if (a.providerOrder !== b.providerOrder) return a.providerOrder - b.providerOrder;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
}

// ---------------------------------------------------------------------------
// Provider calls. Every call asks for a plain JSON answer, whatever the
// client asked for; streaming is re-created locally on the way back. A 2xx
// answer is final even if some field is missing — the trial records it as a
// success, and a second success for the same request would break the
// one-provider rule — so parsing here is lenient and never throws the answer
// away. Only refusals and transport failures move on to the next candidate.
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
  /** Stop family: 0 stop, 1 length, 2 tool use, 3 refusal. */
  family: number;
}

type Attempt =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; detail: string; message: string };

/**
 * Whether a refusal means the provider is spent for the rest of the trial.
 * The providers say so in plain words: a spent budget or an outage refuses
 * every later call, while a lone refused call is transient and the next call
 * may go through. Only the exact permanent wordings mark a provider dead;
 * anything else — including a missing message — stays transient.
 */
function isPermanentRefusal(message: string): boolean {
  return message.endsWith("'s free-tier budget is spent") || message.endsWith(' is out');
}

/** The message of a provider error body, in whichever wire's envelope. */
function errorMessageOf(body: unknown): string {
  if (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string') {
    return body.error.message;
  }
  return '';
}

const PROVIDER_TIMEOUT_MS = 8000;

function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function countOf(value: unknown): number {
  const n = asNumber(value);
  return n === null ? 0 : Math.max(0, Math.floor(n));
}

/** The assistant text of an OpenAI answer: a string, or its text parts joined. */
function openAIAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return '';
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return '';
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
  }
  return '';
}

function anthropicAnswerText(body: Json): string {
  if (!Array.isArray(body.content)) return '';
  return body.content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function geminiAnswerText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null || !isRecord(candidate.content)) return '';
  const parts = Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

function parseProviderAnswer(wire: Wire, body: unknown): ProviderAnswer {
  const value = isRecord(body) ? body : {};
  if (wire === 'openai') {
    const usage = isRecord(value.usage) ? value.usage : {};
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    return {
      text: openAIAnswerText(value),
      input: countOf(usage.prompt_tokens),
      output: countOf(usage.completion_tokens),
      family: stopFamilyOf(choice.finish_reason),
    };
  }
  if (wire === 'anthropic') {
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: anthropicAnswerText(value),
      input: countOf(usage.input_tokens),
      output: countOf(usage.output_tokens),
      family: stopFamilyOf(value.stop_reason),
    };
  }
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  return {
    text: geminiAnswerText(value),
    input: countOf(usage.promptTokenCount),
    output: countOf(usage.candidatesTokenCount),
    family: stopFamilyOf(candidate.finishReason),
  };
}

async function callProvider(envUrl: string, entry: ModelEntry, canon: CanonRequest): Promise<Attempt> {
  const provider = entry.provider;
  const base = joinBase(envUrl, provider.base);
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = renderOpenAI(canon, entry.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = renderAnthropic(canon, entry.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(entry.id)}:generateContent`;
    payload = renderGemini(canon);
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, detail: `transport: ${(err as Error).message}`, message: '' };
  }
  if (!res.ok) {
    let message = '';
    try {
      message = errorMessageOf(JSON.parse(await res.text()));
    } catch {
      message = '';
    }
    return { ok: false, status: res.status, detail: `HTTP ${res.status}`, message };
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  return { ok: true, answer: parseProviderAnswer(provider.wire, body) };
}

// ---------------------------------------------------------------------------
// Answers back to the client, in the wire it used, with the provider's text
// whole, its usage counts exact, and its stop reason mapped by family.
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const OPENAI_ANSWER_ID = 'chatcmpl-agg';
const ANTHROPIC_ANSWER_ID = 'msg_agg';
const STREAM_PIECE = 32;

function openAIStopOf(family: number): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStopOf(family: number): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces;
}

function openAIJsonAnswer(model: string, answer: ProviderAnswer): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStopOf(answer.family),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.input,
      completion_tokens: answer.output,
      total_tokens: answer.input + answer.output,
    },
  };
}

function sseData(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function sseNamed(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * The OpenAI event order: one role chunk, the content pieces, exactly one
 * finish chunk, one usage chunk (with an empty choices array, as every chunk
 * must carry one), then [DONE] last.
 */
function openAIStreamFrames(model: string, answer: ProviderAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseData(chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitPieces(answer.text)) {
    frames.push(sseData(chunk({ content: piece }, null)));
  }
  frames.push(sseData(chunk({}, openAIStopOf(answer.family))));
  frames.push(
    sseData({
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.input,
        completion_tokens: answer.output,
        total_tokens: answer.input + answer.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicJsonAnswer(model: string, answer: ProviderAnswer): Json {
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.family),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/**
 * The Anthropic event order: message_start, one content block opened, its
 * text deltas, the block closed, message_delta with the stop reason and the
 * output count, then message_stop last.
 */
function anthropicStreamFrames(model: string, answer: ProviderAnswer): string {
  const frames: string[] = [
    sseNamed('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseNamed('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitPieces(answer.text)) {
    frames.push(
      sseNamed('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseNamed('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseNamed('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.family), stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseNamed('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

function openAIErrorBody(message: string): Json {
  return { error: { message, type: 'api_error', code: null } };
}

function anthropicErrorBody(message: string): Json {
  return { type: 'error', error: { type: 'api_error', message } };
}

// --- GET /v1/models: the deduplicated free-model set of the catalog -------

interface ServedModel {
  id: string;
  provider: string;
}

function servedModels(catalog: Catalog): ServedModel[] {
  const seen = new Set<string>();
  const models: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      models.push({ id: model.id, provider: provider.id });
    }
  }
  return models;
}

function openAIModelList(models: ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
      free: true,
    })),
  };
}

function anthropicModelList(models: ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.id,
      created_at: FIXED_CREATED_AT,
      free: true,
    })),
    has_more: false,
  };
}

// ---------------------------------------------------------------------------
// HTTP server: the two inference routes, the model list, and the trial
// protocol (register once, poll /work, exit 0 when the workload is done).
// ---------------------------------------------------------------------------

const MAX_BODY_BYTES = 2 * 1024 * 1024;
const WORK_POLL_MS = 100;
const REGISTER_TIMEOUT_MS = 5000;
const REGISTER_ATTEMPTS = 60;
const REGISTER_RETRY_MS = 200;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readBody(req: IncomingMessage): Promise<Buffer | null> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(tooLarge ? null : Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = Buffer.from(`${JSON.stringify(value)}\n`, 'utf8');
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': String(body.length) });
  res.end(body);
}

function sendSse(res: ServerResponse, text: string): void {
  const body = Buffer.from(text, 'utf8');
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': String(body.length),
  });
  res.end(body);
}

interface Served {
  status: number;
  json?: Json;
  sse?: string;
}

class Aggregator {
  private readonly envUrl: string;
  private readonly brief: Brief;
  private readonly catalog: Catalog;
  private readonly entries: ModelEntry[];
  private readonly calls = new Map<string, number>();
  private readonly failures = new Map<string, number>();
  /** Providers whose refusals said they are spent for the trial. */
  private readonly dead = new Set<string>();
  private arrivals = 0;

  constructor(envUrl: string, brief: Brief, catalog: Catalog, entries: ModelEntry[]) {
    this.envUrl = envUrl;
    this.brief = brief;
    this.catalog = catalog;
    this.entries = entries;
  }

  models(): ServedModel[] {
    return servedModels(this.catalog);
  }

  /** Serve one inference request: walk the candidates until one answers. */
  async serve(api: 'openai' | 'anthropic', body: Json): Promise<Served> {
    const index = this.arrivals;
    this.arrivals += 1;
    const canon = api === 'openai' ? parseOpenAIRequest(body) : parseAnthropicRequest(body);
    const row = this.brief.manifest[index] ?? null;
    const requires = requirementsFor(row, canon);
    const candidates = orderCandidates(this.entries, requires, canon.jsonMode, this.calls, this.failures);
    const label = row?.id ?? `request-${index + 1}`;
    let attempts = 0;
    for (const entry of candidates) {
      if (this.dead.has(entry.provider.id)) continue;
      attempts += 1;
      const attempt = await callProvider(this.envUrl, entry, canon);
      this.calls.set(entry.provider.id, (this.calls.get(entry.provider.id) ?? 0) + 1);
      if (!attempt.ok) {
        this.failures.set(entry.provider.id, (this.failures.get(entry.provider.id) ?? 0) + 1);
        if (isPermanentRefusal(attempt.message)) {
          this.dead.add(entry.provider.id);
          log(`${label}: ${entry.provider.id} is spent (${attempt.message}); skipping it from now on`);
        } else {
          log(`${label}: ${entry.provider.id}/${entry.id} refused (${attempt.detail}), failing over`);
        }
        continue;
      }
      log(`${label}: served by ${entry.provider.id}/${entry.id} after ${attempts} attempt(s)`);
      const answer = attempt.answer;
      if (api === 'openai') {
        if (canon.stream) return { status: 200, sse: openAIStreamFrames(entry.id, answer) };
        return { status: 200, json: openAIJsonAnswer(entry.id, answer) };
      }
      if (canon.stream) return { status: 200, sse: anthropicStreamFrames(entry.id, answer) };
      return { status: 200, json: anthropicJsonAnswer(entry.id, answer) };
    }
    log(`${label}: no provider answered after ${attempts} attempt(s)`);
    const message = 'every provider refused the request';
    return {
      status: 502,
      json: api === 'openai' ? openAIErrorBody(message) : anthropicErrorBody(message),
    };
  }
}

async function handleRequest(agg: Aggregator, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    // The wire follows the auth header: an x-api-key names Anthropic's list.
    const anthropic = typeof req.headers['x-api-key'] === 'string';
    const models = agg.models();
    sendJson(res, 200, anthropic ? anthropicModelList(models) : openAIModelList(models));
    return;
  }
  const api =
    req.method === 'POST' && path === '/v1/chat/completions'
      ? 'openai'
      : req.method === 'POST' && path === '/v1/messages'
        ? 'anthropic'
        : null;
  if (api === null) {
    sendJson(res, 404, { error: { message: `no route ${req.method ?? ''} ${path}`, type: 'invalid_request_error', code: null } });
    return;
  }
  try {
    const raw = await readBody(req);
    if (raw === null) {
      const message = 'the request body is too large';
      sendJson(res, 413, api === 'openai' ? openAIErrorBody(message) : anthropicErrorBody(message));
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw.toString('utf8'));
    } catch {
      const message = 'the request body is not JSON';
      sendJson(res, 400, api === 'openai' ? openAIErrorBody(message) : anthropicErrorBody(message));
      return;
    }
    if (!isRecord(parsed)) {
      const message = 'the request body is not one JSON object';
      sendJson(res, 400, api === 'openai' ? openAIErrorBody(message) : anthropicErrorBody(message));
      return;
    }
    const served = await agg.serve(api, parsed);
    if (served.sse !== undefined) sendSse(res, served.sse);
    else sendJson(res, served.status, served.json ?? {});
  } catch (err) {
    // One bad request must never take the trial down with it.
    log(`request failed: ${(err as Error).message}`);
    if (!res.headersSent) {
      const message = 'the request could not be served';
      sendJson(res, 502, api === 'openai' ? openAIErrorBody(message) : anthropicErrorBody(message));
    } else {
      res.end();
    }
  }
}

async function register(envUrl: string, port: number, catalog: Catalog): Promise<boolean> {
  const url = `${envUrl.replace(/\/+$/, '')}/register`;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
      });
      if (res.ok) {
        await res.arrayBuffer().catch(() => undefined);
        return true;
      }
      await res.arrayBuffer().catch(() => undefined);
      log(`register attempt ${attempt} answered HTTP ${res.status}`);
    } catch (err) {
      log(`register attempt ${attempt} failed: ${(err as Error).message}`);
    }
    await sleep(REGISTER_RETRY_MS);
  }
  return false;
}

async function pollWorkUntilDone(envUrl: string): Promise<void> {
  const url = `${envUrl.replace(/\/+$/, '')}/work`;
  for (;;) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS) });
      if (res.ok) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) return;
      } else {
        await res.arrayBuffer().catch(() => undefined);
      }
    } catch {
      // The env may not be answering yet; the next poll asks again.
    }
    await sleep(WORK_POLL_MS);
  }
}

async function buildCatalog(envUrl: string, brief: Brief): Promise<{ catalog: Catalog; entries: ModelEntry[] }> {
  const probes = await probeAll(envUrl, brief);
  return buildCatalogAndIndex(brief, probes);
}

async function listMode(): Promise<void> {
  const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  const { catalog } = await buildCatalog(envUrl, brief);
  const text = `${JSON.stringify(catalog)}\n`;
  await new Promise<void>((resolve) => {
    process.stdout.write(text, () => resolve());
  });
}

async function serveMode(): Promise<void> {
  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
  if (brief.error !== null) log(`brief: ${brief.error}`);
  const { catalog, entries } = await buildCatalog(envUrl, brief);
  log(
    `catalog: ${catalog.providers.length} providers, ${entries.length} models, ` +
      `${entries.filter((e) => e.free).length} free; manifest: ${brief.manifest.length} requests`,
  );
  const agg = new Aggregator(envUrl, brief, catalog, entries);
  const server = createServer((req, res) => {
    void handleRequest(agg, req, res).catch((err: Error) => {
      log(`handler error: ${err.message}`);
      if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal error', type: 'api_error', code: null } });
      else res.end();
    });
  });
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      server.removeListener('error', reject);
      resolve();
    });
  });
  const address = server.address();
  const port = address !== null && typeof address === 'object' ? address.port : 0;
  log(`listening on 0.0.0.0:${port}`);
  if (envUrl !== '') {
    const ok = await register(envUrl, port, catalog);
    log(ok ? 'registered with the trial env' : 'registration failed; still serving');
    await pollWorkUntilDone(envUrl);
    log('workload done; exiting 0');
  }
  server.close();
  process.exit(0);
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    await listMode();
  } else {
    await serveMode();
  }
}

main().catch((err: Error) => {
  process.stderr.write(`[agg] fatal: ${err.message}\n`);
  process.exit(1);
});
