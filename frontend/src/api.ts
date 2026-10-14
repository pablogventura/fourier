import type { ContourBundle, FourierResult, JpegResult, Metrics } from "./types.ts"

async function readError(response: Response): Promise<string> {
  const text = await response.text()
  try {
    const body = JSON.parse(text) as { detail?: string }
    if (typeof body.detail === "string") return body.detail
  } catch {
    /* el cuerpo no era JSON */
  }
  return text || `Error ${response.status}`
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as T
}

export async function postForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(path, { method: "POST", body: form })
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as T
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as T
}

export function loadPreset(name: string, sampleCount: number): Promise<ContourBundle> {
  return postJson("/api/contours/preset", { name, sample_count: sampleCount })
}

export function loadStroke(
  points: Array<[number, number]>,
  sampleCount: number,
  closed: boolean | null,
): Promise<ContourBundle> {
  return postJson("/api/contours/stroke", { points, sample_count: sampleCount, closed })
}

export function loadFile(file: File, params: {
  threshold: number
  blurSigma: number
  minLength: number
  method: string
  polarity: string
  sampleCount: number
}): Promise<ContourBundle> {
  const form = new FormData()
  form.set("file", file)
  form.set("threshold", String(params.threshold))
  form.set("blur_sigma", String(params.blurSigma))
  form.set("min_length", String(params.minLength))
  form.set("method", params.method)
  form.set("polarity", params.polarity)
  form.set("sample_count", String(params.sampleCount))
  return postForm("/api/contours/file", form)
}

export function analyze(points: Array<[number, number]>, closed: boolean, center: boolean): Promise<FourierResult> {
  return postJson("/api/fourier", { points, closed, center })
}

export function loadMetrics(body: Record<string, unknown>): Promise<Metrics> {
  return postJson("/api/compression/metrics", body)
}

export async function download(path: string, body: unknown, filename: string): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await readError(response))
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function importFlc1(file: File): Promise<{
  points: Array<[number, number]>
  closed: boolean
  centered: boolean
  note: string
  sample_count: number
}> {
  const form = new FormData()
  form.set("file", file)
  return postForm("/api/compression/import", form)
}

export function loadJpeg(file: File): Promise<{ image_id: string; width: number; height: number; preview: string; note: string }> {
  const form = new FormData()
  form.set("file", file)
  return postForm("/api/jpeg/load", form)
}

export function reconstructJpeg(body: Record<string, unknown>): Promise<JpegResult> {
  return postJson("/api/jpeg/reconstruct", body)
}

export function loadBasis(): Promise<{ patterns: number[][][] }> {
  return getJson("/api/jpeg/basis")
}
