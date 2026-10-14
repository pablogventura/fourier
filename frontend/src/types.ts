import type { Coefficient, Strategy } from "./math/series.ts"

export type { Coefficient, Strategy }

export type Contour = {
  id: number
  closed: boolean
  is_hole: boolean
  perimeter: number
  sample_count: number
  points: Array<[number, number]>
  bbox: [number, number, number, number]
  label: string
  note?: string | null
}

export type ContourBundle = {
  source: string
  width: number
  height: number
  preview: string | null
  warnings: string[]
  contours: Contour[]
}

export type FourierResult = {
  sample_count: number
  closed: boolean
  centered: boolean
  offset: [number, number]
  coefficients: Coefficient[]
  x_coefficients: Coefficient[]
  y_coefficients: Coefficient[]
  warnings: string[]
}

export type RatePoint = {
  component_count: number
  stored_coefficients: number
  encoded_bytes: number
  rmse: number
  max_error: number
  hausdorff: number
}

export type Metrics = {
  sample_count: number
  component_count: number
  stored_coefficients: number
  number_format: string
  quant_step: number
  raw_points_bytes: number
  raw_coefficients_bytes: number
  encoded_bytes: number
  ratio_vs_points: number | null
  ratio_vs_coefficients: number | null
  rmse: number
  max_error: number
  hausdorff: number
  curve: RatePoint[]
  note: string
}

export type JpegBlock = {
  row: number
  col: number
  original: number[][]
  quantized: number[][]
  dequantized: number[][]
}

export type JpegResult = {
  width: number
  height: number
  blocks_x: number
  blocks_y: number
  psnr: number | null
  ssim: number
  block: JpegBlock | null
  quality_table: number[][]
  grayscale: boolean
  channel: number
  original: string
  reconstructed: string
  difference: string
  note: string
}

export type Appearance = {
  background: string
  original: string
  reconstruction: string
  circles: string
  strokeWidth: number
  showCircles: boolean
  showVectors: boolean
  showGuides: boolean
  showOriginal: boolean
  colorByError: boolean
}

export type DrawMode = "image" | "blank"
