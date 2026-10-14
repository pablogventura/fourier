import { useEffect, useRef, useState } from "react"
import type { Contour, DrawMode } from "../types.ts"

type Props = {
  preview: string | null
  contours: Contour[]
  selectedId: number
  onSelect: (id: number) => void
  drawing: boolean
  drawMode: DrawMode
  onStroke: (points: Array<[number, number]>) => void
}

type Frame = { scale: number; ox: number; oy: number; width: number; height: number }

const BLANK = { width: 480, height: 320 }

export function ContourStage({ preview, contours, selectedId, onSelect, drawing, drawMode, onStroke }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<Frame>({ scale: 1, ox: 0, oy: 0, width: BLANK.width, height: BLANK.height })
  const imageRef = useRef<HTMLImageElement | null>(null)
  const strokeRef = useRef<Array<[number, number]>>([])
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (!preview || drawMode === "blank") {
      imageRef.current = null
      return
    }
    const image = new Image()
    image.onload = () => {
      imageRef.current = image
      setTick((value) => value + 1)
    }
    image.src = preview
  }, [preview, drawMode])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext("2d")
    if (!context) return
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const width = Math.max(1, rect.width)
    const height = Math.max(1, rect.height)
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.fillStyle = "#10141b"
    context.fillRect(0, 0, width, height)

    const image = drawMode === "blank" ? null : imageRef.current
    const world = image
      ? { minX: 0, minY: 0, width: image.naturalWidth, height: image.naturalHeight }
      : boundsOf(contours, drawMode === "blank" ? BLANK : null)
    const scale = Math.min((width - 16) / world.width, (height - 16) / world.height)
    const dw = world.width * scale
    const dh = world.height * scale
    const ox = (width - dw) / 2 - world.minX * scale
    const oy = (height - dh) / 2 - world.minY * scale
    frameRef.current = { scale, ox, oy, width: world.width, height: world.height }

    if (image) context.drawImage(image, ox, oy, image.naturalWidth * scale, image.naturalHeight * scale)
    else {
      context.strokeStyle = "#243041"
      context.strokeRect(ox, oy, world.width * scale, world.height * scale)
    }

    for (const contour of contours) {
      const active = contour.id === selectedId
      context.beginPath()
      contour.points.forEach(([x, y], index) => {
        const sx = ox + x * scale
        const sy = oy + y * scale
        if (index === 0) context.moveTo(sx, sy)
        else context.lineTo(sx, sy)
      })
      if (contour.closed) context.closePath()
      context.strokeStyle = active ? "#8ef0d0" : "rgba(169, 182, 198, 0.55)"
      context.lineWidth = active ? 2 : 1
      context.setLineDash(contour.is_hole ? [4, 4] : [])
      context.stroke()
      context.setLineDash([])
    }

    const stroke = strokeRef.current
    if (stroke.length > 1) {
      context.beginPath()
      stroke.forEach(([x, y], index) => {
        const sx = ox + x * scale
        const sy = oy + y * scale
        if (index === 0) context.moveTo(sx, sy)
        else context.lineTo(sx, sy)
      })
      context.strokeStyle = "#e4b15a"
      context.lineWidth = 2
      context.stroke()
    }
  }, [contours, selectedId, drawMode, tick, drawing])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const observer = new ResizeObserver(() => setTick((value) => value + 1))
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  function toWorld(event: React.PointerEvent<HTMLCanvasElement>): [number, number] | null {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const frame = frameRef.current
    const x = (event.clientX - rect.left - frame.ox) / frame.scale
    const y = (event.clientY - rect.top - frame.oy) / frame.scale
    return [x, y]
  }

  return (
    <canvas
      ref={canvasRef}
      className="stage side-stage"
      onPointerDown={(event) => {
        const point = toWorld(event)
        if (!point) return
        if (!drawing) {
          const nearest = nearestContour(contours, point, 16 / frameRef.current.scale)
          if (nearest !== null) onSelect(nearest)
          return
        }
        strokeRef.current = [point]
        event.currentTarget.setPointerCapture(event.pointerId)
        setTick((value) => value + 1)
      }}
      onPointerMove={(event) => {
        if (!drawing || strokeRef.current.length === 0 || event.buttons === 0) return
        const point = toWorld(event)
        if (!point) return
        const last = strokeRef.current[strokeRef.current.length - 1]
        if (Math.hypot(point[0] - last[0], point[1] - last[1]) < 1.2) return
        strokeRef.current.push(point)
        setTick((value) => value + 1)
      }}
      onPointerUp={() => {
        const stroke = strokeRef.current
        strokeRef.current = []
        setTick((value) => value + 1)
        if (drawing && stroke.length >= 8) onStroke(stroke)
      }}
    />
  )
}

function boundsOf(contours: Contour[], blank: { width: number; height: number } | null) {
  if (blank && contours.length === 0) return { minX: 0, minY: 0, width: blank.width, height: blank.height }
  let minX = blank ? 0 : Infinity
  let minY = blank ? 0 : Infinity
  let maxX = blank ? blank.width : -Infinity
  let maxY = blank ? blank.height : -Infinity
  for (const contour of contours) {
    for (const [x, y] of contour.points) {
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }
  }
  if (!Number.isFinite(minX)) return { minX: 0, minY: 0, width: BLANK.width, height: BLANK.height }
  const pad = Math.max(8, (maxX - minX) * 0.04)
  return { minX: minX - pad, minY: minY - pad, width: maxX - minX + pad * 2, height: maxY - minY + pad * 2 }
}

function nearestContour(contours: Contour[], point: [number, number], tolerance: number): number | null {
  let best: number | null = null
  let bestDistance = tolerance
  for (const contour of contours) {
    for (const [x, y] of contour.points) {
      const distance = Math.hypot(x - point[0], y - point[1])
      if (distance < bestDistance) {
        bestDistance = distance
        best = contour.id
      }
    }
  }
  return best
}
