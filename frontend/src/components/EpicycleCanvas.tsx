import { useEffect, useLayoutEffect, useMemo, useRef } from "react"
import {
  axisChain,
  realValue,
  reconstructSource,
  sourceChain,
  type Coefficient,
} from "../math/series.ts"
import type { Appearance } from "../types.ts"

type Point = { x: number; y: number }

type Props = {
  points: Array<[number, number]>
  closed: boolean
  offset: [number, number]
  coefficients: Coefficient[]
  xCoefficients: Coefficient[]
  yCoefficients: Coefficient[]
  mode: "complex" | "cartesian"
  playing: boolean
  speed: number
  resetToken: number
  appearance: Appearance
}

function mix(amount: number): string {
  const low = [42, 111, 219]
  const high = [232, 93, 76]
  const u = Math.max(0, Math.min(1, amount))
  const rgb = low.map((channel, index) => Math.round(channel + (high[index] - channel) * u))
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

export function EpicycleCanvas({
  points,
  closed,
  offset,
  coefficients,
  xCoefficients,
  yCoefficients,
  mode,
  playing,
  speed,
  resetToken,
  appearance,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const playingRef = useRef(playing)
  const speedRef = useRef(speed)
  const modeRef = useRef(mode)
  const appearanceRef = useRef(appearance)
  const dataRef = useRef({ points, closed, offset, coefficients, xCoefficients, yCoefficients })
  const tRef = useRef(0)
  const trailRef = useRef<Point[]>([])
  const viewRef = useRef({ zoom: 1, panX: 0, panY: 0 })
  const dragRef = useRef<{ x: number; y: number } | null>(null)

  const prepared = useMemo(() => {
    const samples = reconstructSource(coefficients, points.length, offset[0], offset[1])
    const errors = samples.map((point, index) => {
      const source = points[index]
      if (!source) return 0
      return Math.hypot(point.x - source[0], point.y - source[1])
    })
    const peak = Math.max(1e-9, ...errors)
    return { samples, errors, peak }
  }, [coefficients, points, offset])
  const preparedRef = useRef(prepared)

  useLayoutEffect(() => {
    playingRef.current = playing
    speedRef.current = speed
    modeRef.current = mode
    appearanceRef.current = appearance
    dataRef.current = { points, closed, offset, coefficients, xCoefficients, yCoefficients }
    preparedRef.current = prepared
  }, [playing, speed, mode, appearance, points, closed, offset, coefficients, xCoefficients, yCoefficients, prepared])

  useEffect(() => {
    tRef.current = 0
    trailRef.current = []
    viewRef.current = { zoom: 1, panX: 0, panY: 0 }
  }, [resetToken])

  useEffect(() => {
    trailRef.current = []
  }, [coefficients, mode])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let frame = 0
    let last = performance.now()

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const context = canvas.getContext("2d")
      if (!context) return
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const width = Math.max(1, rect.width)
      const height = Math.max(1, rect.height)
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr)
        canvas.height = Math.floor(height * dpr)
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
      const look = appearanceRef.current
      context.fillStyle = look.background
      context.fillRect(0, 0, width, height)

      const data = dataRef.current
      const source = data.points.map(([x, y]) => ({ x, y }))
      if (source.length < 2) {
        frame = requestAnimationFrame(draw)
        return
      }
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity
      for (const point of source) {
        minX = Math.min(minX, point.x)
        minY = Math.min(minY, point.y)
        maxX = Math.max(maxX, point.x)
        maxY = Math.max(maxY, point.y)
      }
      const spanX = Math.max(maxX - minX, 1)
      const spanY = Math.max(maxY - minY, 1)
      const yAnchor = minY - spanY * 0.38
      const xAnchor = minX - spanX * 0.38
      if (modeRef.current === "cartesian") {
        minX = Math.min(minX, xAnchor - spanX * 0.08)
        minY = Math.min(minY, yAnchor - spanY * 0.08)
        maxX += spanX * 0.06
        maxY += spanY * 0.06
      }
      const centerX = (minX + maxX) / 2
      const centerY = (minY + maxY) / 2
      const base = Math.min((width - 36) / (maxX - minX), (height - 36) / (maxY - minY))
      const view = viewRef.current
      const scale = base * view.zoom
      const project = (x: number, y: number) => ({
        x: width / 2 + (x - centerX) * scale + view.panX,
        y: height / 2 + (y - centerY) * scale + view.panY,
      })

      if (playingRef.current) {
        const next = tRef.current + dt * speedRef.current
        if (next >= 1) {
          tRef.current = next % 1
          trailRef.current = []
        } else {
          tRef.current = next
        }
      }
      const t = tRef.current
      const pen = currentPen(data, t, modeRef.current)
      if (playingRef.current) trailRef.current.push(pen)

      const stroke = (pts: Point[], color: string, widthPx: number, joinEnds: boolean) => {
        if (pts.length < 2) return
        context.beginPath()
        const first = project(pts[0].x, pts[0].y)
        context.moveTo(first.x, first.y)
        for (let index = 1; index < pts.length; index += 1) {
          const point = project(pts[index].x, pts[index].y)
          context.lineTo(point.x, point.y)
        }
        if (joinEnds) {
          context.lineTo(first.x, first.y)
        }
        context.strokeStyle = color
        context.lineWidth = widthPx
        context.lineJoin = "round"
        context.lineCap = "round"
        context.stroke()
      }

      if (look.showOriginal) stroke(source, look.original, look.strokeWidth, data.closed)
      const curve = preparedRef.current
      if (look.colorByError && curve.samples.length === source.length) {
        context.lineWidth = look.strokeWidth
        context.lineCap = "round"
        for (let index = 0; index < curve.samples.length - (data.closed ? 0 : 1); index += 1) {
          const a = project(curve.samples[index].x, curve.samples[index].y)
          const b = project(
            curve.samples[(index + 1) % curve.samples.length].x,
            curve.samples[(index + 1) % curve.samples.length].y,
          )
          context.beginPath()
          context.moveTo(a.x, a.y)
          context.lineTo(b.x, b.y)
          context.strokeStyle = mix(curve.errors[index] / curve.peak)
          context.stroke()
        }
      } else {
        stroke(curve.samples, hexFade(look.reconstruction, 0.35), look.strokeWidth, data.closed)
      }
      stroke(trailRef.current, look.reconstruction, look.strokeWidth + 0.6, false)

      if (modeRef.current === "complex") {
        drawChain(context, sourceChain(data.coefficients, t, data.offset[0], data.offset[1]), project, scale, look)
      } else {
        const xChain = axisChain(data.xCoefficients, t, data.offset[0], yAnchor, "x")
        const yChain = axisChain(data.yCoefficients, t, xAnchor, data.offset[1], "y")
        drawChain(context, xChain, project, scale, look)
        drawChain(context, yChain, project, scale, look)
        if (look.showGuides) {
          context.strokeStyle = hexFade(look.circles, 0.85)
          context.lineWidth = 1
          const target = project(pen.x, pen.y)
          if (xChain.length) {
            const from = project(pen.x, xChain[xChain.length - 1].y)
            context.beginPath()
            context.moveTo(from.x, from.y)
            context.lineTo(target.x, target.y)
            context.stroke()
          }
          if (yChain.length) {
            const from = project(yChain[yChain.length - 1].x, pen.y)
            context.beginPath()
            context.moveTo(from.x, from.y)
            context.lineTo(target.x, target.y)
            context.stroke()
          }
        }
      }

      const dot = project(pen.x, pen.y)
      context.beginPath()
      context.fillStyle = look.reconstruction
      context.arc(dot.x, dot.y, 4.2, 0, Math.PI * 2)
      context.fill()
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frame)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const cursorX = event.clientX - rect.left
      const cursorY = event.clientY - rect.top
      const view = viewRef.current
      const factor = event.deltaY < 0 ? 1.08 : 1 / 1.08
      const next = Math.min(14, Math.max(0.25, view.zoom * factor))
      const ratio = next / view.zoom
      view.panX = (cursorX - rect.width / 2) * (1 - ratio) + view.panX * ratio
      view.panY = (cursorY - rect.height / 2) * (1 - ratio) + view.panY * ratio
      view.zoom = next
    }
    canvas.addEventListener("wheel", onWheel, { passive: false })
    return () => canvas.removeEventListener("wheel", onWheel)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="stage"
      onPointerDown={(event) => {
        dragRef.current = { x: event.clientX, y: event.clientY }
        event.currentTarget.setPointerCapture(event.pointerId)
      }}
      onPointerMove={(event) => {
        if (!dragRef.current) return
        viewRef.current.panX += event.clientX - dragRef.current.x
        viewRef.current.panY += event.clientY - dragRef.current.y
        dragRef.current = { x: event.clientX, y: event.clientY }
      }}
      onPointerUp={() => {
        dragRef.current = null
      }}
      onDoubleClick={() => {
        viewRef.current = { zoom: 1, panX: 0, panY: 0 }
      }}
    />
  )
}

function currentPen(
  data: {
    coefficients: Coefficient[]
    xCoefficients: Coefficient[]
    yCoefficients: Coefficient[]
    offset: [number, number]
  },
  t: number,
  mode: "complex" | "cartesian",
): Point {
  if (mode === "cartesian") {
    return {
      x: data.offset[0] + realValue(data.xCoefficients, t),
      y: data.offset[1] + realValue(data.yCoefficients, t),
    }
  }
  const chain = sourceChain(data.coefficients, t, data.offset[0], data.offset[1])
  const tip = chain.at(-1)
  return tip ? { x: tip.x, y: tip.y } : { x: data.offset[0], y: data.offset[1] }
}

function drawChain(
  context: CanvasRenderingContext2D,
  circles: Array<{ cx: number; cy: number; x: number; y: number; radius: number; frequency: number }>,
  project: (x: number, y: number) => { x: number; y: number },
  scale: number,
  look: Appearance,
) {
  for (const circle of circles) {
    const center = project(circle.cx, circle.cy)
    const tip = project(circle.x, circle.y)
    const radius = circle.radius * Math.abs(scale)
    if (look.showCircles && radius >= 0.75) {
      context.beginPath()
      context.strokeStyle = hexFade(look.circles, 0.75)
      context.lineWidth = 1
      context.arc(center.x, center.y, radius, 0, Math.PI * 2)
      context.stroke()
    }
    if (look.showVectors && radius >= 0.4) {
      context.beginPath()
      context.strokeStyle = look.circles
      context.lineWidth = 1.6
      context.moveTo(center.x, center.y)
      context.lineTo(tip.x, tip.y)
      context.stroke()
    }
  }
}

function hexFade(hex: string, alpha: number): string {
  const text = hex.replace("#", "")
  if (text.length !== 6) return hex
  const r = Number.parseInt(text.slice(0, 2), 16)
  const g = Number.parseInt(text.slice(2, 4), 16)
  const b = Number.parseInt(text.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
