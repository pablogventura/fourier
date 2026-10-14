import { useEffect, useRef } from "react"
import Plotly from "plotly.js/lib/core"
import bar from "plotly.js/lib/bar"
import scatter from "plotly.js/lib/scatter"

let registered = false
function ensurePlotly() {
  if (!registered) {
    Plotly.register([bar, scatter])
    registered = true
  }
}

type Props = {
  data: unknown[]
  layout: Record<string, unknown>
}

const config = { displayModeBar: false, responsive: true }

export function Plot({ data, layout }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const node = ref.current
    return () => {
      if (node) Plotly.purge(node)
    }
  }, [])

  useEffect(() => {
    const node = ref.current
    if (!node) return
    ensurePlotly()
    void Plotly.react(node, data, layout, config)
  }, [data, layout])

  return <div className="plot" ref={ref} />
}
