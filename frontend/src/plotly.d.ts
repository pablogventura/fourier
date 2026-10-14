declare module "plotly.js/lib/core" {
  interface PlotlyStatic {
    register(traces: unknown[]): void
    react(root: HTMLElement, data: unknown[], layout: unknown, config?: unknown): Promise<unknown>
    purge(root: HTMLElement): void
  }
  const Plotly: PlotlyStatic
  export default Plotly
}

declare module "plotly.js/lib/bar" {
  const bar: unknown
  export default bar
}

declare module "plotly.js/lib/scatter" {
  const scatter: unknown
  export default scatter
}
