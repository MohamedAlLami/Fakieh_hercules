import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import { WaterSystemLayout } from '../../components/water-system/WaterSystemLayout'
import { ChartComponent } from '../../components/water-system/ChartComponent'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  Sparkles,
  Bot,
  Send,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Target,
  TrendingUp,
  TrendingDown,
  Activity,
  Radio,
  RefreshCw,
  Play,
  Pause,
  FlaskConical,
  X,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Wrench,
} from 'lucide-react'

const AI = '/api/ai'

// ---- Types -----------------------------------------------------------------
interface ProviderStatus {
  provider: string
  model: string
  configured: boolean
}
interface Health {
  any_provider_configured: boolean
  ml_model_ready: boolean
  providers: ProviderStatus[]
}
interface Overall {
  count: number
  accuracy_score: number
  on_target: number
  over: number
  under: number
  on_target_pct: number
  over_pct: number
  under_pct: number
  flagged: number
}
interface MaterialAccuracy {
  material_name: string
  material_code: string
  count: number
  mean_dev_pct: number
  mean_abs_dev_pct: number
  over: number
  under: number
  on_target: number
}
interface DoseRow {
  material_name: string
  product_name: string
  batch_name: string
  setpoint: number
  actual: number | null
  deviation_pct: number | null
  status: string
  date: string | null
}
interface Insights {
  summary: string
  provider: string | null
  cached: boolean
  insights: {
    meta: { row_count: number; batch_count: number; product_count: number }
    overall: Overall
    worst_materials: MaterialAccuracy[]
  }
}
interface ChartSpec {
  type: 'bar' | 'line'
  title?: string
  labels: string[]
  values: number[]
  unit?: string
}
interface ChatMsg {
  role: 'user' | 'ai'
  text: string
  chart?: ChartSpec | null
  provider?: string | null
  cached?: boolean
}
// The single Dose-Quality model: its metadata (dropdown options + the derived
// metrics for risk / severity / direction) and one prediction that carries all
// three answers + the 5-bucket probability spectrum.
interface DoseInfo {
  classes: string[]
  class_labels: Record<string, string>
  n_total: number
  metrics: {
    risk: { accuracy: number; roc_auc: number }
    severity: { accuracy: number; macro_f1: number }
    direction: { accuracy: number; roc_auc: number }
  }
  options: {
    materials: { material_code: string; material_name: string }[]
    products: string[]
  }
}
interface DosePrediction {
  risk_pct: number
  band: string
  flagged: boolean
  prediction: string
  severity: string
  severity_label: string
  severity_confidence_pct: number
  direction: 'over' | 'under'
  direction_label: string
  direction_confidence_pct: number
  top_class: string
  top_class_label: string
  class_probabilities: Record<string, number>
  inputs: { material_name: string; product_name?: string; setpoint: number }
}

// ---- Live-monitor types ----------------------------------------------------
interface LiveFeedItem {
  id: string
  seq: number
  batch_name: string
  order_id?: string
  material_name: string
  material_code: string
  product_name: string
  setpoint: number
  quantity?: number | null
  risk_pct: number
  band: 'High' | 'Medium' | 'Low'
  flagged: boolean
  prediction: string
  actual_status: string
  actual_pct: number | null
  correct: boolean
  severity?: string | null
  severity_label?: string | null
  direction?: string | null
  direction_label?: string | null
  recommendation: string
  model_version: number
  at: string
}
interface LiveDrift {
  status: 'warming' | 'healthy' | 'watch' | 'drift'
  novel_rate: number | null
  setpoint_psi: number | null
  new_recipes: string[]
  checked_at: string | null
}
interface LiveModel {
  version: number
  trained_at: string
  trained_on: number
  accuracy: number | null
}
interface LiveStats {
  processed: number
  flagged: number
  on_target: number
  over: number
  under: number
  correct: number
  rolling_accuracy: number | null
}
interface LiveNotif {
  id: string
  kind: 'drift' | 'retrain' | 'maintenance'
  title: string
  message: string
  at: string
}
interface LiveState {
  source: string
  running: boolean
  speed: number
  retraining: boolean
  cursor: number
  total: number
  model: LiveModel
  stats: LiveStats
  drift: LiveDrift
  feed: LiveFeedItem[]
  notifications: LiveNotif[]
}

// ---- Helpers ---------------------------------------------------------------
// Charts use the app's cyan accent so they read as native Hercules charts.
const ACCENT = '#22d3ee'
// Native input/select styling copied from the existing pages so form controls match.
const FIELD =
  'bg-slate-800 light:bg-white border-slate-700 light:border-gray-300 text-slate-300 light:text-gray-700'

function specToChartData(spec: ChartSpec) {
  return {
    labels: spec.labels,
    datasets: [
      {
        label: spec.unit || 'Value',
        data: spec.values,
        backgroundColor: 'rgba(34, 211, 238, 0.45)',
        borderColor: ACCENT,
        borderWidth: 2,
        tension: 0.3,
      },
    ],
  }
}

function bandColor(band: string) {
  if (band === 'High') return 'text-red-400 bg-red-500/15 border-red-500/40'
  if (band === 'Medium') return 'text-amber-400 bg-amber-500/15 border-amber-500/40'
  return 'text-emerald-400 bg-emerald-500/15 border-emerald-500/40'
}

// The single model's 5 buckets, low → high, with a spectrum colour each.
const DOSE_ORDER = ['under_severe', 'under_watch', 'on_target', 'over_watch', 'over_severe'] as const
const DOSE_META: Record<string, { label: string; bar: string; text: string }> = {
  under_severe: { label: 'Severe under', bar: 'bg-red-500', text: 'text-red-400' },
  under_watch: { label: 'Slight under', bar: 'bg-amber-500', text: 'text-amber-400' },
  on_target: { label: 'On target', bar: 'bg-emerald-500', text: 'text-emerald-400' },
  over_watch: { label: 'Slight over', bar: 'bg-amber-500', text: 'text-amber-400' },
  over_severe: { label: 'Severe over', bar: 'bg-red-500', text: 'text-red-400' },
}

const SUGGESTED = [
  'Which ingredients are most over-dosed, and by how much?',
  'What are the 3 worst batches for dosing accuracy?',
  'Compare dosing accuracy across products.',
  'Why are micro-ingredients less accurate?',
]

// ---- Native building blocks (match the existing Hercules pages) -------------

/** Card panel — identical styling to the Fakieh Dashboard's cards. */
function Panel({
  title,
  icon,
  right,
  className = '',
  children,
}: {
  title?: string
  icon?: React.ReactNode
  right?: React.ReactNode
  className?: string
  children: React.ReactNode
}) {
  return (
    <div
      className={
        'bg-slate-800/50 light:bg-white border border-slate-700/50 light:border-gray-200 ' +
        'rounded-lg p-6 shadow-lg light:shadow-xl ' +
        className
      }
    >
      {title && (
        <div className="mb-4 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            {icon && <span className="text-cyan-400 light:text-cyan-600">{icon}</span>}
            <h3 className="text-lg font-semibold text-white light:text-gray-900">{title}</h3>
          </div>
          {right}
        </div>
      )}
      {children}
    </div>
  )
}

const STAT_COLORS: Record<string, { text: string; dot: string; chip: string; glow: string }> = {
  green: { text: 'text-green-400', dot: 'bg-green-400', chip: 'from-green-500 to-green-600', glow: 'from-green-500/20 via-transparent to-green-500/20' },
  blue: { text: 'text-blue-400', dot: 'bg-blue-400', chip: 'from-blue-500 to-blue-600', glow: 'from-blue-500/20 via-transparent to-blue-500/20' },
  cyan: { text: 'text-cyan-400', dot: 'bg-cyan-400', chip: 'from-cyan-500 to-cyan-600', glow: 'from-cyan-500/20 via-transparent to-cyan-500/20' },
  orange: { text: 'text-orange-400', dot: 'bg-orange-400', chip: 'from-orange-500 to-orange-600', glow: 'from-orange-500/20 via-transparent to-orange-500/20' },
  red: { text: 'text-red-400', dot: 'bg-red-400', chip: 'from-red-500 to-red-600', glow: 'from-red-500/20 via-transparent to-red-500/20' },
}

/** KPI stat card — identical to the Fakieh Dashboard's KPI cards. When
 *  `onClick` is given, the whole card becomes a button (cursor, ring-on-hover,
 *  a "Tap to view" hint) so it's obvious there's a list behind the number. */
function StatCard({
  label,
  value,
  subtitle,
  icon,
  color,
  pulse = true,
  onClick,
}: {
  label: string
  value: string
  subtitle?: string
  icon: React.ReactNode
  color: keyof typeof STAT_COLORS
  pulse?: boolean
  onClick?: () => void
}) {
  const c = STAT_COLORS[color]
  return (
    <div
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => (e.key === 'Enter' || e.key === ' ') && onClick() : undefined}
      className={`bg-slate-800/50 light:bg-white border border-slate-700/50 light:border-gray-200 rounded-lg p-6 shadow-lg light:shadow-xl relative overflow-hidden group hover:shadow-2xl transition-all duration-300 ${
        onClick ? 'cursor-pointer hover:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/50' : ''
      }`}
    >
      <div className={`absolute inset-0 bg-gradient-to-r ${c.glow} opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div>
      <div className="relative z-10">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-400 light:text-gray-600">{label}</p>
            <p className="text-2xl font-bold text-white light:text-gray-900">{value}</p>
            {subtitle && (
              <p className={`text-xs ${c.text} flex items-center`}>
                <span className={`w-2 h-2 ${c.dot} rounded-full mr-2 ${pulse ? 'animate-pulse' : ''}`}></span>
                {subtitle}
              </p>
            )}
          </div>
          <div className={`w-12 h-12 bg-gradient-to-br ${c.chip} rounded-lg flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300`}>
            {icon}
          </div>
        </div>
        {onClick && (
          <p className="mt-3 flex items-center gap-1 text-[11px] font-medium text-slate-500 light:text-gray-400 opacity-0 transition-opacity group-hover:opacity-100">
            <ChevronRight className="h-3 w-3" /> Tap to view the list
          </p>
        )}
      </div>
    </div>
  )
}

// ---- Live-monitor building blocks ------------------------------------------
// Plain-language status labels — no ML/statistics terms. Operators see a
// color and a short phrase; the numbers behind it live in a hidden details panel.
const DRIFT_META: Record<string, { label: string; text: string; dot: string; ring: string }> = {
  warming: { label: 'Starting…', text: 'text-slate-300 light:text-gray-600', dot: 'bg-slate-400', ring: 'border-slate-600/50 light:border-gray-300' },
  healthy: { label: 'All Good', text: 'text-emerald-400', dot: 'bg-emerald-400', ring: 'border-emerald-500/40' },
  watch: { label: 'Something New', text: 'text-amber-400', dot: 'bg-amber-400', ring: 'border-amber-500/40' },
  drift: { label: 'Updating…', text: 'text-red-400', dot: 'bg-red-400', ring: 'border-red-500/40' },
}

function outcomeTag(f: LiveFeedItem): { t: string; c: string } {
  if (f.actual_status === 'on_target') return { t: 'On target', c: 'text-emerald-400' }
  if (f.actual_status === 'over') return { t: `${f.actual_pct}% over`, c: 'text-red-400' }
  if (f.actual_status === 'under') return { t: `${Math.abs(f.actual_pct ?? 0)}% under`, c: 'text-red-400' }
  return { t: '—', c: 'text-slate-500' }
}

/** One scored batch in the live operator feed. Flagged doses show the full
 *  operator guidance inline (before the batch runs). */
function FeedRow({ f }: { f: LiveFeedItem }) {
  const dot = f.band === 'High' ? 'bg-red-500' : f.band === 'Medium' ? 'bg-amber-500' : 'bg-emerald-500'
  const wrap = f.flagged
    ? f.band === 'High'
      ? 'border-red-500/40 bg-red-500/5 light:bg-red-50'
      : 'border-amber-500/40 bg-amber-500/5 light:bg-amber-50'
    : 'border-slate-700/50 light:border-gray-200 bg-slate-900/40 light:bg-gray-50'
  const out = outcomeTag(f)
  return (
    <div className={`animate-in fade-in slide-in-from-top-2 duration-300 rounded-lg border px-3 py-2 ${wrap}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-2">
          <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${dot}`}></span>
          <div className="min-w-0">
            <div className="truncate text-sm text-slate-200 light:text-gray-800">
              {f.material_name} <span className="text-slate-500 light:text-gray-400">· {f.setpoint} kg</span>
            </div>
            <div className="truncate text-xs text-slate-500 light:text-gray-500">
              {f.product_name} · batch {f.batch_name}
            </div>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-[9px] uppercase tracking-wide text-slate-500 light:text-gray-400">AI predicted</div>
          <span className={`rounded-full border px-2 py-0.5 text-xs ${bandColor(f.band)}`}>{f.risk_pct}% risk</span>
          {f.direction && (
            <div className="mt-1 flex items-center justify-end gap-1 text-[10px] font-medium text-slate-400 light:text-gray-500">
              {f.direction === 'over' ? (
                <TrendingUp className="h-3 w-3 text-amber-400" />
              ) : (
                <TrendingDown className="h-3 w-3 text-cyan-400" />
              )}
              {f.direction === 'over' ? 'Likely high' : 'Likely low'}
            </div>
          )}
          <div className="mt-1 text-[9px] uppercase tracking-wide text-slate-500 light:text-gray-400">Actual result</div>
          <div className={`text-[11px] font-medium ${out.c}`}>{out.t}</div>
        </div>
      </div>
      {f.flagged && f.recommendation && (
        <div className="mt-2 flex items-start gap-2 border-t border-slate-700/40 light:border-gray-200 pt-2 text-xs text-slate-300 light:text-gray-600">
          <AlertTriangle className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${f.band === 'High' ? 'text-red-400' : 'text-amber-400'}`} />
          <span>{f.recommendation}</span>
        </div>
      )}
    </div>
  )
}

/** Ranked list of the ingredients most often off-target — directly actionable
 *  (which feeder to check), unlike a per-recipe accuracy chart. */
function MaterialWatchlist({ materials }: { materials: MaterialAccuracy[] }) {
  return (
    <div className="space-y-2.5">
      {materials.map((m) => {
        const severe = m.mean_abs_dev_pct >= 50
        const barColor = severe ? 'bg-red-500' : 'bg-amber-500'
        const textColor = severe ? 'text-red-400' : 'text-amber-400'
        const direction = m.mean_dev_pct > 0 ? 'usually over target' : m.mean_dev_pct < 0 ? 'usually under target' : 'mixed'
        return (
          <div
            key={m.material_name}
            className="rounded-lg border border-slate-700/50 light:border-gray-200 bg-slate-900/40 light:bg-gray-50 p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-sm text-slate-200 light:text-gray-800">{m.material_name}</span>
              <span className={`shrink-0 text-sm font-bold ${textColor}`}>
                {Math.round(m.mean_abs_dev_pct)}% off
              </span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800 light:bg-gray-200">
              <div
                className={`h-full ${barColor}`}
                style={{ width: `${Math.min(100, m.mean_abs_dev_pct)}%` }}
              ></div>
            </div>
            <div className="mt-1 text-[11px] text-slate-500 light:text-gray-500">{direction}</div>
          </div>
        )
      })}
    </div>
  )
}

/** Strip of small colored blocks — one per recent batch, colored by the real
 *  outcome (not the AI's prediction). Plain divs, not a chart: nothing to
 *  rescale or rotate, so it never "jumps" as new batches arrive live. */
function OutcomeStrip({ items }: { items: LiveFeedItem[] }) {
  function color(f: LiveFeedItem) {
    if (f.actual_status === 'on_target') return 'bg-emerald-500'
    if (f.actual_status === 'over') return 'bg-amber-500'
    if (f.actual_status === 'under') return 'bg-red-500'
    return 'bg-slate-600'
  }
  function label(f: LiveFeedItem) {
    if (f.actual_status === 'on_target') return 'On target'
    if (f.actual_status === 'over') return `${f.actual_pct}% over`
    if (f.actual_status === 'under') return `${Math.abs(f.actual_pct ?? 0)}% under`
    return 'Unknown'
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((f) => (
        <div
          key={f.id}
          title={`${f.material_name} · ${label(f)}`}
          className={`h-7 w-3.5 rounded-sm ${color(f)} transition-opacity`}
        ></div>
      ))}
    </div>
  )
}

/** Renders a written AI answer operator-simple: a big leading number (when the
 *  answer has one) or a short bold headline, with the full explanation
 *  collapsed behind a "Show details" toggle instead of a wall of text. */
function AnswerCard({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const sentences = text.split(/(?<=[.!?])\s+/).filter(Boolean)
  const headline = (sentences[0] || text).trim()
  const rest = sentences.slice(1).join(' ').trim()

  // Pull a number out to headline big ONLY when it sits at the start or end
  // of the sentence (that's what makes it "the" stat, not just a stray
  // figure). Critically: the caption always stays the FULL, UNMODIFIED
  // sentence — never a stripped remainder. Removing the number from the text
  // (an earlier version of this component did that) breaks grammar whenever
  // the number is the object of a trailing word, e.g. "...deviation of 28%"
  // becomes "...deviation of" with nothing after it. Showing the number twice
  // (once big, once in context) is a trivial redundancy; a dangling sentence
  // fragment in front of an audience is not.
  const leadMatch = headline.match(/^-?\d{1,3}(?:\.\d+)?\s?%\b/)
  const trailMatch = headline.match(/-?\d{1,3}(?:\.\d+)?\s?%[.\s]*$/)
  const statText = leadMatch
    ? leadMatch[0].trim()
    : trailMatch
      ? trailMatch[0].replace(/[.\s]+$/, '').trim()
      : null

  return (
    <div>
      {statText && (
        <div className="mb-1 text-3xl font-bold text-cyan-300">{statText}</div>
      )}
      <p className="text-base font-semibold leading-snug text-white light:text-gray-900">{headline}</p>
      {rest && (
        <>
          <button
            onClick={() => setOpen((o) => !o)}
            className="mt-2 flex items-center gap-1 text-xs text-cyan-400 light:text-cyan-600 hover:underline"
          >
            {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            {open ? 'Hide details' : 'Show details'}
          </button>
          {open && (
            <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-300 light:text-gray-600">
              {rest}
            </p>
          )}
        </>
      )}
    </div>
  )
}

function doseStatusTag(status: string): { t: string; c: string; dot: string } {
  if (status === 'on_target') return { t: 'On target', c: 'text-emerald-400', dot: 'bg-emerald-500' }
  if (status === 'over') return { t: 'Over', c: 'text-amber-400', dot: 'bg-amber-500' }
  if (status === 'under') return { t: 'Under', c: 'text-red-400', dot: 'bg-red-500' }
  return { t: status, c: 'text-slate-400', dot: 'bg-slate-500' }
}

function DoseRowView({ d }: { d: DoseRow }) {
  const tag = doseStatusTag(d.status)
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-700/50 light:border-gray-200 bg-slate-900/40 light:bg-gray-50 px-3 py-2">
      <div className="flex min-w-0 items-center gap-2">
        <span className={`h-2 w-2 shrink-0 rounded-full ${tag.dot}`}></span>
        <div className="min-w-0">
          <div className="truncate text-sm text-slate-200 light:text-gray-800">
            {d.material_name} <span className="text-slate-500 light:text-gray-400">· {d.setpoint} kg</span>
          </div>
          <div className="truncate text-xs text-slate-500 light:text-gray-500">
            {d.product_name} · batch {d.batch_name}
          </div>
        </div>
      </div>
      <div className="shrink-0 text-right">
        <span className={`text-xs font-medium ${tag.c}`}>{tag.t}</span>
        {d.deviation_pct != null && (
          <div className="text-[11px] text-slate-500 light:text-gray-400">
            {d.deviation_pct > 0 ? '+' : ''}
            {d.deviation_pct}%
          </div>
        )}
      </div>
    </div>
  )
}

/** Tab 1 popup — fetches the individual historical doses behind a KPI card
 *  (e.g. all 196 "Over-dosed" rows) so a number is never a dead end. */
function DoseListDialog({ status, title, onClose }: { status: string; title: string; onClose: () => void }) {
  const [rows, setRows] = useState<DoseRow[] | null>(null)
  const [total, setTotal] = useState(0)

  useEffect(() => {
    setRows(null)
    axios
      .get(`${AI}/doses`, { params: { status, limit: 150 } })
      .then((r) => {
        setRows(r.data.doses)
        setTotal(r.data.total)
      })
      .catch(() => setRows([]))
  }, [status])

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl border-slate-700/50 light:border-gray-200 bg-slate-900 light:bg-white">
        <DialogHeader>
          <DialogTitle className="text-white light:text-gray-900">{title}</DialogTitle>
        </DialogHeader>
        {rows === null ? (
          <div className="flex items-center justify-center gap-2 py-10 text-slate-400 light:text-gray-500">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading…
          </div>
        ) : rows.length === 0 ? (
          <p className="py-10 text-center text-sm text-slate-500 light:text-gray-400">No doses in this category.</p>
        ) : (
          <>
            {total > rows.length && (
              <p className="text-xs text-slate-500 light:text-gray-400">
                Showing the most recent {rows.length} of {total}.
              </p>
            )}
            <div className="max-h-[60vh] space-y-2 overflow-y-auto pr-1">
              {rows.map((d, i) => (
                <DoseRowView key={i} d={d} />
              ))}
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

/** Tab 2 popup — the live batches behind a KPI card, reusing the same
 *  FeedRow already used in the main feed so the two views read identically. */
function FeedListDialog({
  items,
  title,
  onClose,
}: {
  items: LiveFeedItem[]
  title: string
  onClose: () => void
}) {
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl border-slate-700/50 light:border-gray-200 bg-slate-900 light:bg-white">
        <DialogHeader>
          <DialogTitle className="text-white light:text-gray-900">{title}</DialogTitle>
        </DialogHeader>
        {items.length === 0 ? (
          <p className="py-10 text-center text-sm text-slate-500 light:text-gray-400">Nothing in this category yet.</p>
        ) : (
          <div className="max-h-[60vh] space-y-2 overflow-y-auto pr-1">
            {items.map((f) => (
              <FeedRow key={f.id} f={f} />
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

// ---- Component -------------------------------------------------------------
export function AiAssistant() {
  const [health, setHealth] = useState<Health | null>(null)
  const [insights, setInsights] = useState<Insights | null>(null)
  const [insightsLoading, setInsightsLoading] = useState(true)

  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  const [doseInfo, setDoseInfo] = useState<DoseInfo | null>(null)
  const [material, setMaterial] = useState('')
  const [product, setProduct] = useState('')
  const [setpoint, setSetpoint] = useState('0.9')
  const [dosePrediction, setDosePrediction] = useState<DosePrediction | null>(null)
  const [predicting, setPredicting] = useState(false)

  // KPI-card drilldown popups
  const [tab1DoseModal, setTab1DoseModal] = useState<{ status: string; title: string } | null>(null)
  const [tab2FeedModal, setTab2FeedModal] = useState<{ items: LiveFeedItem[]; title: string } | null>(null)
  const [whatIfOpen, setWhatIfOpen] = useState(false)

  // Live monitor
  const [live, setLive] = useState<LiveState | null>(null)
  const [feedFilter, setFeedFilter] = useState<'all' | 'flagged' | 'ok'>('all')
  const [toasts, setToasts] = useState<LiveNotif[]>([])
  const seenNotif = useRef<Set<string>>(new Set())
  const firstLoad = useRef(true)

  // Initial loads
  useEffect(() => {
    axios.get<Health>(`${AI}/health`).then((r) => setHealth(r.data)).catch(() => {})

    setInsightsLoading(true)
    axios
      .get<Insights>(`${AI}/insights`)
      .then((r) => setInsights(r.data))
      .catch(() => {})
      .finally(() => setInsightsLoading(false))

    axios
      .get<DoseInfo>(`${AI}/ml/dose/info`)
      .then((r) => {
        setDoseInfo(r.data)
        const cu = r.data.options.materials.find((m) =>
          m.material_name.toLowerCase().includes('copper proteinate')
        )
        setMaterial(cu ? cu.material_code : r.data.options.materials[0]?.material_code || '')
        setProduct(r.data.options.products[0] || '')
      })
      .catch(() => {})
  }, [])

  // Poll the live monitor (drives the feed, KPIs, charts, and drift/retrain popups).
  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const { data } = await axios.get<LiveState>(`${AI}/live/state`)
        if (!alive) return
        setLive(data)
        if (firstLoad.current) {
          // Don't replay pre-existing notifications as popups on first open.
          data.notifications.forEach((n) => seenNotif.current.add(n.id))
          firstLoad.current = false
        } else {
          const fresh = data.notifications.filter((n) => !seenNotif.current.has(n.id))
          if (fresh.length) {
            fresh.forEach((n) => seenNotif.current.add(n.id))
            setToasts((t) => [...fresh, ...t].slice(0, 4))
            // Maintenance alerts stay until dismissed by hand — too important to
            // let disappear on a timer. Drift/retrain updates auto-clear.
            fresh
              .filter((n) => n.kind !== 'maintenance')
              .forEach((n) =>
                setTimeout(() => setToasts((t) => t.filter((x) => x.id !== n.id)), 9000)
              )
          }
        }
      } catch {
        /* backend momentarily unreachable — keep last state */
      }
    }
    poll()
    const id = setInterval(poll, 1200)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  async function handleAsk(q?: string) {
    const text = (q ?? question).trim()
    if (!text || asking) return
    // Recent turns so the backend can resolve follow-ups ("make it as a
    // graph") that refer back to what was just asked.
    const history = messages.slice(-6).map((m) => ({ role: m.role, text: m.text }))
    setMessages((m) => [...m, { role: 'user', text }])
    setQuestion('')
    setAsking(true)
    try {
      const { data } = await axios.post(`${AI}/ask`, { question: text, history })
      setMessages((m) => [
        ...m,
        {
          role: 'ai',
          text: data.answer,
          chart: data.chart,
          provider: data.provider,
          cached: data.cached,
        },
      ])
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'ai', text: 'Sorry — the AI service is unreachable. Is the backend running?' },
      ])
    } finally {
      setAsking(false)
    }
  }

  async function handlePredict() {
    if (!material || predicting) return
    setPredicting(true)
    // ONE call to the single dose-quality model → risk + severity + direction.
    try {
      const { data } = await axios.post<DosePrediction>(`${AI}/ml/dose/predict`, {
        material_code: material,
        product_name: product,
        setpoint: parseFloat(setpoint) || 0,
      })
      setDosePrediction(data)
    } catch {
      setDosePrediction(null)
    }
    setPredicting(false)
  }

  async function control(action: string, value?: number) {
    try {
      const { data } = await axios.post<LiveState>(`${AI}/live/control`, { action, value })
      setLive(data)
      if (action === 'reset') {
        setToasts([])
      }
    } catch {
      /* ignore */
    }
  }

  // ---- Derived chart data (Tab 1) ------------------------------------------
  const overall = insights?.insights.overall
  const distData = overall
    ? {
        labels: ['On target', 'Over-dosed', 'Under-dosed'],
        datasets: [
          {
            data: [overall.on_target, overall.over, overall.under],
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderColor: '#0f172a',
            borderWidth: 2,
          },
        ],
      }
    : null

  const worstMaterials = insights?.insights.worst_materials || []

  const liveProvider = health?.providers.find((p) => p.configured)

  // ---- Derived data (Tab 2 — live monitor) ---------------------------------
  const st = live?.stats
  const dr = live?.drift
  const driftMeta = DRIFT_META[dr?.status || 'warming']
  const onTargetRate = st && st.processed ? Math.round((st.on_target / st.processed) * 100) : 0

  // Recent batches, oldest -> newest, for the outcome strip.
  const recentOutcomes = live ? [...live.feed].slice(0, 30).reverse() : []

  const feedAll = live?.feed ?? []
  const feedFlaggedCount = feedAll.filter((f) => f.flagged).length
  const feedOkCount = feedAll.length - feedFlaggedCount
  const filteredFeed = feedAll.filter((f) =>
    feedFilter === 'all' ? true : feedFilter === 'flagged' ? f.flagged : !f.flagged
  )
  const streamPct = live && live.total ? Math.round((live.cursor / live.total) * 100) : 0
  const flaggedPct = st && st.processed ? Math.round((st.flagged / st.processed) * 100) : 0

  const outcomeMix = st
    ? {
        labels: ['On target', 'Over-dosed', 'Under-dosed'],
        datasets: [
          {
            data: [st.on_target, st.over, st.under],
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderColor: '#0f172a',
            borderWidth: 2,
          },
        ],
      }
    : null

  const speedPresets: { label: string; value: number }[] = [
    { label: 'Slow', value: 2.0 },
    { label: 'Normal', value: 1.1 },
    { label: 'Fast', value: 0.4 },
  ]

  return (
    <WaterSystemLayout
      title="Hercules AI"
      subtitle="Generative insights and an automatic live dosing-quality monitor over the Fakieh plant data"
    >
      {/* Drift / retrain popups (native card styling, auto-dismiss) */}
      <div className="pointer-events-none fixed right-4 top-24 z-50 flex w-[380px] max-w-[92vw] flex-col gap-2">
        {toasts.map((n) => (
          <div
            key={n.id}
            className={
              'pointer-events-auto animate-in fade-in slide-in-from-right-4 duration-300 rounded-lg border p-4 shadow-2xl backdrop-blur ' +
              (n.kind === 'maintenance'
                ? 'border-red-500/60 bg-red-950/90 light:bg-red-50 light:border-red-400'
                : n.kind === 'drift'
                  ? 'border-amber-500/50 bg-amber-950/85 light:bg-amber-50 light:border-amber-300'
                  : 'border-cyan-500/50 bg-cyan-950/85 light:bg-cyan-50 light:border-cyan-300')
            }
          >
            <div className="flex items-start gap-3">
              <span
                className={
                  n.kind === 'maintenance'
                    ? 'text-red-400'
                    : n.kind === 'drift'
                      ? 'text-amber-400'
                      : 'text-cyan-400 light:text-cyan-600'
                }
              >
                {n.kind === 'maintenance' ? (
                  <Wrench className="h-5 w-5" />
                ) : n.kind === 'drift' ? (
                  <AlertTriangle className="h-5 w-5" />
                ) : (
                  <RefreshCw className="h-5 w-5" />
                )}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-white light:text-gray-900">{n.title}</div>
                <div className="mt-1 text-xs leading-relaxed text-slate-300 light:text-gray-600">{n.message}</div>
              </div>
              <button
                onClick={() => setToasts((t) => t.filter((x) => x.id !== n.id))}
                className="text-slate-500 transition hover:text-slate-300 light:hover:text-gray-700"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-6">
        {/* Status row — matches the Orders page live-status indicators */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <span className="flex items-center text-slate-300 light:text-gray-700">
            <span
              className={`mr-2 h-2 w-2 rounded-full ${
                liveProvider ? 'animate-pulse bg-green-400' : 'bg-slate-500'
              }`}
            ></span>
            {liveProvider ? 'AI Assistant Online' : 'AI Assistant Offline'}
          </span>
          <span className="flex items-center text-slate-300 light:text-gray-700">
            <span
              className={`mr-2 h-2 w-2 rounded-full ${
                health?.ml_model_ready ? 'animate-pulse bg-green-400' : 'bg-red-400'
              }`}
            ></span>
            {health?.ml_model_ready ? 'AI Ready' : 'AI Not Ready'}
          </span>
          {insights && (
            <span className="text-slate-400 light:text-gray-500">
              {insights.insights.meta.row_count.toLocaleString()} doses ·{' '}
              {insights.insights.meta.batch_count} batches · {insights.insights.meta.product_count} products
            </span>
          )}
        </div>

        <Tabs defaultValue="assistant" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="assistant">
              <Bot className="mr-2 h-4 w-4" /> AI Assistant
            </TabsTrigger>
            <TabsTrigger value="predict">
              <Activity className="mr-2 h-4 w-4" /> Live Monitoring
            </TabsTrigger>
          </TabsList>

          {/* ================= USE CASE 1 — Generative ================= */}
          <TabsContent value="assistant" className="space-y-6">
            {overall && (
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
                <StatCard
                  label="Accuracy score"
                  value={`${overall.accuracy_score}%`}
                  subtitle="overall dosing accuracy"
                  icon={<Target className="h-6 w-6 text-white" />}
                  color="blue"
                  onClick={() => setTab1DoseModal({ status: 'all', title: `All Doses (${overall.count})` })}
                />
                <StatCard
                  label="On target"
                  value={`${overall.on_target_pct}%`}
                  subtitle={`${overall.on_target} doses`}
                  icon={<CheckCircle2 className="h-6 w-6 text-white" />}
                  color="green"
                  onClick={() => setTab1DoseModal({ status: 'on_target', title: `On-Target Doses (${overall.on_target})` })}
                />
                <StatCard
                  label="Over-dosed"
                  value={`${overall.over}`}
                  subtitle={`${overall.over_pct}% of all doses`}
                  icon={<TrendingUp className="h-6 w-6 text-white" />}
                  color="orange"
                  onClick={() => setTab1DoseModal({ status: 'over', title: `Over-Dosed (${overall.over})` })}
                />
                <StatCard
                  label="Flagged doses"
                  value={`${overall.flagged}`}
                  subtitle={`${Math.round((overall.flagged / overall.count) * 100)}% need review`}
                  icon={<AlertTriangle className="h-6 w-6 text-white" />}
                  color="red"
                  onClick={() => setTab1DoseModal({ status: 'flagged', title: `Flagged Doses (${overall.flagged})` })}
                />
              </div>
            )}

            <Panel title="AI Dosing-Accuracy Briefing" icon={<Sparkles className="h-5 w-5" />}>
              {insightsLoading ? (
                <div className="flex items-center gap-2 text-slate-400 light:text-gray-500">
                  <Loader2 className="h-4 w-4 animate-spin" /> Generating briefing…
                </div>
              ) : (
                <>
                  {insights?.summary && <AnswerCard text={insights.summary} />}
                </>
              )}
            </Panel>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {distData && <ChartComponent type="doughnut" data={distData} title="Dose outcomes" height={240} />}
              {worstMaterials.length > 0 && (
                <Panel title="Ingredients to Watch" icon={<AlertTriangle className="h-5 w-5" />}>
                  <MaterialWatchlist materials={worstMaterials} />
                </Panel>
              )}
            </div>

            <Panel title="Ask Hercules" icon={<Bot className="h-5 w-5" />}>
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  {SUGGESTED.map((s) => (
                    <button
                      key={s}
                      onClick={() => handleAsk(s)}
                      disabled={asking}
                      className="rounded-full border border-slate-600/50 light:border-gray-300 bg-slate-900/40 light:bg-white px-3 py-1 text-xs text-slate-300 light:text-gray-700 transition hover:border-cyan-500/50 hover:text-cyan-300 light:hover:text-cyan-600 disabled:opacity-50"
                    >
                      {s}
                    </button>
                  ))}
                </div>

                <div className="max-h-[420px] space-y-3 overflow-y-auto rounded-lg border border-slate-700/50 light:border-gray-200 bg-slate-900/40 light:bg-gray-50 p-4">
                  {messages.length === 0 && (
                    <p className="text-sm text-slate-500 light:text-gray-400">
                      Ask anything about the plant's dosing accuracy — over/under-dosing, worst batches,
                      product comparisons, or a live prediction (e.g. "will 0.9kg of Copper proteinate overdose?").
                    </p>
                  )}
                  {messages.map((m, i) => (
                    <div key={i} className={m.role === 'user' ? 'text-right' : 'text-left'}>
                      <div
                        className={
                          'inline-block max-w-[85%] rounded-2xl px-4 py-2 text-sm ' +
                          (m.role === 'user'
                            ? 'bg-cyan-600/20 text-cyan-100 light:bg-cyan-100 light:text-cyan-900'
                            : 'border border-slate-700/50 light:border-gray-200 bg-slate-800 light:bg-white text-slate-100 light:text-gray-900')
                        }
                      >
                        {m.role === 'ai' ? (
                          <AnswerCard text={m.text} />
                        ) : (
                          <p className="whitespace-pre-wrap leading-relaxed">{m.text}</p>
                        )}
                        {m.chart && m.chart.labels?.length > 0 && (
                          <div className="mt-3 w-[min(520px,70vw)]">
                            <ChartComponent
                              type={m.chart.type === 'line' ? 'line' : 'bar'}
                              data={specToChartData(m.chart)}
                              title={m.chart.title || 'Chart'}
                              height={200}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {asking && (
                    <div className="flex items-center gap-2 text-sm text-slate-400 light:text-gray-500">
                      <Loader2 className="h-4 w-4 animate-spin" /> Hercules AI is thinking…
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div className="flex gap-2">
                  <Input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
                    placeholder="Ask about dosing accuracy…"
                    className={FIELD}
                  />
                  <Button onClick={() => handleAsk()} disabled={asking || !question.trim()}>
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </Panel>
          </TabsContent>

          {/* ================= USE CASE 2 — Automatic live monitoring ================= */}
          <TabsContent value="predict" className="space-y-6">
            {/* Live control / status bar */}
            <div className="flex flex-col gap-3 rounded-lg border border-slate-700/50 light:border-gray-200 bg-slate-800/50 light:bg-white p-4 shadow-lg light:shadow-xl md:flex-row md:items-center md:justify-between">
              <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
                <span className="flex items-center gap-2 text-slate-300 light:text-gray-700">
                  <Radio className={`h-4 w-4 ${live?.running ? 'text-green-400' : 'text-slate-500'}`} />
                  <span className={`h-2 w-2 rounded-full ${live?.running ? 'animate-pulse bg-green-400' : 'bg-slate-500'}`}></span>
                  <span className="font-semibold">{live?.running ? 'LIVE' : 'PAUSED'}</span>
                </span>
                <span className={`flex items-center gap-2 rounded-full border px-2.5 py-0.5 text-xs font-medium ${driftMeta.ring} ${driftMeta.text}`}>
                  <span className={`h-2 w-2 rounded-full ${driftMeta.dot} ${dr?.status === 'drift' ? 'animate-pulse' : ''}`}></span>
                  {live?.retraining ? 'Updating…' : driftMeta.label}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="mr-1 hidden text-[10px] uppercase tracking-wide text-slate-500 light:text-gray-400 sm:inline">
                  Demo speed
                </span>
                <div className="mr-1 flex items-center overflow-hidden rounded-md border border-slate-700 light:border-gray-300">
                  {speedPresets.map((s) => (
                    <button
                      key={s.label}
                      onClick={() => control('speed', s.value)}
                      className={`px-2.5 py-1 text-xs transition ${
                        Math.abs((live?.speed ?? 1.1) - s.value) < 0.05
                          ? 'bg-cyan-600 text-white'
                          : 'text-slate-400 light:text-gray-500 hover:text-slate-200 light:hover:text-gray-800'
                      }`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
                <Button variant="outline" size="sm" onClick={() => control(live?.running ? 'pause' : 'start')}>
                  {live?.running ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Button variant="outline" size="sm" onClick={() => control('reset')}>
                  <RefreshCw className="mr-1.5 h-4 w-4" /> Restart
                </Button>
              </div>
            </div>

            {/* Live KPI row */}
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
              <StatCard
                label="Batches monitored"
                value={`${st?.processed ?? 0}`}
                subtitle={live ? `${streamPct}% of ${live.total} total` : 'starting…'}
                icon={<Activity className="h-6 w-6 text-white" />}
                color="blue"
                onClick={() => setTab2FeedModal({ items: feedAll, title: `Batches Monitored (${feedAll.length})` })}
              />
              <StatCard
                label="Needs Attention"
                value={`${st?.flagged ?? 0}`}
                subtitle={`${flaggedPct}% of monitored so far`}
                icon={<AlertTriangle className="h-6 w-6 text-white" />}
                color="red"
                onClick={() =>
                  setTab2FeedModal({ items: feedAll.filter((f) => f.flagged), title: `Needs Attention (${feedFlaggedCount})` })
                }
              />
              <StatCard
                label="On Target"
                value={`${onTargetRate}%`}
                subtitle={`${st?.on_target ?? 0} of ${st?.processed ?? 0} doses`}
                icon={<CheckCircle2 className="h-6 w-6 text-white" />}
                color="green"
                onClick={() =>
                  setTab2FeedModal({ items: feedAll.filter((f) => !f.flagged), title: `On Target (${feedOkCount})` })
                }
              />
            </div>

            {/* Live operator feed */}
            <Panel
              title="Live operator feed"
              icon={<Radio className="h-5 w-5" />}
              right={
                <span className="flex items-center gap-1.5 text-xs text-slate-400 light:text-gray-500">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-green-400"></span> streaming
                </span>
              }
            >
              <p className="mb-3 text-xs leading-relaxed text-slate-400 light:text-gray-500">
                Every batch is checked <b>before it runs</b>. Flagged ones need a quick look.
              </p>
              <div className="mb-3 flex flex-wrap items-center gap-2">
                {(
                  [
                    ['all', `All (${feedAll.length})`],
                    ['flagged', `Needs Attention (${feedFlaggedCount})`],
                    ['ok', `On Target (${feedOkCount})`],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setFeedFilter(key)}
                    className={`rounded-full border px-3 py-1 text-xs transition ${
                      feedFilter === key
                        ? 'border-cyan-500 bg-cyan-600/20 text-cyan-300'
                        : 'border-slate-600/50 light:border-gray-300 text-slate-400 light:text-gray-500 hover:text-slate-200 light:hover:text-gray-800'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="max-h-[460px] space-y-2 overflow-y-auto pr-1">
                {(!live || live.feed.length === 0) && (
                  <div className="flex items-center gap-2 py-6 text-sm text-slate-500 light:text-gray-400">
                    <Loader2 className="h-4 w-4 animate-spin" /> Waiting for the first batch…
                  </div>
                )}
                {live && live.feed.length > 0 && filteredFeed.length === 0 && (
                  <p className="py-6 text-center text-sm text-slate-500 light:text-gray-400">
                    Nothing in this view yet.
                  </p>
                )}
                {filteredFeed.map((f) => (
                  <FeedRow key={f.id} f={f} />
                ))}
              </div>
            </Panel>

            {/* Live charts */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Panel title="Recent Batches" icon={<Activity className="h-5 w-5" />}>
                <p className="mb-3 text-xs leading-relaxed text-slate-400 light:text-gray-500">
                  How the last {recentOutcomes.length || 0} doses actually turned out — oldest to newest.
                </p>
                {recentOutcomes.length > 0 ? (
                  <>
                    <OutcomeStrip items={recentOutcomes} />
                    <div className="mt-4 flex items-center gap-4 text-[11px] text-slate-400 light:text-gray-500">
                      <span className="flex items-center gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500"></span> On target
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-sm bg-amber-500"></span> Over
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-sm bg-red-500"></span> Under
                      </span>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-slate-500 light:text-gray-400">Waiting for the first batch…</p>
                )}
              </Panel>
              {outcomeMix && (
                <ChartComponent type="doughnut" data={outcomeMix} title="Live dose outcomes (this session)" height={240} />
              )}
            </div>

            {/* Engineer what-if (hidden by default) */}
            <Panel
              title="Engineer what-if"
              icon={<FlaskConical className="h-5 w-5" />}
              right={
                <Button variant="outline" size="sm" onClick={() => setWhatIfOpen((v) => !v)}>
                  {whatIfOpen ? 'Hide' : 'Open'}
                </Button>
              }
            >
              {!whatIfOpen ? (
                <p className="text-sm text-slate-400 light:text-gray-500">
                  Operators don't use this — the plant feed is scored automatically above. This is a manual
                  sandbox for engineers to test a hypothetical dose against the model.
                </p>
              ) : (
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <div className="space-y-4">
                    <p className="text-xs leading-relaxed text-slate-400 light:text-gray-500">
                      One model looks at just two things — the ingredient and the target weight — and sorts the
                      dose into one of five buckets. Risk, severity and direction all come from that single call.
                    </p>
                    <div className="space-y-2">
                      <label className="text-xs uppercase tracking-wide text-slate-400 light:text-gray-500">Ingredient</label>
                      <Select value={material} onValueChange={setMaterial}>
                        <SelectTrigger className={FIELD}>
                          <SelectValue placeholder="Select ingredient" />
                        </SelectTrigger>
                        <SelectContent className={`max-h-72 ${FIELD}`}>
                          {doseInfo?.options.materials.map((m) => (
                            <SelectItem key={m.material_code} value={m.material_code}>
                              {m.material_name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs uppercase tracking-wide text-slate-400 light:text-gray-500">Target weight (kg)</label>
                      <Input type="number" step="0.1" value={setpoint} onChange={(e) => setSetpoint(e.target.value)} className={FIELD} />
                    </div>
                    <Button onClick={handlePredict} disabled={predicting} className="w-full">
                      {predicting ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Score this dose'}
                    </Button>
                  </div>

                  <div>
                    {dosePrediction ? (
                      <div className="rounded-lg border border-slate-700/50 light:border-gray-200 bg-slate-900/40 light:bg-gray-50 p-4">
                        <div className="text-center">
                          <div className="text-xs font-medium uppercase tracking-wide text-slate-400 light:text-gray-500">
                            Dose Quality Model · one model, three answers
                          </div>
                          <div className={`mt-1 inline-block rounded-lg border px-4 py-1 text-4xl font-bold ${bandColor(dosePrediction.band)}`}>
                            {dosePrediction.risk_pct}%
                          </div>
                          <div className="mt-1 text-sm font-medium text-slate-200 light:text-gray-700">
                            {dosePrediction.band} risk · {dosePrediction.prediction}
                          </div>
                        </div>

                        {dosePrediction.flagged && (
                          <div className="mt-3 grid grid-cols-2 gap-2">
                            <div className="rounded-lg border border-slate-700/50 light:border-gray-200 p-2 text-center">
                              <div className="text-[10px] uppercase tracking-wide text-slate-400 light:text-gray-500">How bad</div>
                              <div className="text-sm font-semibold text-slate-100 light:text-gray-800">{dosePrediction.severity_label}</div>
                              <div className="text-[10px] text-slate-500">{dosePrediction.severity_confidence_pct}%</div>
                            </div>
                            <div className="rounded-lg border border-slate-700/50 light:border-gray-200 p-2 text-center">
                              <div className="text-[10px] uppercase tracking-wide text-slate-400 light:text-gray-500">Which way</div>
                              <div className={`flex items-center justify-center gap-1 text-sm font-semibold ${dosePrediction.direction === 'over' ? 'text-amber-400' : 'text-cyan-400'}`}>
                                {dosePrediction.direction === 'over' ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                                {dosePrediction.direction === 'over' ? 'Runs high' : 'Runs low'}
                              </div>
                              <div className="text-[10px] text-slate-500">{dosePrediction.direction_confidence_pct}%</div>
                            </div>
                          </div>
                        )}

                        <div className="mt-4">
                          <div className="mb-1.5 text-[11px] uppercase tracking-wide text-slate-400 light:text-gray-500">
                            The 5 buckets — how the AI sees this dose
                          </div>
                          <div className="space-y-1.5">
                            {DOSE_ORDER.map((cls) => {
                              const meta = DOSE_META[cls]
                              const pct = dosePrediction.class_probabilities[cls] ?? 0
                              const isTop = cls === dosePrediction.top_class
                              return (
                                <div key={cls} className="flex items-center gap-2">
                                  <span className={`w-24 shrink-0 text-[11px] ${isTop ? meta.text + ' font-semibold' : 'text-slate-400 light:text-gray-500'}`}>
                                    {meta.label}
                                  </span>
                                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800 light:bg-gray-200">
                                    <div className={`h-full ${meta.bar}`} style={{ width: `${pct}%` }} />
                                  </div>
                                  <span className="w-10 shrink-0 text-right text-[11px] text-slate-400 light:text-gray-500">{pct}%</span>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                        <p className="mt-3 text-[11px] text-slate-500 light:text-gray-400">
                          {dosePrediction.inputs.material_name} · {dosePrediction.inputs.setpoint} kg. Everything
                          above is read from this single prediction, so the three answers can't contradict.
                        </p>
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500 light:text-gray-400">
                        Pick an ingredient and target weight, then score the dose.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </Panel>
          </TabsContent>
        </Tabs>
      </div>

      {tab1DoseModal && (
        <DoseListDialog
          status={tab1DoseModal.status}
          title={tab1DoseModal.title}
          onClose={() => setTab1DoseModal(null)}
        />
      )}
      {tab2FeedModal && (
        <FeedListDialog items={tab2FeedModal.items} title={tab2FeedModal.title} onClose={() => setTab2FeedModal(null)} />
      )}
    </WaterSystemLayout>
  )
}

export default AiAssistant
