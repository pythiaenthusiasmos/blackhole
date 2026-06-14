import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls.js'
import './App.css'

type RayParams = {
  startR: number
  startTheta: number
  startPhi: number
  lz: number
  q: number
  radialSign: number
  thetaSign: number
}

type MetricParams = {
  mass: number
  spin: number
}

type TraceParams = {
  steps: number
  stepSize: number
}

type ClusterParams = {
  mode: 'single' | '1d' | '2d'
  axisA: ClusterKey
  axisB: ClusterKey
  countA: number
  countB: number
  stepA: number
  stepB: number
}

type RenderParams = {
  eventHorizon: boolean
  ergosphere: boolean
}

type ClusterKey = 'startR' | 'startTheta' | 'startPhi' | 'lz' | 'q' | 'spin'

type RayTrace = {
  points: THREE.Vector3[]
  color: string
  stopped: 'horizon' | 'turning-point' | 'escaped' | 'complete'
}

const clusterOptions: Array<{ key: ClusterKey; label: string }> = [
  { key: 'startR', label: 'Start r' },
  { key: 'startTheta', label: 'Start theta' },
  { key: 'startPhi', label: 'Start phi' },
  { key: 'lz', label: 'Lz' },
  { key: 'q', label: 'Carter Q' },
  { key: 'spin', label: 'Spin a' },
]

const rayControls: Array<{
  key: keyof RayParams
  label: string
  min: number
  max: number
  step: number
  suffix?: string
}> = [
  { key: 'startR', label: 'Start r', min: 2.2, max: 28, step: 0.1, suffix: ' M' },
  { key: 'startTheta', label: 'Start theta', min: 3, max: 177, step: 1, suffix: ' deg' },
  { key: 'startPhi', label: 'Start phi', min: 0, max: 360, step: 1, suffix: ' deg' },
  { key: 'lz', label: 'Angular momentum Lz', min: -9, max: 9, step: 0.05 },
  { key: 'q', label: 'Carter Q', min: 0, max: 64, step: 0.2 },
]

const metricControls: Array<{
  key: keyof MetricParams
  label: string
  min: number
  max: number
  step: number
}> = [
  { key: 'mass', label: 'Mass M', min: 0.5, max: 3, step: 0.05 },
  { key: 'spin', label: 'Spin a', min: -0.99, max: 0.99, step: 0.01 },
]

const traceControls: Array<{
  key: keyof TraceParams
  label: string
  min: number
  max: number
  step: number
}> = [
  { key: 'steps', label: 'Integrator steps', min: 200, max: 6000, step: 100 },
  { key: 'stepSize', label: 'Step size', min: 0.01, max: 0.2, step: 0.01 },
]

const defaultRay: RayParams = {
  startR: 15,
  startTheta: 90,
  startPhi: 0,
  lz: 3.2,
  q: 8,
  radialSign: -1,
  thetaSign: 1,
}

const defaultMetric: MetricParams = {
  mass: 1,
  spin: 0.72,
}

const defaultTrace: TraceParams = {
  steps: 2200,
  stepSize: 0.055,
}

const defaultCluster: ClusterParams = {
  mode: '1d',
  axisA: 'lz',
  axisB: 'q',
  countA: 17,
  countB: 9,
  stepA: 0.24,
  stepB: 0.8,
}

const defaultRender: RenderParams = {
  eventHorizon: true,
  ergosphere: true,
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function degreesToRadians(value: number) {
  return (value * Math.PI) / 180
}

function formatValue(value: number, step: number) {
  return value.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0)
}

function horizonRadius(metric: MetricParams) {
  const mass = metric.mass
  const spin = clamp(metric.spin, -0.999 * mass, 0.999 * mass)
  return mass + Math.sqrt(Math.max(0, mass * mass - spin * spin))
}

function ergosphereRadius(metric: MetricParams, theta: number) {
  const mass = metric.mass
  const spin = clamp(metric.spin, -0.999 * mass, 0.999 * mass)
  return mass + Math.sqrt(Math.max(0, mass * mass - spin * spin * Math.cos(theta) ** 2))
}

function kerrToCartesian(r: number, theta: number, phi: number, spin: number) {
  const radial = Math.sqrt(Math.max(0, r * r + spin * spin))
  return new THREE.Vector3(
    radial * Math.sin(theta) * Math.cos(phi),
    r * Math.cos(theta),
    radial * Math.sin(theta) * Math.sin(phi),
  )
}

function applyClusterOffset(ray: RayParams, metric: MetricParams, key: ClusterKey, offset: number) {
  const nextRay = { ...ray }
  const nextMetric = { ...metric }

  if (key === 'spin') {
    nextMetric.spin = clamp(nextMetric.spin + offset, -0.99 * nextMetric.mass, 0.99 * nextMetric.mass)
  } else if (key === 'startTheta') {
    nextRay.startTheta = clamp(nextRay.startTheta + offset, 1, 179)
  } else if (key === 'startPhi') {
    nextRay.startPhi = ((nextRay.startPhi + offset) % 360 + 360) % 360
  } else if (key === 'startR') {
    nextRay.startR = Math.max(1.2, nextRay.startR + offset)
  } else if (key === 'q') {
    nextRay.q = Math.max(0, nextRay.q + offset)
  } else {
    nextRay.lz += offset
  }

  return { ray: nextRay, metric: nextMetric }
}

function rayEquations(ray: RayParams, metric: MetricParams, trace: TraceParams) {
  const points: THREE.Vector3[] = []
  const mass = metric.mass
  const spin = clamp(metric.spin, -0.999 * mass, 0.999 * mass)
  const horizon = horizonRadius(metric)
  let r = Math.max(ray.startR, horizon + 0.02)
  let theta = clamp(degreesToRadians(ray.startTheta), 0.01, Math.PI - 0.01)
  let phi = degreesToRadians(ray.startPhi)
  let radialSign = ray.radialSign >= 0 ? 1 : -1
  let thetaSign = ray.thetaSign >= 0 ? 1 : -1
  const energy = 1
  const maxR = Math.max(36, ray.startR * 2.8)
  let stopped: RayTrace['stopped'] = 'complete'

  for (let step = 0; step < trace.steps; step += 1) {
    points.push(kerrToCartesian(r, theta, phi, spin))

    if (r <= horizon + 0.035) {
      stopped = 'horizon'
      break
    }

    if (r > maxR && radialSign > 0) {
      stopped = 'escaped'
      break
    }

    const sinTheta = Math.max(0.025, Math.sin(theta))
    const cosTheta = Math.cos(theta)
    const delta = Math.max(0.0005, r * r - 2 * mass * r + spin * spin)
    const sigma = Math.max(0.0005, r * r + spin * spin * cosTheta * cosTheta)
    const p = energy * (r * r + spin * spin) - spin * ray.lz
    const radialPotential =
      p * p - delta * (ray.q + (ray.lz - spin * energy) * (ray.lz - spin * energy))
    const thetaPotential =
      ray.q + spin * spin * energy * energy * cosTheta * cosTheta - (ray.lz * ray.lz * cosTheta * cosTheta) / (sinTheta * sinTheta)

    if (radialPotential < 0 && thetaPotential < 0) {
      stopped = 'turning-point'
      break
    }

    let dr = 0
    if (radialPotential >= 0) {
      dr = (radialSign * Math.sqrt(radialPotential)) / sigma
    } else {
      radialSign *= -1
    }

    let dTheta = 0
    if (thetaPotential >= 0) {
      dTheta = (thetaSign * Math.sqrt(thetaPotential)) / sigma
    } else {
      thetaSign *= -1
    }

    const dPhi = (ray.lz / (sinTheta * sinTheta) - spin * energy + (spin * p) / delta) / sigma

    r += dr * trace.stepSize
    theta += dTheta * trace.stepSize
    phi += dPhi * trace.stepSize

    if (theta <= 0.01 || theta >= Math.PI - 0.01) {
      theta = clamp(theta, 0.01, Math.PI - 0.01)
      thetaSign *= -1
    }
  }

  return { points, stopped }
}

function buildRayCluster(ray: RayParams, metric: MetricParams, trace: TraceParams, cluster: ClusterParams) {
  const countA = cluster.mode === 'single' ? 1 : Math.max(1, Math.round(cluster.countA))
  const countB = cluster.mode === '2d' ? Math.max(1, Math.round(cluster.countB)) : 1
  const rays: RayTrace[] = []

  for (let a = 0; a < countA; a += 1) {
    for (let b = 0; b < countB; b += 1) {
      const offsetA = (a - (countA - 1) / 2) * cluster.stepA
      const offsetB = (b - (countB - 1) / 2) * cluster.stepB
      let variant = applyClusterOffset(ray, metric, cluster.axisA, offsetA)

      if (cluster.mode === '2d') {
        variant = applyClusterOffset(variant.ray, variant.metric, cluster.axisB, offsetB)
      }

      const traced = rayEquations(variant.ray, variant.metric, trace)
      const hue = (210 + a * 17 + b * 43) % 360
      rays.push({ ...traced, color: `hsl(${hue} 85% 62%)` })
    }
  }

  return rays
}

function makeHorizonMesh(metric: MetricParams) {
  const spin = metric.spin
  const radius = horizonRadius(metric)
  const geometry = new THREE.SphereGeometry(radius, 80, 40)
  const position = geometry.attributes.position

  for (let index = 0; index < position.count; index += 1) {
    const vertex = new THREE.Vector3().fromBufferAttribute(position, index)
    const theta = Math.acos(clamp(vertex.y / radius, -1, 1))
    const phi = Math.atan2(vertex.z, vertex.x)
    const next = kerrToCartesian(radius, theta, phi, spin)
    position.setXYZ(index, next.x, next.y, next.z)
  }

  position.needsUpdate = true
  geometry.computeVertexNormals()
  return geometry
}

function makeErgosphereMesh(metric: MetricParams) {
  const spin = metric.spin
  const widthSegments = 96
  const heightSegments = 48
  const geometry = new THREE.BufferGeometry()
  const vertices: number[] = []
  const indices: number[] = []

  for (let y = 0; y <= heightSegments; y += 1) {
    const theta = (y / heightSegments) * Math.PI
    for (let x = 0; x <= widthSegments; x += 1) {
      const phi = (x / widthSegments) * Math.PI * 2
      const r = ergosphereRadius(metric, theta)
      const point = kerrToCartesian(r, theta, phi, spin)
      vertices.push(point.x, point.y, point.z)
    }
  }

  for (let y = 0; y < heightSegments; y += 1) {
    for (let x = 0; x < widthSegments; x += 1) {
      const a = y * (widthSegments + 1) + x
      const b = a + widthSegments + 1
      indices.push(a, b, a + 1, b, b + 1, a + 1)
    }
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

function KerrScene({
  rays,
  metric,
  render,
}: {
  rays: RayTrace[]
  metric: MetricParams
  render: RenderParams
}) {
  const mountRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const mount = mountRef.current

    if (!mount) {
      return
    }

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x06080d)
    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 180)
    camera.position.set(13, 8, 14)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    mount.appendChild(renderer.domElement)

    const controls = new TrackballControls(camera, renderer.domElement)
    controls.rotateSpeed = 3.2
    controls.zoomSpeed = 1
    controls.panSpeed = 0.6
    controls.dynamicDampingFactor = 0.1

    scene.add(new THREE.AmbientLight(0xb7c7ff, 1.5))
    const light = new THREE.DirectionalLight(0xffffff, 1.8)
    light.position.set(6, 9, 5)
    scene.add(light)

    const disk = new THREE.Mesh(
      new THREE.RingGeometry(horizonRadius(metric) * 1.35, 15, 160),
      new THREE.MeshBasicMaterial({
        color: 0xffb25b,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    )
    disk.rotation.x = -Math.PI / 2
    scene.add(disk)

    if (render.eventHorizon) {
      const horizon = new THREE.Mesh(
        makeHorizonMesh(metric),
        new THREE.MeshStandardMaterial({
          color: 0x111827,
          roughness: 0.55,
          metalness: 0.05,
          transparent: true,
          opacity: 0.72,
        }),
      )
      scene.add(horizon)
    }

    if (render.ergosphere) {
      const ergo = new THREE.Mesh(
        makeErgosphereMesh(metric),
        new THREE.MeshStandardMaterial({
          color: 0x3dc9ff,
          roughness: 0.35,
          transparent: true,
          opacity: 0.18,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      )
      scene.add(ergo)
    }

    const axes = new THREE.Group()
    const axisMaterial = new THREE.LineBasicMaterial({ color: 0x9fb7c7, transparent: true, opacity: 0.35 })
    const axisGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-18, 0, 0),
      new THREE.Vector3(18, 0, 0),
      new THREE.Vector3(0, -12, 0),
      new THREE.Vector3(0, 12, 0),
      new THREE.Vector3(0, 0, -18),
      new THREE.Vector3(0, 0, 18),
    ])
    axes.add(new THREE.LineSegments(axisGeometry, axisMaterial))
    scene.add(axes)

    rays.forEach((ray) => {
      if (ray.points.length < 2) {
        return
      }

      const geometry = new THREE.BufferGeometry().setFromPoints(ray.points)
      const material = new THREE.LineBasicMaterial({
        color: new THREE.Color(ray.color),
        transparent: true,
        opacity: ray.stopped === 'turning-point' ? 0.42 : 0.78,
      })
      scene.add(new THREE.Line(geometry, material))
    })

    const resize = () => {
      const width = Math.max(mount.clientWidth, 1)
      const height = Math.max(mount.clientHeight, 1)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      controls.handleResize()
    }

    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(mount)
    resize()

    let animationFrame = 0
    const animate = () => {
      animationFrame = requestAnimationFrame(animate)
      disk.rotation.z += 0.001
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(animationFrame)
      resizeObserver.disconnect()
      controls.dispose()
      scene.traverse((object) => {
        if ('geometry' in object && object.geometry instanceof THREE.BufferGeometry) {
          object.geometry.dispose()
        }
        if ('material' in object) {
          const material = object.material

          if (Array.isArray(material)) {
            material.forEach((item) => item.dispose())
          } else if (material instanceof THREE.Material) {
            material.dispose()
          }
        }
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [metric, rays, render])

  return <div className="scene" ref={mountRef} aria-label="Interactive Kerr ray tracing scene" />
}

function NumericControl<T extends Record<string, number>>({
  item,
  values,
  onChange,
}: {
  item: { key: keyof T; label: string; min: number; max: number; step: number; suffix?: string }
  values: T
  onChange: (key: keyof T, value: number) => void
}) {
  return (
    <label className="control">
      <span>
        {item.label}
        <strong>
          {formatValue(values[item.key], item.step)}
          {item.suffix ?? ''}
        </strong>
      </span>
      <input
        type="range"
        min={item.min}
        max={item.max}
        step={item.step}
        value={values[item.key]}
        onChange={(event) => onChange(item.key, Number(event.target.value))}
      />
    </label>
  )
}

function App() {
  const [ray, setRay] = useState(defaultRay)
  const [metric, setMetric] = useState(defaultMetric)
  const [trace, setTrace] = useState(defaultTrace)
  const [cluster, setCluster] = useState(defaultCluster)
  const [render, setRender] = useState(defaultRender)
  const rays = useMemo(() => buildRayCluster(ray, metric, trace, cluster), [cluster, metric, ray, trace])
  const stats = useMemo(() => {
    const horizon = rays.filter((item) => item.stopped === 'horizon').length
    const escaped = rays.filter((item) => item.stopped === 'escaped').length
    const turning = rays.filter((item) => item.stopped === 'turning-point').length

    return { horizon, escaped, turning, total: rays.length }
  }, [rays])

  const updateRay = (key: keyof RayParams, value: number) => {
    setRay((current) => ({ ...current, [key]: value }))
  }

  const updateMetric = (key: keyof MetricParams, value: number) => {
    setMetric((current) => {
      const next = { ...current, [key]: value }
      next.spin = clamp(next.spin, -0.99 * next.mass, 0.99 * next.mass)
      return next
    })
  }

  const updateTrace = (key: keyof TraceParams, value: number) => {
    setTrace((current) => ({ ...current, [key]: value }))
  }

  return (
    <main className="app-shell">
      <aside className="panel" aria-label="Kerr ray controls">
        <header className="brand">
          <h1>Blackhole</h1>
          <p>Kerr null geodesic explorer</p>
        </header>

        <section className="control-section">
          <div className="section-title">Metric</div>
          {metricControls.map((item) => (
            <NumericControl item={item} key={item.key} values={metric} onChange={updateMetric} />
          ))}
        </section>

        <section className="control-section">
          <div className="section-title">Ray</div>
          {rayControls.map((item) => (
            <NumericControl item={item} key={item.key} values={ray} onChange={updateRay} />
          ))}
          <div className="button-row" aria-label="Ray direction">
            <button
              className={ray.radialSign < 0 ? 'active' : ''}
              type="button"
              onClick={() => updateRay('radialSign', -1)}
            >
              Inward
            </button>
            <button
              className={ray.radialSign > 0 ? 'active' : ''}
              type="button"
              onClick={() => updateRay('radialSign', 1)}
            >
              Outward
            </button>
          </div>
          <div className="button-row" aria-label="Polar direction">
            <button
              className={ray.thetaSign < 0 ? 'active' : ''}
              type="button"
              onClick={() => updateRay('thetaSign', -1)}
            >
              North
            </button>
            <button
              className={ray.thetaSign > 0 ? 'active' : ''}
              type="button"
              onClick={() => updateRay('thetaSign', 1)}
            >
              South
            </button>
          </div>
        </section>

        <section className="control-section">
          <div className="section-title">Cluster</div>
          <div className="button-row three" aria-label="Cluster mode">
            <button className={cluster.mode === 'single' ? 'active' : ''} type="button" onClick={() => setCluster((current) => ({ ...current, mode: 'single' }))}>
              Single
            </button>
            <button className={cluster.mode === '1d' ? 'active' : ''} type="button" onClick={() => setCluster((current) => ({ ...current, mode: '1d' }))}>
              1D
            </button>
            <button className={cluster.mode === '2d' ? 'active' : ''} type="button" onClick={() => setCluster((current) => ({ ...current, mode: '2d' }))}>
              2D
            </button>
          </div>
          <label className="select-control">
            <span>Axis A</span>
            <select value={cluster.axisA} onChange={(event) => setCluster((current) => ({ ...current, axisA: event.target.value as ClusterKey }))}>
              {clusterOptions.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="control">
            <span>
              Count A
              <strong>{cluster.countA}</strong>
            </span>
            <input
              type="range"
              min={1}
              max={41}
              step={2}
              value={cluster.countA}
              onChange={(event) => setCluster((current) => ({ ...current, countA: Number(event.target.value) }))}
            />
          </label>
          <label className="control">
            <span>
              Step A
              <strong>{cluster.stepA.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.02}
              value={cluster.stepA}
              onChange={(event) => setCluster((current) => ({ ...current, stepA: Number(event.target.value) }))}
            />
          </label>
          <label className="select-control">
            <span>Axis B</span>
            <select
              disabled={cluster.mode !== '2d'}
              value={cluster.axisB}
              onChange={(event) => setCluster((current) => ({ ...current, axisB: event.target.value as ClusterKey }))}
            >
              {clusterOptions.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="control">
            <span>
              Count B
              <strong>{cluster.countB}</strong>
            </span>
            <input
              type="range"
              min={1}
              max={25}
              step={2}
              disabled={cluster.mode !== '2d'}
              value={cluster.countB}
              onChange={(event) => setCluster((current) => ({ ...current, countB: Number(event.target.value) }))}
            />
          </label>
          <label className="control">
            <span>
              Step B
              <strong>{cluster.stepB.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min={-4}
              max={4}
              step={0.05}
              disabled={cluster.mode !== '2d'}
              value={cluster.stepB}
              onChange={(event) => setCluster((current) => ({ ...current, stepB: Number(event.target.value) }))}
            />
          </label>
        </section>

        <section className="control-section">
          <div className="section-title">Trace</div>
          {traceControls.map((item) => (
            <NumericControl item={item} key={item.key} values={trace} onChange={updateTrace} />
          ))}
          <label className="toggle">
            <input
              type="checkbox"
              checked={render.eventHorizon}
              onChange={(event) => setRender((current) => ({ ...current, eventHorizon: event.target.checked }))}
            />
            <span>Event horizon</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={render.ergosphere}
              onChange={(event) => setRender((current) => ({ ...current, ergosphere: event.target.checked }))}
            />
            <span>Ergosphere</span>
          </label>
        </section>
      </aside>

      <section className="workspace" aria-label="Kerr ray tracing workspace">
        <div className="stats" aria-label="Ray trace statistics">
          <div>
            <span>Rays</span>
            <strong>{stats.total}</strong>
          </div>
          <div>
            <span>Captured</span>
            <strong>{stats.horizon}</strong>
          </div>
          <div>
            <span>Escaped</span>
            <strong>{stats.escaped}</strong>
          </div>
          <div>
            <span>Turned</span>
            <strong>{stats.turning}</strong>
          </div>
        </div>
        <KerrScene rays={rays} metric={metric} render={render} />
      </section>
    </main>
  )
}

export default App
