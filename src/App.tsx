import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls.js'
import {
  AppFrame,
  ControlGroup,
  NumericControl,
  SegmentedControl,
  SelectControl,
  StatGrid,
  StatItem,
  ToggleControl,
} from '@openclaw/sim-ui'
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
  escapeRadius: number
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
  color: number
  stopped: 'horizon' | 'escaped' | 'range-limit'
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
  { key: 'startR', label: 'Start r', min: 0, max: 80, step: 0.1, suffix: ' M' },
  { key: 'startTheta', label: 'Start theta', min: 3, max: 177, step: 1, suffix: ' deg' },
  { key: 'startPhi', label: 'Start phi', min: 0, max: 360, step: 1, suffix: ' deg' },
  { key: 'lz', label: 'Angular momentum Lz/M', min: -30, max: 30, step: 0.05 },
  { key: 'q', label: 'Carter Q', min: 0, max: 256, step: 0.5 },
]

const metricControls: Array<{
  key: keyof MetricParams
  label: string
  min: number
  max: number
  step: number
}> = [
  { key: 'mass', label: 'Mass M', min: 0, max: 10, step: 0.05 },
  { key: 'spin', label: 'Spin a', min: -2, max: 2, step: 0.01 },
]

const traceControls: Array<{
  key: keyof TraceParams
  label: string
  min: number
  max: number
  step: number
}> = [
  { key: 'steps', label: 'Trace budget', min: 1000, max: 50000, step: 500 },
  { key: 'stepSize', label: 'Step size', min: 0.01, max: 0.2, step: 0.01 },
  { key: 'escapeRadius', label: 'Escape radius', min: 1, max: 180, step: 1 },
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
  steps: 14000,
  stepSize: 0.055,
  escapeRadius: 42,
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
    const spinLimit = Math.max(0, 0.99 * nextMetric.mass)
    nextMetric.spin = clamp(nextMetric.spin + offset, -spinLimit, spinLimit)
  } else if (key === 'startTheta') {
    nextRay.startTheta = clamp(nextRay.startTheta + offset, 1, 179)
  } else if (key === 'startPhi') {
    nextRay.startPhi = ((nextRay.startPhi + offset) % 360 + 360) % 360
  } else if (key === 'startR') {
    nextRay.startR = Math.max(0, nextRay.startR + offset)
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
  const lz = ray.lz * mass
  let r = Math.max(ray.startR, horizon > 0 ? horizon + 0.02 : 0.02)
  let theta = clamp(degreesToRadians(ray.startTheta), 0.01, Math.PI - 0.01)
  let phi = degreesToRadians(ray.startPhi)
  let radialSign = ray.radialSign >= 0 ? 1 : -1
  let thetaSign = ray.thetaSign >= 0 ? 1 : -1
  const energy = 1
  const escapeRadius = Math.max(trace.escapeRadius, ray.startR + 0.1, horizon + 1, 1)
  const maxSteps = Math.max(1, Math.round(trace.steps))
  const pointStride = Math.max(1, Math.floor(maxSteps / 6500))
  let stopped: RayTrace['stopped'] = 'range-limit'

  const pushPoint = () => {
    points.push(kerrToCartesian(r, theta, phi, spin))
  }

  pushPoint()

  for (let step = 0; step < maxSteps; step += 1) {
    if (horizon > 0 && r <= horizon + 0.035) {
      stopped = 'horizon'
      break
    }

    if (r >= escapeRadius) {
      stopped = 'escaped'
      break
    }

    const sinTheta = Math.max(0.025, Math.sin(theta))
    const cosTheta = Math.cos(theta)
    const delta = Math.max(0.0005, r * r - 2 * mass * r + spin * spin)
    const sigma = Math.max(0.0005, r * r + spin * spin * cosTheta * cosTheta)
    const p = energy * (r * r + spin * spin) - spin * lz
    const radialPotential =
      p * p - delta * (ray.q + (lz - spin * energy) * (lz - spin * energy))
    const thetaPotential =
      ray.q + spin * spin * energy * energy * cosTheta * cosTheta - (lz * lz * cosTheta * cosTheta) / (sinTheta * sinTheta)

    let dr = 0
    if (radialPotential >= 0) {
      dr = (radialSign * Math.sqrt(radialPotential)) / sigma
    } else {
      radialSign *= -1
      r = Math.max(horizon > 0 ? horizon + 0.04 : 0.02, r + radialSign * trace.stepSize * 0.5)
    }

    let dTheta = 0
    if (thetaPotential >= 0) {
      dTheta = (thetaSign * Math.sqrt(thetaPotential)) / sigma
    } else {
      thetaSign *= -1
      theta = clamp(theta + thetaSign * trace.stepSize * 0.5, 0.01, Math.PI - 0.01)
    }

    const dPhi = (lz / (sinTheta * sinTheta) - spin * energy + (spin * p) / delta) / sigma

    r += dr * trace.stepSize
    theta += dTheta * trace.stepSize
    phi += dPhi * trace.stepSize

    if (theta <= 0.01 || theta >= Math.PI - 0.01) {
      theta = clamp(theta, 0.01, Math.PI - 0.01)
      thetaSign *= -1
    }

    if (horizon > 0 && r <= horizon + 0.035) {
      stopped = 'horizon'
      pushPoint()
      break
    }

    if (r >= escapeRadius) {
      stopped = 'escaped'
      pushPoint()
      break
    }

    if ((step + 1) % pointStride === 0 || step === maxSteps - 1) {
      pushPoint()
    }
  }

  return { points, stopped }
}

function buildRayCluster(ray: RayParams, metric: MetricParams, trace: TraceParams, cluster: ClusterParams) {
  const countA = cluster.mode === 'single' ? 1 : Math.max(1, Math.round(cluster.countA))
  const countB = cluster.mode === '2d' ? Math.max(1, Math.round(cluster.countB)) : 1
  const total = countA * countB
  const rays: RayTrace[] = []
  let rayIndex = 0

  for (let a = 0; a < countA; a += 1) {
    for (let b = 0; b < countB; b += 1) {
      const offsetA = (a - (countA - 1) / 2) * cluster.stepA
      const offsetB = (b - (countB - 1) / 2) * cluster.stepB
      let variant = applyClusterOffset(ray, metric, cluster.axisA, offsetA)

      if (cluster.mode === '2d') {
        variant = applyClusterOffset(variant.ray, variant.metric, cluster.axisB, offsetB)
      }

      const traced = rayEquations(variant.ray, variant.metric, trace)
      const hue = total === 1 ? 205 : (rayIndex / Math.max(1, total - 1)) * 330
      const color = new THREE.Color().setHSL(hue / 360, 0.9, 0.58).getHex()
      rays.push({ ...traced, color })
      rayIndex += 1
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

    if (render.eventHorizon && horizonRadius(metric) > 0.001) {
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

    if (render.ergosphere && metric.mass > 0.001) {
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
        color: ray.color,
        transparent: true,
        opacity: ray.stopped === 'range-limit' ? 0.62 : 0.88,
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
    const limited = rays.filter((item) => item.stopped === 'range-limit').length

    return { horizon, escaped, limited, total: rays.length }
  }, [rays])

  const updateRay = (key: keyof RayParams, value: number) => {
    setRay((current) => ({ ...current, [key]: value }))
  }

  const updateMetric = (key: keyof MetricParams, value: number) => {
    setMetric((current) => {
      const next = { ...current, [key]: value }
      const spinLimit = Math.max(0, 0.99 * next.mass)
      next.spin = clamp(next.spin, -spinLimit, spinLimit)
      return next
    })
  }

  const updateTrace = (key: keyof TraceParams, value: number) => {
    setTrace((current) => ({ ...current, [key]: value }))
  }

  return (
    <AppFrame
      className="blackhole-app"
      title="Blackhole"
      viewportLabel="Kerr ray tracing workspace"
      controls={
        <>
          <ControlGroup title="Metric">
            {metricControls.map((item) => (
              <NumericControl item={item} key={item.key} values={metric} onChange={updateMetric} />
            ))}
          </ControlGroup>
          <ControlGroup title="Ray">
            {rayControls.map((item) => (
              <NumericControl item={item} key={item.key} values={ray} onChange={updateRay} />
            ))}
            <SegmentedControl
              label="Ray direction"
              value={ray.radialSign < 0 ? 'inward' : 'outward'}
              options={[
                { value: 'inward', label: 'Inward' },
                { value: 'outward', label: 'Outward' },
              ]}
              onChange={(value) => updateRay('radialSign', value === 'inward' ? -1 : 1)}
            />
            <SegmentedControl
              label="Polar direction"
              value={ray.thetaSign < 0 ? 'north' : 'south'}
              options={[
                { value: 'north', label: 'North' },
                { value: 'south', label: 'South' },
              ]}
              onChange={(value) => updateRay('thetaSign', value === 'north' ? -1 : 1)}
            />
          </ControlGroup>
          <ControlGroup title="Cluster">
            <SegmentedControl
              label="Cluster mode"
              value={cluster.mode}
              options={[
                { value: 'single', label: 'Single' },
                { value: '1d', label: '1D' },
                { value: '2d', label: '2D' },
              ]}
              onChange={(value) => setCluster((current) => ({ ...current, mode: value }))}
            />
            <SelectControl
              label="Axis A"
              value={cluster.axisA}
              options={clusterOptions.map((option) => ({ value: option.key, label: option.label }))}
              onChange={(value) => setCluster((current) => ({ ...current, axisA: value }))}
            />
            <NumericControl
              item={{ key: 'countA', label: 'Count A', min: 1, max: 41, step: 2 }}
              values={cluster}
              onChange={(key, value) => setCluster((current) => ({ ...current, [key]: value }))}
            />
            <NumericControl
              item={{ key: 'stepA', label: 'Step A', min: -2, max: 2, step: 0.02 }}
              values={cluster}
              onChange={(key, value) => setCluster((current) => ({ ...current, [key]: value }))}
            />
            <SelectControl
              label="Axis B"
              disabled={cluster.mode !== '2d'}
              value={cluster.axisB}
              options={clusterOptions.map((option) => ({ value: option.key, label: option.label }))}
              onChange={(value) => setCluster((current) => ({ ...current, axisB: value }))}
            />
            <NumericControl
              item={{ key: 'countB', label: 'Count B', min: 1, max: 25, step: 2 }}
              disabled={cluster.mode !== '2d'}
              values={cluster}
              onChange={(key, value) => setCluster((current) => ({ ...current, [key]: value }))}
            />
            <NumericControl
              item={{ key: 'stepB', label: 'Step B', min: -4, max: 4, step: 0.05 }}
              disabled={cluster.mode !== '2d'}
              values={cluster}
              onChange={(key, value) => setCluster((current) => ({ ...current, [key]: value }))}
            />
          </ControlGroup>
          <ControlGroup title="Trace">
            {traceControls.map((item) => (
              <NumericControl item={item} key={item.key} values={trace} onChange={updateTrace} />
            ))}
            <ToggleControl
              label="Event horizon"
              checked={render.eventHorizon}
              onChange={(checked) => setRender((current) => ({ ...current, eventHorizon: checked }))}
            />
            <ToggleControl
              label="Ergosphere"
              checked={render.ergosphere}
              onChange={(checked) => setRender((current) => ({ ...current, ergosphere: checked }))}
            />
          </ControlGroup>
        </>
      }
      stats={
        <StatGrid>
          <StatItem label="Rays" value={stats.total} />
          <StatItem label="Captured" value={stats.horizon} />
          <StatItem label="Escaped" value={stats.escaped} />
          <StatItem label="Limited" value={stats.limited} />
        </StatGrid>
      }
      viewport={<KerrScene rays={rays} metric={metric} render={render} />}
    />
  )
}

export default App
