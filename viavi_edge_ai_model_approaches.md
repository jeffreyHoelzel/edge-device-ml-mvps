# Task-Specific Edge AI Model Approaches for VIAVI Measurement Products

**Status:** Concept exploration based on publicly available VIAVI product information  
**Date:** August 2, 2026  
**Scope:** Compact, task-specific models that operate directly on edge measurement products and convert real-time telemetry into structured, actionable insights for downstream consumers.

> **Important boundary:** This document is based only on public VIAVI materials and general machine-learning design principles. Product telemetry, compute resources, internal architecture, customer data, and roadmap assumptions must be validated with the appropriate VIAVI product and engineering teams.

---

## 1. Executive Summary

VIAVI already has products positioned at or near the network edge that continuously collect high-value, domain-specific measurements. The clearest public precedent is the **FTH-DAS distributed acoustic sensing interrogator**, which performs AI/ML-based event detection, classification, localization, and tracking directly on the device. Public VIAVI materials emphasize that edge inference can reduce raw-data transmission, lower latency, improve autonomy, and send actionable information upstream instead of continuously forwarding all sensor data.

This same pattern can be generalized across other VIAVI product families:

```text
Real-time signal or network telemetry
                ↓
Existing deterministic measurement pipeline
                ↓
Compact task-specific model
                ↓
Prediction, classification, confidence, and evidence
                ↓
Dashboard, API, alarm system, controller, or cloud analytics
```

Four candidate approaches are outlined in this document:

1. **XEdge predictive service assurance** — predict an impending service-level degradation and its likely cause.
2. **OneAdvisor 800 / CellAdvisor 5G interference intelligence** — classify and characterize RF interference from real-time spectral or IQ-derived data.
3. **Observer GigaStor M edge root-cause inference** — transform packet-derived metadata into ranked service-degradation diagnoses.
4. **FTH-DAS event progression and risk forecasting** — predict how a detected physical event is likely to evolve after initial classification and localization.

The central recommendation is to begin with a **5–30 million parameter dense streaming model**, not a general-purpose language model or mixture of experts. The model should emit a strict machine-readable schema. Natural-language explanation can be generated downstream when it is actually needed.

---

## 2. Public Product Foundation

### 2.1 FTH-DAS as the architectural precedent

VIAVI publicly describes FTH-DAS as a true-phase distributed acoustic sensing system designed for the edge. Its embedded AI/ML capabilities include real-time event identification, classification, localization, and tracking. VIAVI also describes reduced raw-data transmission and faster local decisions as benefits of edge-based inference.

This establishes a reusable product pattern:

- A device observes a high-rate physical or network signal.
- Deterministic processing converts the raw signal into usable measurements.
- An embedded model identifies meaningful behavior.
- Only events, predictions, evidence, and selected raw captures are sent upstream.
- A central system manages fleet-level analytics, visualization, model updates, and long-term storage.

### 2.2 Other relevant public VIAVI product capabilities

- **XEdge** provides automated edge monitoring for private 4G, 5G, and Wi-Fi environments. Public materials describe sensors used for continuous monitoring, walk testing, drive testing, and high-capacity edge data collection.
- **OneAdvisor 800** supports RF spectrum and interference analysis, signal-quality assessment, coverage analysis, and—in some radio-analysis configurations—IQ capture.
- **CellAdvisor 5G** combines real-time spectrum analysis and 5G beam analysis for cell-site deployment, maintenance, and troubleshooting.
- **Observer GigaStor M** is described as a metadata-only platform that delivers packet-derived insight from the network edge and supports End-User Experience scoring through integration with Observer Apex.
- VIAVI’s network-performance materials identify flow data, syslog, packet-based metadata, and infrastructure metrics as important observability inputs.

These capabilities provide plausible data foundations for specialized edge models, although the exact feature availability and sampling rates must be confirmed internally.

---

## 3. Shared Edge-Intelligence Design

### 3.1 Core design objective

The embedded model should perform one narrow task that has:

- A clearly defined input window
- A measurable target
- An objective confidence score
- A bounded output schema
- A known downstream consumer
- A latency requirement tied to the measurement interval
- A fallback path when confidence is low

A representative task definition is:

> Given the most recent measurement sequence, predict whether a specific incident will occur within a defined horizon, identify the most likely cause, and return the evidence needed by the downstream system.

### 3.2 Recommended output contract

Every model should return structured data rather than unconstrained text:

```json
{
  "device_id": "sensor-17",
  "timestamp": "2026-08-02T18:12:31Z",
  "model_version": "edge-model-1.2.0",
  "event_type": "predicted_service_degradation",
  "probability": 0.91,
  "severity": "high",
  "predicted_horizon_seconds": 60,
  "likely_cause": "rf_interference",
  "cause_confidence": 0.84,
  "evidence": {
    "sinr_trend": "declining",
    "retransmission_trend": "increasing",
    "latency_variance": "increasing"
  },
  "recommended_next_action": "capture_high_resolution_spectrum",
  "raw_capture_reference": null
}
```

A downstream application may convert this object into a dashboard card, alarm, workflow trigger, or natural-language explanation.

### 3.3 Edge/cloud responsibility split

#### Edge device

- Continuously ingest measurements
- Normalize and aggregate signals
- Maintain short-term model state
- Perform low-latency inference
- Suppress redundant normal observations
- Emit incidents, predictions, confidence, and evidence
- Trigger temporary high-resolution capture when necessary
- Continue functioning during intermittent connectivity

#### Central or cloud system

- Aggregate insights across the deployed fleet
- Correlate events from multiple devices
- Store long-term histories
- Provide dashboards and natural-language explanations
- Review uncertain events
- Retrain, validate, sign, and distribute model versions
- Detect model drift across environments

---

# 4. Approach A: XEdge Predictive Service Assurance

## 4.1 Proposed task

Predict whether a monitored device, access point, cell, application test, or site will violate a service-quality objective within the next **30–120 seconds**, and classify the likely failure domain.

This changes the system from reactive monitoring to an early-warning system.

## 4.2 Candidate inputs

The exact telemetry must be confirmed, but plausible inputs include:

- Round-trip latency
- Jitter
- Packet loss
- Uplink and downlink throughput
- Signal strength and signal quality
- SINR or related radio-quality indicators
- Connection and registration attempts
- Handover or roaming activity
- Serving cell or access-point changes
- Application-test response times
- DNS, TCP, TLS, or HTTP phase timing where available
- Modem state and retry counts
- Previous incident history
- Device, site, carrier, technology, and band identifiers
- Time-of-day and recent traffic-load context

## 4.3 Model outputs

```json
{
  "event_type": "sla_violation_forecast",
  "probability": 0.89,
  "forecast_horizon_seconds": 60,
  "predicted_metric": "uplink_throughput",
  "predicted_severity": "major",
  "likely_cause": "radio_interference",
  "cause_confidence": 0.77,
  "recommended_next_action": "run_targeted_rf_capture"
}
```

Candidate cause classes:

- Coverage deterioration
- RF interference
- Access-network congestion
- Backhaul degradation
- Handover instability
- Device or modem fault
- DNS or application-endpoint issue
- Unknown

## 4.4 Recommended architecture

### Primary candidate: temporal convolution plus state-space model

```text
Rolling multivariate KPI stream
              ↓
Per-device/site normalization
              ↓
Dilated temporal convolution blocks
              ↓
Two to four Mamba-style or gated recurrent blocks
              ↓
Shared temporal representation
      ┌────────┼─────────┬──────────┐
      ↓        ↓         ↓          ↓
  incident   cause    severity   uncertainty
 probability class      score       score
```

Suggested starting range:

- Parameters: **5–15M**
- Weights: INT8 initially; evaluate INT4 after validation
- Input window: 30–180 seconds
- Inference cadence: 1–5 seconds
- Streaming state: persistent across windows
- Runtime target: less than 20% of the inference cadence at batch size one

### Baselines

- Gradient-boosted trees over manually aggregated trends
- Temporal convolutional network
- GRU or LSTM
- Tiny transformer encoder with a fixed window

The production model should only replace simpler baselines if it produces a meaningful improvement in lead time, recall, false-alarm rate, or root-cause quality.

## 4.5 Training strategy

1. **Self-supervised pretraining:** predict future KPI windows and reconstruct masked measurements.
2. **Weak supervision:** derive provisional labels from existing alarms, thresholds, failed tests, and known outage windows.
3. **Gold-label fine-tuning:** use technician-confirmed incidents, laboratory impairments, and resolved support cases.
4. **Calibration:** calibrate incident and cause probabilities separately.
5. **Site-aware validation:** hold out entire sites, customers, device types, or network environments to measure generalization.

## 4.6 Business value

- Earlier warning before customer-visible degradation
- Lower alarm volume through learned multivariate context
- Faster troubleshooting through likely-cause ranking
- Automated escalation based on predicted severity
- Selective capture of expensive high-resolution data
- Better prioritization of technician attention

## 4.7 Main risks

- Root-cause labels may be incomplete or ambiguous
- A model can learn customer- or site-specific shortcuts
- Network upgrades may create feature drift
- Forecast accuracy may degrade during rare events
- False alarms may reduce operator trust

---

# 5. Approach B: OneAdvisor 800 / CellAdvisor 5G Interference Intelligence

## 5.1 Proposed task

Detect, classify, characterize, and optionally forecast RF interference using real-time spectrum traces, spectrograms, signal-quality measurements, or IQ-derived features.

The model should answer:

> What type of interference is present, where is it located in frequency and time, how persistent is it, and how likely is it to affect service?

## 5.2 Candidate inputs

- Power spectral density traces
- Spectrogram windows
- Waterfall or persistence data
- Center frequency and span
- Channel power and occupied bandwidth
- Peak frequency and width
- Noise-floor estimates
- Signal quality and coverage measurements
- Beam-analysis measurements
- IQ samples or engineered IQ features where available
- GPS/location context
- Antenna or direction-finding measurements
- Instrument configuration and test mode

## 5.3 Model outputs

```json
{
  "event_type": "rf_interference",
  "class": "periodic_wideband_emitter",
  "confidence": 0.94,
  "frequency_start_mhz": 3721.4,
  "frequency_stop_mhz": 3724.8,
  "duty_cycle": 0.18,
  "persistence": "intermittent",
  "mobility": "stationary",
  "estimated_service_impact": "moderate",
  "recommended_next_action": "begin_direction_finding"
}
```

Potential classes should be determined with RF subject-matter experts and may include:

- Narrowband continuous emitter
- Wideband intermittent emitter
- Periodic impulsive interference
- Adjacent-channel leakage
- Passive intermodulation signature
- Unlicensed-device activity
- Repeater or oscillator fault signature
- Unknown interference

## 5.4 Recommended architecture

### Spectrogram path

```text
Spectrogram window
       ↓
Small 2D convolutional encoder
       ↓
Temporal convolution or state-space blocks
       ↓
Feature pyramid
   ┌──────┼────────┬───────────┐
   ↓      ↓        ↓           ↓
 class  time/freq persistence  impact
        localization
```

### IQ path

```text
I channel ─┐
           ├─ complex-aware or two-channel 1D convolution
Q channel ─┘
                          ↓
                temporal/state-space encoder
                          ↓
             class and localization heads
```

Suggested starting range:

- Parameters: **5–30M**
- Input: short overlapping windows
- Quantization: INT8 convolution and linear layers
- Optional accelerator use: NPU, GPU, or DSP when available
- Deployment behavior: continuous low-cost scanning with escalation to a larger or higher-resolution model only after a trigger

## 5.5 Cascaded inference design

```text
Low-cost anomaly detector
          ↓ trigger
High-resolution capture
          ↓
Interference classifier/localizer
          ↓
Structured event and recommended test
```

This avoids running the most expensive analysis continuously.

## 5.6 Training strategy

- Controlled laboratory generation of known interference classes
- Field recordings with technician annotations
- Synthetic augmentation across frequency, amplitude, bandwidth, duty cycle, and noise level
- Domain randomization across instruments and calibration states
- Hard-negative mining from legitimate but unusual signals
- Open-set training so unknown emitters are not forced into a known class

## 5.7 Evaluation metrics

- Event-level precision and recall
- False alarms per monitoring hour
- Classification macro-F1
- Frequency/time localization error
- Time to detection
- Performance versus SNR
- Unknown-class rejection quality
- Inference latency and sustained power use

## 5.8 Business value

- Faster interference identification in the field
- Reduced dependence on remote specialists
- Consistent interpretation across technicians
- Automatic selection of the next diagnostic workflow
- Better prioritization and escalation of impactful signals
- Creation of reusable interference-signature libraries

## 5.9 Main risks

- Field conditions may differ greatly from laboratory data
- Interference classes may overlap physically
- Model explanations must remain tied to measurable evidence
- IQ data can be large and expensive to retain
- Unknown-signal handling is essential for safety and trust

---

# 6. Approach C: Observer GigaStor M Edge Root-Cause Inference

## 6.1 Proposed task

Use packet-derived metadata, enriched flow information, and performance metrics to rank the most likely cause of a current or impending application-performance incident.

The edge model should transform a large volume of observations into a compact diagnosis:

```text
Observed symptom: application response degradation
Most likely cause: server-side delay
Alternative causes: WAN congestion, packet loss, DNS delay
Supporting evidence: increased server response time without a comparable network RTT increase
```

## 6.2 Candidate inputs

Based on public Observer and network-performance descriptions, plausible inputs include:

- Packet-derived metadata
- Flow or enriched-flow records
- End-User Experience components
- Client, server, network, and application timing
- TCP retransmissions
- Round-trip time
- Connection setup time
- DNS timing
- Application response time
- Conversation counts and byte volume
- Interface or infrastructure metrics
- Syslog-derived events
- Site, subnet, application, client, and server identifiers
- Historical baseline for the same service path

## 6.3 Model outputs

```json
{
  "event_type": "application_performance_incident",
  "affected_service": "customer-portal",
  "severity": "major",
  "root_cause_ranking": [
    {"cause": "server_delay", "probability": 0.72},
    {"cause": "wan_congestion", "probability": 0.17},
    {"cause": "packet_loss", "probability": 0.07},
    {"cause": "dns_delay", "probability": 0.04}
  ],
  "evidence": {
    "server_response_time": "elevated",
    "network_rtt": "stable",
    "tcp_retransmissions": "normal"
  },
  "recommended_next_action": "inspect_server_tier"
}
```

## 6.4 Recommended architecture

### Hierarchical temporal model

Network observations naturally exist at several levels:

```text
Packets → conversations → applications → sites → enterprise
```

A practical edge model can encode the lower levels and emit application- or site-level insights:

```text
Per-flow feature encoder
          ↓
Short-window flow aggregation
          ↓
Temporal transformer, TCN, or Mamba blocks
          ↓
Application/service representation
          ↓
Root-cause ranking and severity heads
```

Suggested starting range:

- Parameters: **10–30M**
- Input cadence: continuously updated flow summaries
- Inference: every few seconds and on anomaly triggers
- Quantization: INT8
- Context: fixed recent window plus rolling baseline features

### Optional graph extension

When topology and dependency information are available, a graph model can represent relationships among:

- Clients
- Network segments
- Applications
- Servers
- Cloud services
- Sites

A graph neural network may help distinguish symptoms observed at many endpoints from a shared upstream cause. It should be treated as a later extension because deployment and topology synchronization add substantial complexity.

## 6.5 Training strategy

- Align historical incidents with metadata windows
- Use existing EUE deductions or alarms as weak labels
- Incorporate technician or operations-team incident conclusions
- Train contrastively on healthy and degraded instances of the same application
- Create synthetic impairments in a controlled test network
- Mine confusing cases where multiple layers degrade simultaneously
- Evaluate on completely unseen applications and sites

## 6.6 Edge data-reduction strategy

The model can reduce upstream data volume by applying tiered retention:

### Normal operation

- Send periodic health summaries
- Retain only aggregate statistics
- Discard most raw observations after a short buffer

### Suspicious operation

- Send model insights and supporting features
- Extend the local raw-data buffer
- Increase observation resolution

### Confirmed incident

- Send structured diagnosis immediately
- Preserve selected packets, flows, or metadata around the event
- Provide a reference that allows downstream tools to retrieve forensic evidence

## 6.7 Business value

- Faster first-pass root-cause analysis
- Reduced upstream metadata and storage requirements
- Better incident prioritization based on user impact
- More useful alerts for NetOps, SecOps, and service owners
- Automated routing of incidents to the correct team
- Preservation of the most relevant forensic evidence

## 6.8 Main risks

- Root cause and symptom are easily confused
- Packet-derived metadata may contain sensitive information
- Application behavior varies across customers
- Encrypted and multiplexed protocols can obscure attribution
- A diagnosis must never replace access to underlying evidence

---

# 7. Approach D: FTH-DAS Event Progression and Risk Forecasting

## 7.1 Proposed task

Extend the existing edge event-detection pattern from identifying **what is happening now** to predicting **how the event is likely to evolve**.

A proposed model would answer:

> Given the detected event, its location, recent motion or intensity, and surrounding sensor history, what is the probability that it will become a higher-risk event within the next several minutes?

This approach builds directly on the public FTH-DAS capabilities of event detection, classification, localization, and tracking.

## 7.2 Candidate inputs

- True-phase DAS measurements or derived features
- Event class probabilities from the existing detector
- Spatial location along the fiber
- Event trajectory and speed
- Signal amplitude and frequency content
- Temporal persistence
- Multi-event interactions
- Historical background at the same location
- Asset geometry or protected-zone boundaries
- Time-of-day or environmental context where appropriate

## 7.3 Model outputs

```json
{
  "event_type": "excavation_activity",
  "location_m": 1842,
  "current_confidence": 0.96,
  "trajectory": "approaching_protected_asset",
  "estimated_speed_m_per_min": 4.2,
  "risk_horizon_seconds": 300,
  "escalation_probability": 0.81,
  "predicted_risk": "high",
  "recommended_next_action": "escalate_and_begin_continuous_tracking"
}
```

## 7.4 Recommended architecture

### Spatial-temporal event model

```text
Recent DAS feature map
          ↓
Spatial convolutional encoder
          ↓
Temporal state-space or recurrent encoder
          ↓
Current event track representation
     ┌────────┼──────────┬───────────┐
     ↓        ↓          ↓           ↓
trajectory  future     escalation   uncertainty
          location       risk
```

Suggested starting range:

- Parameters: **5–20M**
- Inputs: event-centered spatial-temporal windows
- Inference cadence: tied to event tracking updates
- Quantization: INT8
- Persistent state: one compact state per active event track

## 7.5 Alternative formulation: survival analysis

Instead of predicting a fixed future class, the model can estimate a time-dependent hazard:

```text
Probability that the event crosses a risk boundary within:
30 seconds
1 minute
5 minutes
10 minutes
```

This can be more operationally useful than a single binary prediction because downstream systems may use different thresholds for different assets and customers.

## 7.6 Training strategy

- Link event tracks over time
- Label transitions from benign to concerning behavior
- Use known controlled events for ground truth
- Train with censored sequences where an event ends without escalation
- Balance high-risk rare events against common benign activity
- Validate spatial generalization across different fiber routes and installations
- Evaluate calibration carefully because risk probabilities may trigger costly actions

## 7.7 Business value

- Earlier escalation of genuinely dangerous activity
- Reduced alarm fatigue from benign detected events
- Prioritization based on trajectory and future risk
- Better integration with automated security or infrastructure workflows
- More efficient use of human monitoring resources
- A natural progression from detection to predictive protection

## 7.8 Main risks

- Dangerous events are rare and difficult to label
- Prediction errors may have physical safety consequences
- Environmental and installation differences may affect generalization
- Operational thresholds must remain customer-configurable
- The model must expose uncertainty and never suppress critical deterministic alarms

---

# 8. Model Architecture Comparison

| Architecture | Best fit | Advantages | Limitations |
|---|---|---|---|
| Gradient-boosted trees | Aggregated KPI features | Strong baseline, fast, explainable | Limited representation of raw sequences |
| Temporal convolutional network | KPI, waveform, and event sequences | Fast, quantization-friendly, predictable | Fixed receptive field unless designed carefully |
| GRU/LSTM | Streaming time series | Simple recurrent state, mature tooling | Less parallel during training and may underperform newer sequence models |
| Tiny transformer encoder | Short structured sequences | Strong interaction modeling, mature ecosystem | Fixed context and attention cost |
| Mamba/state-space model | Long-running telemetry streams | Streaming state and linear sequence processing | Less mature edge deployment ecosystem |
| 1D CNN | IQ, OTDR, waveform, and sensor traces | Efficient local pattern recognition | May need a temporal layer for long dependencies |
| 2D CNN | Spectrograms and spatial-temporal maps | Strong local feature extraction | Input conversion and resolution affect cost |
| Graph neural network | Network topology and dependencies | Models relational root cause | Complex deployment and topology maintenance |
| Mixture of experts | Multiple distinct tasks or regimes | Conditional specialization | Extra storage, routing complexity, validation burden |

## 8.1 Recommendation

For the first product prototype:

1. Build a strong deterministic and tree-based baseline.
2. Build a compact dense neural model in the **5–30M parameter** range.
3. Prefer TCN, CNN, recurrent, or state-space architectures based on the signal type.
4. Add a transformer only when attention demonstrably improves the target metric.
5. Avoid a learned MoE until multiple distinct tasks justify expert specialization.
6. Use deterministic routing or separate heads before introducing learned expert routing.

---

# 9. Why Not Use a General-Purpose Tiny LLM at the Measurement Point?

A small language model can be useful downstream, but it is usually not the correct first model for raw edge telemetry.

A domain-specific encoder has several advantages:

- Fewer parameters
- Lower latency
- Easier quantization
- More deterministic output
- Better numerical and temporal inductive bias
- Easier safety validation
- Easier calibration
- Less hallucination risk
- Clearer mapping to business metrics

The recommended pattern is:

```text
Task-specific edge model
        ↓
Structured insight
        ↓
Optional central LLM for explanation, correlation, and interaction
```

The central LLM can explain the result, answer operator questions, correlate events across products, or invoke tools. It should not be required for the edge device to detect and report the incident.

---

# 10. Data and Labeling Strategy

## 10.1 Self-supervised learning

Use unlabeled product telemetry to learn normal structure:

- Predict the next measurement window
- Reconstruct masked channels or intervals
- Detect whether windows are temporally ordered
- Learn whether two windows came from the same device, site, or event
- Contrast normal and artificially corrupted windows
- Predict future latent representations rather than exact raw values

## 10.2 Weak supervision

Generate provisional labels from existing product logic:

- Alarm thresholds
- Expert rules
- Test pass/fail outcomes
- Existing event classifiers
- Workflow selections
- Technician actions
- Support-ticket categories
- Known laboratory injections

Weak labels must retain their source and confidence so they are not treated as equivalent to verified ground truth.

## 10.3 Gold-label collection

High-value labels may come from:

- Technician-confirmed diagnoses
- Controlled laboratory scenarios
- Resolved field incidents
- Customer-approved support records
- Existing forensic workflows
- Physical event logs
- Root-cause reviews

A practical annotation interface should show the model’s evidence and allow experts to correct the event, cause, severity, and recommended action.

## 10.4 Data partitioning

Avoid random row-level splits. Hold out meaningful domains:

- Entire customer environments
- Sites
- Instruments
- Hardware revisions
- Firmware versions
- Network technologies
- Geographic regions
- Time periods

This better tests whether the model will generalize after deployment.

---

# 11. Edge Deployment Requirements

## 11.1 Resource targets

For a Raspberry Pi-class or embedded Linux target with approximately 512 MB–1 GB RAM:

- Model parameters: 5–30M
- Weight format: INT8 initially
- Model file: approximately 5–30 MB before packaging overhead
- Runtime working memory: target below 100 MB
- Batch size: 1
- Static or bounded memory allocation
- No dependency on cloud connectivity for core inference
- Signed and versioned model packages

These are planning targets, not measurements. The actual budget must include the product OS, acquisition pipeline, UI, storage buffers, and existing application workload.

## 11.2 Runtime options

Potential runtimes include:

- ONNX Runtime with a reduced build
- TensorFlow Lite or LiteRT
- ExecuTorch
- Apache TVM
- Vendor-specific NPU/GPU runtimes
- Custom C/C++ inference for highly constrained architectures

The runtime should be selected using measured end-to-end latency and memory usage on the actual device, not desktop benchmarks.

## 11.3 Quantization

Recommended sequence:

1. Train the floating-point baseline.
2. Evaluate INT8 post-training quantization.
3. Use quantization-aware training if accuracy drops materially.
4. Evaluate INT4 only after the INT8 model is stable.
5. Calibrate using representative field data, not only laboratory samples.
6. Validate every output head separately after quantization.

## 11.4 Update and rollback

Every deployment should support:

- Cryptographic model signing
- Model and feature-schema versioning
- Compatibility checks with firmware and hardware
- Staged or canary rollout
- Automatic health monitoring
- Rollback to the previous validated model
- A deterministic fallback path
- Audit logs for model-driven actions

---

# 12. Evaluation Framework

## 12.1 Model-quality metrics

- Precision, recall, F1, and PR-AUC
- False alarms per device-hour
- Mean time to detection
- Forecast lead time
- Root-cause top-1 and top-k accuracy
- Localization error
- Severity-estimation error
- Expected calibration error
- Unknown-class rejection

## 12.2 Product metrics

- Reduction in upstream raw-data volume
- Reduction in time to diagnose
- Reduction in false or low-value alarms
- Increase in incidents detected before user impact
- Technician acceptance rate of recommendations
- Percentage of incidents routed to the correct workflow
- Amount of forensic evidence retained per incident

## 12.3 System metrics

- P50, P95, and P99 inference latency
- Peak and sustained memory usage
- CPU/GPU/NPU utilization
- Power consumption
- Thermal behavior
- Event throughput
- Dropped measurement rate
- Recovery after process restart
- Performance during loss of connectivity

---

# 13. Safety, Privacy, and Trust

- Keep deterministic safety-critical alarms in place unless a model replacement is formally validated.
- Treat the model as an additional evidence source, not the sole authority, during early deployments.
- Emit confidence and uncertainty with every prediction.
- Preserve access to the underlying measurements or selected forensic evidence.
- Avoid exposing sensitive packet or customer information in model outputs.
- Document the intended operating domain and unsupported conditions.
- Detect distribution drift and out-of-domain inputs.
- Use an explicit `unknown` or `insufficient_evidence` output.
- Require human review before high-impact automated actions until sufficient evidence supports automation.

---

# 14. Suggested Prototype Roadmap

## Phase 1: Select one task

Choose a single task using these criteria:

- Strong customer value
- Existing data source
- Obtainable ground truth
- Repeated manual analysis today
- Clear latency requirement
- Safe fallback behavior
- Edge deployment advantage over cloud-only analysis

## Phase 2: Establish the baseline

- Define the feature and output schemas
- Reproduce existing threshold or expert-rule performance
- Train a tree-based baseline
- Establish offline and device-level benchmarks

## Phase 3: Train a compact sequence model

- Train a 5–15M parameter TCN, CNN, recurrent, or state-space model
- Compare against the baseline on held-out sites and devices
- Calibrate confidence
- Quantize to INT8

## Phase 4: Shadow deployment

- Run inference without changing product behavior
- Compare predictions to actual incidents
- Record operator acceptance and rejection
- Measure latency, memory, thermal load, and stability

## Phase 5: Assisted workflow

- Surface predictions and evidence to technicians or operators
- Allow explicit feedback
- Trigger recommended tests only after user approval

## Phase 6: Selective automation

- Automate low-risk, reversible actions
- Preserve rollback and auditability
- Continue monitoring drift and calibration

---

# 15. Prioritization Matrix

| Approach | Data readiness | Label difficulty | Edge advantage | Technical risk | Suggested priority |
|---|---:|---:|---:|---:|---:|
| XEdge SLA forecasting | Medium–High | Medium | High | Medium | High |
| RF interference classification | Medium | Medium–High | Very High | Medium–High | High |
| Observer root-cause inference | High telemetry potential | High | High | High | Medium–High |
| FTH-DAS progression forecasting | Existing AI precedent | High | Very High | High | High if labels exist |

A practical decision should depend less on model novelty and more on which team can provide the cleanest combination of telemetry, labels, subject-matter expertise, and a product workflow capable of consuming the output.

---

# 16. Recommended Initial Proposal

A strong first internal proposal would be:

> **Develop a compact streaming model that predicts an impending service-quality incident from a rolling window of edge telemetry, identifies the most likely failure domain, and emits a calibrated structured insight that can trigger targeted measurement capture and downstream workflow automation.**

Recommended initial configuration:

```text
Model type:           TCN + state-space hybrid
Parameters:           5–15M
Quantization:         INT8
Input window:         30–120 seconds
Inference cadence:    1–5 seconds
Primary outputs:      incident probability, cause, severity, uncertainty
Deployment mode:      shadow inference
Fallback:             existing deterministic analytics
Downstream consumer:  dashboard/API/workflow engine
```

This proposal is broad enough to reuse across products but narrow enough to evaluate on one specific product and dataset.

---

# 17. Public Sources

1. [VIAVI FTH-DAS product page](https://www.viavisolutions.com/en-us/products/fth-das)
2. [VIAVI announcement: FTH-DAS fiber sensing interrogator with AI/ML at the edge, March 10, 2026](https://investor.viavisolutions.com/news-events/news-releases/news-details/2026/VIAVI-Launches-Industry-Leading-True-Phase-DAS-Fiber-Sensing-Interrogator-with-AIML-at-the-Edge-to-Enable-Real-Time-Infrastructure-Monitoring/default.aspx)
3. [VIAVI: Phase-Based DAS and real-time infrastructure intelligence](https://blog.viavisolutions.com/2026/03/10/phase-based-das-advancing-distributed-acoustic-sensing-for-real-time-infrastructure-intelligence/)
4. [VIAVI learning center: What is Fiber Optic Sensing?](https://www.viavisolutions.com/en-uk/resources/learning-center/what-fiber-optic-sensing)
5. [VIAVI XEdge product page](https://www.viavisolutions.com/en-us/products/xedge)
6. [VIAVI announcement: XEdge sensor options for private network assurance, January 14, 2026](https://www.viavisolutions.com/en-us/news-releases/viavi-expands-edge-monitoring-platform-new-sensor-options-meet-diverse-private-network-assurance)
7. [VIAVI OneAdvisor 800 Wireless Platform](https://www.viavisolutions.com/en-uk/products/oneadvisor-800-wireless-platform)
8. [VIAVI CellAdvisor 5G](https://www.viavisolutions.com/en-us/products/celladvisor-5g)
9. [VIAVI Observer GigaStor M](https://www.viavisolutions.com/en-us/enterprise/products/observer-gigastor-m)
10. [VIAVI Observer GigaStor](https://www.viavisolutions.com/en-us/enterprise/products/observer-gigastor)
11. [VIAVI learning center: Network Performance Monitoring](https://www.viavisolutions.com/en-us/enterprise/resources/learning-center/what-network-performance-monitoring-npm)
12. [VIAVI learning center: Network Performance Metrics](https://www.viavisolutions.com/en-us/enterprise/resources/learning-center/what-are-network-performance-metrics)

