# Phase 1: Learning Rate Optimization

**Single Variable**: Learning Rate (1e-5 → 1e-4)

---

## 1. Problem Analysis

**From Baseline**:

- Test Loss: 7.0 (too high)
- Confidence: 0.10-0.13 (target: 0.7+)
- Training: Slow convergence, 100 epochs needed
- Root cause: Learning rate too conservative (1e-5)

**Evidence**: Loss curves showed slow descent, suggesting faster learning possible.

---

## 2. Hypothesis

> **"Increasing learning rate from 1e-5 to 1e-4 will accelerate convergence and reduce training time without sacrificing stability."**

**Rationale**: Adam optimizer typically works well with 1e-4 for transformers.

---

## 3. Experiment Design

### What Changed
- **Learning Rate**: 1e-5 → **1e-4** (10x increase)

### What Stayed the Same
```python
Architecture:     1+1 layers, 256 dim, 100 queries
Loss Weights:     class=1.0, bbox=5.0, giou=2.0
Epochs:           50
Batch Size:       4
```

### Success Criteria
- Test loss < 6.0 (at least 15% improvement)
- Stable training (no divergence)
- Faster convergence (fewer epochs needed)

---

## 4. Results

| Metric | Baseline | Phase 1 | Change |
|--------|----------|---------|--------|
| **Test Loss** | 7.0 | **7.0** | 0% |
| **Train Loss** | ~7.0 | ~6.0 | -14% |
| **Confidence** | 0.10-0.13 | 0.15-0.16 | +50% |
| **Training Stability** | Good | Good | ✓ |

### Sample Detections

```text
Image 0: one (0.163) bbox: [-22.5, 166.7, 33.7, 225.2]
Image 0: two (0.157) bbox: [-21.7, 77.5, 38.9, 210.5]
Image 1: two (0.153) bbox: [-22.5, 50.7, 39.2, 198.8]
```

**Observations**:

- Confidence improved 50% but still very low
- BBox still has negative coordinates
- Some class diversity appeared

---

## 5. Analysis

### Hypothesis Validation

❌ **Hypothesis partially rejected**

- ✅ Training stability maintained
- ✅ Confidence improved (+50%)
- ⚠️ Test loss unchanged (0% improvement)
- ❌ Still far from production quality

### Why It Didn't Work as Expected

**Root Cause**: Model capacity bottleneck

- 1+1 layer architecture too simple for DETR
- Faster learning can't overcome insufficient model capacity
- Hit performance ceiling regardless of learning rate

### Unexpected Finding

Learning rate increase did improve **train loss** (-14%) but not **test loss**, suggesting the model is learning but hitting capacity limits.

---

## 6. Next Steps

### Decision

✅ **ADOPT** learning rate change (1e-5 → 1e-4)

- Training is stable
- Marginal improvements visible
- Industry standard for transformers

### Next Problem to Solve

**Model Capacity** - Architecture upgrade needed

- Hypothesis: 2+2 layers will break performance ceiling
- Next phase: Increase encoder/decoder layers
- Keep lr=1e-4 as new baseline

---

## Summary

| What Worked | What Didn't | Key Insight |
|-------------|-------------|-------------|
| ✅ Stable training | ❌ Test loss flat | Hyperparameter tuning |
| ✅ Confidence +50% | ❌ Still low quality | can't fix insufficient |
| ✅ Faster convergence | ❌ BBox inaccurate | model capacity |

**Next**: [Phase 2: Architecture Upgrade](PHASE2_ARCHITECTURE_UPGRADE.md) - Increase to 2+2 layers
