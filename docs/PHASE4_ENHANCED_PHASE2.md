# Phase 4: Multi-Parameter Fine-Tuning ⚠️ MIXED RESULTS

**Multiple Variables**: 5 parameters changed (violated single-variable principle)

---

## 1. Problem Analysis

**From Phase 2 & 3**:

- Phase 2: Best performance (test loss 4.90)
- Phase 3: Proved 2+2 layers is minimum requirement
- Question: Can we improve Phase 2 with fine-tuning?

**Evidence**: Phase 2 loss plateaued around epoch 50, suggesting potential for improvement with better training strategy.

---

## 2. Hypothesis

> **"Phase 2 architecture with adaptive LR scheduling and moderate regularization will improve both performance and training stability."**

**Rationale**:
- ReduceLROnPlateau should help escape plateaus
- Slight dropout increase might reduce overfitting
- Middle-ground loss weights might balance better

**⚠️ Design Flaw**: Changed 5 parameters simultaneously (scheduler, dropout, queries, loss weights, eos_coef) - violates single-variable principle.

---

## 3. Experiment Design

### What Changed (MULTIPLE - Not ideal)

| Parameter | Phase 2 | Phase 4 | Change | Reason |
|-----------|---------|---------|--------|--------|
| **Scheduler** | Cosine | **Plateau** | ✅ Adaptive | Main change |
| Queries | 100 | **50** | -50% | Reduce duplicates |
| Dropout | 0.1 | **0.2** | +100% | More regularization |
| Loss weights | 2/10/5 | **2.5/12/6** | +20-25% | "Middle ground" |
| EOS coef | 0.02 | **0.015** | -25% | Fine-tune |

### What Stayed the Same (Architecture)
```python
Encoder Layers:   2 (from Phase 2) ✅
Decoder Layers:   2 (from Phase 2) ✅
Hidden Dim:       256 (from Phase 2) ✅
Learning Rate:    1e-4
Batch Size:       4
Epochs:           150
```

### Success Criteria
- Test loss < 4.90 (beat Phase 2)
- Better training stability
- Fewer duplicate detections

---

## 4. Results

| Metric | Phase 2 | Phase 4 | Change |
|--------|---------|---------|--------|
| **Test Loss** | **4.90** | **5.81** | **+19% worse** ❌ |
| **Train Loss** | 5.49 | 6.73 | +23% worse |
| **Confidence** | 0.25-0.30 | ~0.29 | ~0% |
| **Training Stability** | Good | **Excellent** ✅ | Better |
| **Last 30 Epoch Std** | 0.45 | **0.084** | **-81%** ✅ |

### Training Progression

| Epoch | Train Loss | Test Loss | Notes |
|-------|------------|-----------|-------|
| 1 | 12.90 | 10.95 | Worse start than Phase 2 |
| 50 | 7.49 | 6.69 | Still catching up |
| 100 | 6.51 | 5.99 | Close to Phase 2 |
| **142** | - | **5.60** | Best test loss |
| 150 | 6.73 | 5.81 | Final (worse than Phase 2) |

### Stability Analysis

**Last 30 Epochs**:
```
Train Loss: Std=0.208
Test Loss:  Std=0.084  (vs Phase 2: 0.45)
```

**✅ Best training stability** across all phases, but **worse final performance**.

---

## 5. Analysis

### Hypothesis Validation

⚠️ **Hypothesis PARTIALLY CONFIRMED**

- ✅ Training stability **dramatically improved** (std 0.084 vs 0.45)
- ✅ ReduceLROnPlateau worked well (smooth convergence)
- ❌ Final test loss **19% worse** than Phase 2 (5.81 vs 4.90)
- ❌ No performance improvement despite better stability

### Why Mixed Results?

**What Went Right**:
1. **Scheduler Change Excellent**: ReduceLROnPlateau provided smooth, stable training
2. **No Overfitting**: Train-test gap healthy (0.92)
3. **Consistent Descent**: Loss decreased steadily without rebounds

**What Went Wrong**:
1. **Too Many Changes**: Can't isolate which parameter hurt performance
2. **Queries Reduction (100→50)**: May have limited model capacity
3. **Increased Dropout (0.1→0.2)**: May have over-regularized
4. **Loss Weight "Middle Ground"**: Unclear if Phase 2 or Phase 3 weights were better

### Critical Mistake

**Multi-variable changes** made it impossible to determine:
- Was scheduler improvement offset by other changes?
- Which specific change degraded performance?
- What's the optimal combination?

---

## 6. Next Steps

### Decision

⚠️ **CONDITIONAL ADOPTION**

- ✅ **KEEP**: ReduceLROnPlateau scheduler (stability improvement)
- ❌ **REJECT**: All other changes (queries, dropout, loss weights)
- 🔬 **ISOLATE**: Should test scheduler change alone

### Lessons Learned

1. **Single-Variable Principle Critical**: Multi-parameter changes make debugging impossible
2. **Stability ≠ Performance**: More stable training doesn't guarantee better results
3. **Phase 2 Already Well-Tuned**: Hard to improve with minor tweaks

### Proper Next Phase

**Recommended: Phase 2 + Scheduler ONLY**

```python
# Phase 2.5 (should have been Phase 4)
config = {
    # Keep ALL Phase 2 settings
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'hidden_dim': 256,
    'num_queries': 100,       # ← Keep 100, not 50
    'dropout': 0.1,           # ← Keep 0.1, not 0.2
    'loss_weights': {
        'class_weighting': 2.0,   # ← Keep Phase 2 weights
        'bbox_weighting': 10.0,
        'giou_weighting': 5.0
    },
    'eos_coef': 0.02,         # ← Keep Phase 2 value

    # ONLY change scheduler
    'scheduler': 'ReduceLROnPlateau',  # ← ONLY change
    'patience': 10,
    'lr_factor': 0.5
}
```

**Expected**: Test loss 4.5-4.8 (slight improvement + better stability)

---

## Summary

| Aspect | Result | Status |
|--------|--------|--------|
| **Final Test Loss** | 5.81 vs 4.90 | ❌ -19% worse |
| **Training Stability** | Std 0.084 | ✅ **Best ever** |
| **Scheduler (Plateau)** | Excellent | ✅ Should adopt |
| **Other Changes** | Unclear impact | ❌ Should reject |
| **Scientific Method** | Violated | ❌ Multi-variable |
| **Lesson Value** | High | ✅ Don't multi-change |

**Key Insight**: "This phase proved that multi-parameter changes are dangerous even when well-intentioned. Scheduler improvement was real but got buried by other changes. Scientific method (single-variable) is not optional—it's essential."

**Actionable**: Run Phase 2 + scheduler-only test to isolate the true impact.

---

**Previous**: [Phase 3: Model Simplification](PHASE3_SMALL_DATASET_OPT.md)
**Next**: Collect more data (optimization ceiling reached with 85 samples)
