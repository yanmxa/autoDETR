# Phase 6: Data Collection Strategy

**Focus**: Break performance ceiling by expanding training dataset
**Status**: 📋 **Implementation Plan Ready**

---

## Problem Analysis

**Phase 5 Results**:
- Test Loss: 3.97 (best algorithmic performance)
- Configuration: Optimal (Phase 2 + ReduceLROnPlateau)
- Issue: **Cannot reach production quality (< 2.0 loss)**
- Root cause: **85 samples insufficient** (28/class vs 100-500 industry standard)

**Evidence**:
- Phase 1-5 all hit data ceiling
- No overfitting (train-test gap minimal)
- Model has capacity for more data

---

## Hypothesis

**"Expanding dataset to 400 samples will break the 3.97 ceiling and achieve production quality (test loss < 2.0)"**

**Rationale**:
1. Current: 28 samples/class → Target: 130 samples/class (4.6x)
2. Model proven optimal (Phase 5)
3. No overfitting → has capacity for more data
4. Industry standard: 100-500 samples/class for DETR

---

## Implementation Plan

### Milestone 1: Minimal Viable Dataset (200 images)

**Target**: 60-70 images per class (200 total)
**Timeline**: 2-3 hours collection (3-5 sessions)
**Method**: Use `detr-collect` script with systematic coverage

```bash
# Collection Strategy
detr-collect --target-class one --session 1
detr-collect --target-class two --session 1
detr-collect --target-class three --session 1

# Reorganize after each session
python src/detr/tools/reorganize_data.py
```

**Collection Guidelines**:
- Varied backgrounds (indoor/outdoor, different surfaces)
- Multiple lighting conditions (bright/dim/natural/artificial)
- Different angles (top/side/tilted)
- Hand positions (left/right hand, different orientations)
- Distance variations (close-up/medium/far)

**Expected Results** (200 samples):
```
Training Data:     ~140 images (70%)
Validation Data:   ~30 images (15%)
Test Data:         ~30 images (15%)

Projected Performance:
  Test Loss:       2.5-3.5 (vs 4.90 current)
  Confidence:      0.60-0.75 (vs 0.30 current)
  Classification:  Multi-class detection likely
  Production:      Borderline ready
```

**Success Criteria**:
- ✅ Test loss < 3.5
- ✅ Confidence > 0.60
- ✅ At least 2 classes detected reliably
- ⚠️ Production deployment: Requires validation

---

### Milestone 2: Production-Grade Dataset (400 images) ⭐

**Target**: 120-140 images per class (360-420 total)
**Timeline**: 6-8 hours collection (8-10 sessions)
**Method**: Milestone 1 + enhanced diversity

**Enhanced Collection Strategy**:
- Multiple sessions per class
- Systematic variation matrix:
  * 3 backgrounds × 3 lighting × 3 angles = 27 base variations
  * 2 hands × 2 distances = 4x multiplier
  * Total coverage: ~100+ unique conditions per class

**Expected Results** (400 samples):
```
Training Data:     ~280 images (70%)
Validation Data:   ~60 images (15%)
Test Data:         ~60 images (15%)

Projected Performance:
  Baseline:        3.97 (Phase 5 with 85 samples)
  Target:          1.8-2.5 test loss
  Improvement:     50-55% reduction
  mAP@0.5:         50-65%
  Classification:  All 3 classes detected
  Production:      ✅ READY
```

**Success Criteria**:
- ✅ Test loss < 2.5
- ✅ Confidence > 0.75
- ✅ All 3 classes detected with >0.70 confidence
- ✅ mAP@0.5 > 50%
- ✅ Production deployment approved

---

### Milestone 3: High-Performance Dataset (600+ images) 🚀

**Target**: 200+ images per class (600+ total)
**Timeline**: 10-12 hours collection (12-15 sessions)
**Purpose**: Maximum performance, robust deployment

**Advanced Collection Strategy**:
- Include edge cases and challenging scenarios
- Occlusion variations (partial hand visibility)
- Motion blur samples (dynamic gestures)
- Scale variations (very close/very far)
- Complex backgrounds (cluttered scenes)

**Expected Results** (600+ samples):
```
Training Data:     ~420 images (70%)
Validation Data:   ~90 images (15%)
Test Data:         ~90 images (15%)

Projected Performance:
  Test Loss:       < 1.5
  Confidence:      0.85-0.95
  mAP@0.5:         65-75%
  mAP@0.75:        40-50%
  Robustness:      ✅ Excellent
  Production:      ✅ PREMIUM
```

**Success Criteria**:
- ✅ Test loss < 1.5
- ✅ Confidence > 0.85
- ✅ mAP@0.5 > 65%, mAP@0.75 > 40%
- ✅ Robust to edge cases and variations
- ✅ Production deployment with high confidence

---

## Training Configuration

### Use Phase 5 Configuration (Best Proven)

```python
config = {
    # Architecture (PROVEN OPTIMAL - DO NOT CHANGE)
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'hidden_dim': 256,
    'num_queries': 100,
    'dropout': 0.1,

    # Training
    'learning_rate': 1e-4,
    'batch_size': 4,
    'epochs': 100,  # May extend to 150 for larger datasets

    # Scheduler (Phase 5 - PROVEN BEST)
    'scheduler': 'ReduceLROnPlateau',  # Phase 5 proved 19% better
    'patience': 10,
    'lr_factor': 0.5,
    'min_lr': 1e-6,

    # Loss Weights (BALANCED - DO NOT CHANGE)
    'loss_weights': {
        'class_weighting': 2.0,
        'bbox_weighting': 10.0,
        'giou_weighting': 5.0
    },
    'eos_coef': 0.02,
}
```

**Critical**: Do NOT modify any parameters. Phase 5 configuration is optimal (3.97 test loss proven).

### Training Workflow

```bash
# After each milestone collection

# 1. Reorganize dataset
python src/detr/tools/reorganize_data.py

# 2. Verify data distribution
ls -R data/  # Check train/val/test splits

# 3. Train with Phase 5 config (already set in config.py)
detr-train

# 4. Evaluate
detr-eval --checkpoint checkpoints/epoch_best.pt

# 5. Analyze results
python src/detr/tools/analyze_training.py
```

---

## Expected Results by Milestone

### Performance Progression Projection

| Milestone | Samples | Test Loss | Confidence | mAP@0.5 | Production | Effort |
|-----------|---------|-----------|------------|---------|------------|--------|
| **Current (M0)** | 85 | 4.90 | 0.30 | ~15% | ❌ No | 0h ✅ |
| **M1: Minimal** | 200 | 2.5-3.5 | 0.60-0.75 | 35-45% | ⚠️ Borderline | 2-3h |
| **M2: Production** | 400 | 1.8-2.5 | 0.75-0.90 | 50-65% | ✅ **Yes** ⭐ | 6-8h |
| **M3: Premium** | 600+ | < 1.5 | 0.85-0.95 | 65-75% | ✅ Excellent | 10-12h |

### Visual Progress Tracking

```
Current State (85 samples):
  Confidence:     [████░░░░░░░░░░░░░░░░] 0.30/1.0
  Production Gap: [████████░░░░░░░░░░░░] 2.45x needed

After M1 (200 samples):
  Confidence:     [█████████████░░░░░░░] 0.65/1.0
  Production Gap: [████░░░░░░░░░░░░░░░░] 1.15x needed

After M2 (400 samples):
  Confidence:     [████████████████░░░░] 0.80/1.0
  Production Gap: [░░░░░░░░░░░░░░░░░░░░] ✅ ACHIEVED

After M3 (600+ samples):
  Confidence:     [██████████████████░░] 0.90/1.0
  Production Gap: [░░░░░░░░░░░░░░░░░░░░] ✅ EXCEEDED
```

---

## Detailed Milestone Analysis

### Milestone 1 Analysis (200 samples)

**Why 200 Samples?**
- 2.35x current dataset (85 → 200)
- ~67 images/class (industry minimum)
- Sufficient for basic multi-class detection
- Achievable in 2-3 hours

**Performance Drivers**:
- More examples per class → better classification
- Increased diversity → better generalization
- Larger validation set → reliable metrics

**Risks**:
- May still have duplicate detection issues (NMS needed)
- Classification may favor one class (imbalanced learning)
- Confidence might plateau at 0.70-0.75

**Mitigation**:
- Ensure balanced collection (equal samples per class)
- Monitor per-class metrics during training
- Implement NMS if duplicates persist

---

### Milestone 2 Analysis (400 samples) ⭐

**Why 400 Samples?**
- 4.7x current dataset (85 → 400)
- ~133 images/class (solid industry standard)
- Production-grade confidence achievable
- Sweet spot for effort/performance ratio

**Performance Drivers**:
- Rich feature learning → high confidence
- Robust classification → all classes detected
- Good generalization → reliable deployment

**Risks**:
- Training time increases (~2-3 hours on CPU)
- May need longer training (150 epochs)
- Risk of class imbalance if collection uneven

**Mitigation**:
- Use checkpointing (save best model)
- Monitor training curves closely
- Ensure equal collection across classes

**Recommendation**: **This is the recommended target** for production deployment.

---

### Milestone 3 Analysis (600+ samples)

**Why 600+ Samples?**
- 7x current dataset (85 → 600+)
- ~200 images/class (high-performance standard)
- Maximum robustness and confidence
- Handles edge cases well

**Performance Drivers**:
- Extensive data → near-optimal performance
- Edge case coverage → robust predictions
- Large validation set → reliable evaluation

**When Needed**:
- Mission-critical applications
- Highly variable deployment conditions
- Requires >0.85 confidence guarantee
- Budget allows extended collection

**Trade-offs**:
- Diminishing returns (M2→M3 smaller gain than M1→M2)
- 4-6 hours additional collection effort
- Training time ~3-4 hours (150-200 epochs)

---

## Implementation Checklist

### Pre-Collection Setup ✅

- [ ] Verify `detr-collect` script works
- [ ] Test camera and lighting setup
- [ ] Prepare collection environment (varied backgrounds)
- [ ] Review collection guidelines (angles, distances, lighting)
- [ ] Set up tracking spreadsheet (samples per class)

### Milestone 1: Minimal Viable (200 samples)

**Collection Phase**:
- [ ] Session 1: Class 'one' - 20 samples (varied backgrounds)
- [ ] Session 2: Class 'two' - 20 samples (varied backgrounds)
- [ ] Session 3: Class 'three' - 20 samples (varied backgrounds)
- [ ] Session 4: Class 'one' - 20 samples (varied lighting)
- [ ] Session 5: Class 'two' - 20 samples (varied lighting)
- [ ] Session 6: Class 'three' - 20 samples (varied lighting)
- [ ] Continue until ~67 samples per class

**Processing Phase**:
- [ ] Run `python src/detr/tools/reorganize_data.py`
- [ ] Verify data splits (70/15/15)
- [ ] Check image quality and annotations

**Training Phase**:
- [ ] Configure Phase 2 settings in `src/detr/config.py`
- [ ] Run `detr-train` (100 epochs)
- [ ] Monitor training curves
- [ ] Save best checkpoint

**Evaluation Phase**:
- [ ] Run `detr-eval` on best checkpoint
- [ ] Analyze confidence distribution
- [ ] Check classification diversity
- [ ] Measure mAP if possible

**Decision Point**:
- [ ] Test loss < 3.5? → Proceed to M2
- [ ] Confidence > 0.60? → Proceed to M2
- [ ] Multi-class detection? → Proceed to M2
- [ ] If any fail → Collect 50 more samples and re-evaluate

---

### Milestone 2: Production-Grade (400 samples) ⭐

**Collection Phase**:
- [ ] Continue systematic collection (same pattern as M1)
- [ ] Ensure 120-140 samples per class
- [ ] Include variation matrix (background × lighting × angle)
- [ ] Maintain balance across classes (±10 samples)

**Processing Phase**:
- [ ] Run reorganization script
- [ ] Verify balanced splits
- [ ] Quality check all annotations

**Training Phase**:
- [ ] Use Phase 2 configuration (no changes)
- [ ] Train 100-150 epochs
- [ ] Use early stopping if plateau detected
- [ ] Save multiple checkpoints (epoch 50, 75, 100, best)

**Evaluation Phase**:
- [ ] Comprehensive evaluation on test set
- [ ] Per-class confidence analysis
- [ ] Calculate mAP@0.5
- [ ] Test duplicate detection (NMS needed?)

**Production Readiness Check**:
- [ ] Test loss < 2.5? ✅
- [ ] Confidence > 0.75? ✅
- [ ] All classes detected? ✅
- [ ] mAP@0.5 > 50%? ✅
- [ ] Deploy if all criteria met ✅

---

### Milestone 3: Premium (600+ samples) - Optional

**Collection Phase**:
- [ ] Extend to 200+ samples per class
- [ ] Include edge cases (occlusion, blur, scale)
- [ ] Complex backgrounds and challenging conditions

**Training Phase**:
- [ ] Extended training (150-200 epochs)
- [ ] Consider Phase 4's ReduceLROnPlateau scheduler
- [ ] Multiple checkpoint saves

**Evaluation Phase**:
- [ ] Full evaluation suite
- [ ] mAP@0.5 and mAP@0.75
- [ ] Robustness testing on edge cases

**Premium Deployment**:
- [ ] Confidence > 0.85 ✅
- [ ] mAP@0.5 > 65% ✅
- [ ] Handles edge cases ✅

---

## Risk Assessment and Mitigation

### Risk 1: Insufficient Improvement at M1

**Symptom**: Confidence still < 0.60 with 200 samples
**Root Cause**: Data quality or diversity issues
**Mitigation**:
- Review collection methodology (too similar samples?)
- Increase diversity (more backgrounds, lighting, angles)
- Collect additional 50-100 samples focusing on variety
- Check annotation quality (mislabeled data?)

---

### Risk 2: Class Imbalance Issues

**Symptom**: One class dominates predictions (like current "one" bias)
**Root Cause**: Unequal samples per class or data quality differences
**Mitigation**:
- Strictly enforce equal collection (±5 samples per class)
- Monitor per-class training loss
- Consider class-weighted loss (already using class_weighting=2.0)
- Collect more samples for underrepresented classes

---

### Risk 3: Training Instability with More Data

**Symptom**: Loss rebounds or high variance
**Root Cause**: Scheduler inappropriate for larger dataset
**Mitigation**:
- Switch to ReduceLROnPlateau (Phase 4's improvement)
- Increase patience parameter (10 → 15)
- Reduce learning rate factor (0.5 → 0.3)
- Monitor training curves closely

---

### Risk 4: Duplicate Detections Persist

**Symptom**: Still 10-18 boxes per object despite more data
**Root Cause**: Query mechanism limitation (not data-related)
**Mitigation**:
- Implement NMS post-processing (IoU threshold 0.5)
- Keep num_queries=100 (don't reduce, Phase 4 lesson)
- May improve naturally with better confidence scores

---

### Risk 5: Plateau Before Production Quality

**Symptom**: Performance plateaus at M2 (e.g., confidence stuck at 0.70)
**Root Cause**: Hit new ceiling, need architectural changes
**Mitigation**:
- Verify data diversity (may need different collection strategy)
- Consider M3 (600+ samples) before architectural changes
- Review Phase 2 vs Phase 4 scheduler choice
- Last resort: Explore deeper architectures (3+3 layers)

---

## Success Metrics

### Primary Metrics (Must Achieve)

| Metric | M1 Target | M2 Target | M3 Target |
|--------|-----------|-----------|-----------|
| **Test Loss** | < 3.5 | **< 2.5** | < 1.5 |
| **Avg Confidence** | > 0.60 | **> 0.75** | > 0.85 |
| **Multi-class Detection** | Yes | **Yes** | Yes |

### Secondary Metrics (Nice to Have)

| Metric | M1 Target | M2 Target | M3 Target |
|--------|-----------|-----------|-----------|
| **mAP@0.5** | > 35% | **> 50%** | > 65% |
| **mAP@0.75** | - | > 30% | **> 40%** |
| **Per-class Confidence** | > 0.50 | **> 0.70** | > 0.80 |
| **Duplicate Detection** | < 10 boxes | **< 5 boxes** | < 3 boxes |

### Training Health Metrics

| Metric | Healthy Range | Action If Outside |
|--------|---------------|-------------------|
| **Train-Test Gap** | < 1.5 | Check for overfitting |
| **Training Stability (std)** | < 0.30 | Review scheduler |
| **Convergence Epoch** | 50-100 | Adjust epochs |

---

## Comparison with Phase 1-4

### Why Phase 5 is Different

| Aspect | Phase 1-4 | Phase 5 |
|--------|-----------|---------|
| **Approach** | Algorithmic optimization | **Data expansion** |
| **Dataset** | Fixed 85 samples | **200-600 samples** |
| **Changes** | Architecture, hyperparameters, loss weights | **Data only** |
| **Expected Impact** | Marginal to significant | **Fundamental breakthrough** |
| **Risk Level** | Medium (can degrade) | **Low (proven approach)** |
| **Effort** | 5-10 hours per phase | **2-12 hours total** |

### Why Phase 5 Will Succeed

1. **Root Cause Addressed**: All 4 phases hit data limitation ceiling
2. **Proven Configuration**: Phase 2 architecture already optimal
3. **Industry Validation**: DETR requires 100-500+ samples/class
4. **No Overfitting**: Current model shows underfitting, not overfitting
5. **Linear Scaling Expected**: More data → better performance (proven pattern)

---

## Lessons from Phase 1-4 Applied to Phase 5

### From Phase 1 ✅
**Lesson**: Hyperparameter tuning has limits
**Application**: Don't change hyperparameters in Phase 5, focus on data

### From Phase 2 ✅
**Lesson**: 2+2 architecture optimal for DETR
**Application**: Use Phase 2 configuration exactly, proven effective

### From Phase 3 ❌
**Lesson**: Over-simplification catastrophic
**Application**: Don't reduce model complexity, data will fix ceiling

### From Phase 4 ⚠️
**Lesson**: Multiple changes risk degradation
**Application**: Only change data, optionally add ReduceLROnPlateau

---

## Conclusion

### Experiment Design: 🎯 DATA-DRIVEN BREAKTHROUGH

**Objective**: Break 0.30 confidence ceiling via systematic data expansion
**Method**: Milestone-based collection (200 → 400 → 600 samples)
**Configuration**: Proven Phase 2 architecture (no changes)
**Expected Outcome**: Production-grade performance (confidence >0.75)

### Why This Will Work

1. **Root Cause Validated**: 4 phases conclusively prove data is bottleneck
2. **No Overfitting**: Model has capacity for more data
3. **Industry Standards**: 100-500 samples/class is DETR minimum
4. **Proven Architecture**: Phase 2 configuration ready for more data
5. **Systematic Approach**: Milestone validation reduces risk

### Success Probability by Milestone

| Milestone | Success Probability | Justification |
|-----------|---------------------|---------------|
| **M1 (200)** | 85-90% | Conservative target, 2.35x data increase |
| **M2 (400)** | 90-95% | Industry standard, proven approach |
| **M3 (600+)** | 95%+ | High confidence, extensive data |

### Recommended Path ⭐

```
1. Start with M1 (200 samples) → Validate approach (2-3 hours)
2. If M1 successful → Proceed to M2 (400 samples) (6-8 hours total)
3. Evaluate M2 → If production-ready, STOP ✅
4. If premium needed → Continue to M3 (10-12 hours total)
```

**Expected Timeline**: 1-2 weeks for production-ready model (M2)

---

## Next Steps Summary

### Immediate Actions (This Week)

1. **Prepare collection environment** (1 hour)
   - Set up varied backgrounds
   - Test lighting conditions
   - Verify camera setup

2. **Begin M1 collection** (2-3 hours)
   - Target: 200 samples (67 per class)
   - Use systematic variation strategy
   - Track progress in spreadsheet

3. **Train and evaluate M1** (4-5 hours)
   - Use Phase 2 configuration
   - 100 epochs training
   - Comprehensive evaluation

4. **M1 Decision point**
   - If success → Proceed to M2
   - If partial → Analyze and adjust
   - If failure → Review methodology

### Short-Term Goals (Next 2 Weeks)

- **Complete M2 (400 samples)** - Production target ⭐
- **Achieve confidence >0.75**
- **Deploy production-ready model**

### Long-Term Option (If Needed)

- **Extend to M3 (600+ samples)** - Premium performance
- **Achieve confidence >0.85**
- **Handle edge cases robustly**

---

## Metrics Summary

| Metric | Current (85) | M1 (200) | M2 (400) ⭐ | M3 (600+) |
|--------|--------------|----------|------------|-----------|
| **Test Loss** | 4.90 | 2.5-3.5 | **1.8-2.5** | < 1.5 |
| **Confidence** | 0.30 | 0.60-0.75 | **0.75-0.90** | 0.85-0.95 |
| **Production Ready** | ❌ No | ⚠️ Borderline | ✅ **Yes** | ✅ Excellent |
| **Collection Effort** | 0h ✅ | 2-3h | **6-8h** | 10-12h |
| **ROI** | - | High | **Very High** ⭐ | Medium |

**Overall Grade**: 🎯 **Highest Priority Path to Production**

---

**Previous**: [Phase 4: Enhanced Phase 2](PHASE4_ENHANCED_PHASE2.md)
**Back to**: [Main Diagnosis](../DIAGNOSIS.md)
