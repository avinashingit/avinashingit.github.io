---
layout: note
title: "YouTube RecSys Case Study (the two papers everyone cites)"
description: "Knowing these two papers cold lets you anchor almost any recsys interview, especially at Google."
note: true
note_collection: "ML system design"
note_section: "Recommendation Systems"
section_order: 2
note_order: 8
math: true
mermaid: false
---
Knowing these two papers cold lets you anchor almost any recsys interview, especially at Google.

## Paper 1 — "Deep Neural Networks for YouTube Recommendations" (Covington et al., 2016)

Defined the modern two-stage shape ([The Two-Stage RecSys Funnel](/notes/ml-system-design/recommendation-systems/the-two-stage-recsys-funnel/)).

**Candidate generation as extreme multiclass classification.** "Which of millions of videos will this user watch next?" — a softmax over the corpus, trained with [sampled softmax](/notes/ml-system-design/foundations/softmax-sampled-softmax-and-the-logq-correction/). User vector = MLP over [average of watched-video embeddings, search-token embeddings, demographics, **example age**]. Serving = user vector + [ANN](/notes/ml-system-design/foundations/approximate-nearest-neighbor-search/) over video vectors — i.e., a [two-tower](/notes/ml-system-design/recommendation-systems/two-tower-retrieval-networks/) in modern terms.

**Details interviewers fish for:**
- **Example age.** Models trained on weeks of logs learn the *average* popularity of a video over the window, but users prefer *fresh* content; the model can't tell "popular now" from "was popular." Fix: feed the age of the training example as a feature (set to 0 at serving) → the model learns the time-dependence explicitly and serving at "age 0" reflects current preference. A classic distribution-shift fix.
- **Predict the *next* watch, not a held-out random watch** — random held-out leaks future info into features (you'd train on context that includes events after the label). Always make train/test splits **temporal**, per example.
- **Per-user example caps** so whales don't dominate the loss.
- Withhold-the-last-search-token style tricks to prevent the model from just parroting the immediately preceding query.

**Ranking with weighted logistic regression for watch time.** Positives weighted by watch time; with low click probability, the learned odds $e^{s}$ ≈ expected watch time → rank by $e^s$. (Explained in [Loss Functions Field Guide](/notes/ml-system-design/foundations/loss-functions-field-guide/) §9.) The general lesson: *you can smuggle a regression target into a classification model via example weights.*

## Paper 2 — "Recommending What Video to Watch Next" (Zhao et al., 2019)

The modern ranking stack:
1. **Multi-task [MMoE](/notes/ml-system-design/recommendation-systems/multi-task-learning-shared-bottom-mmoe-ple/):** engagement tasks (click, watch time) + satisfaction tasks (like, dismissal, survey responses), mixed by per-task gates; final score = weighted combination (weights = policy).
2. **Position/selection bias correction with a "shallow tower":** a small side-model takes *position* (and device) and produces a bias logit **added to the main logit during training only**; at serving, position is dropped (or set to a fixed value). The main tower is thus trained to explain engagement *net of* position. This is the production-grade answer to [position bias](/notes/ml-system-design/recommendation-systems/position-bias-and-counterfactual-learning/) — cheap, effective, widely copied.

## The classic follow-up questions and answers

- *"Why not rank by pCTR alone?"* — clickbait optimization; watch time / satisfaction heads + negative-feedback heads (dismissals) counterbalance.
- *"How does the system avoid filter bubbles / feedback loops?"* — exploration traffic, diversity re-ranking, satisfaction objectives; log-and-learn loops discussed in Monitoring, Drift, and Feedback Loops.
- *"How would you add Shorts/long-video tradeoffs?"* — per-format value models; calibrate dwell across formats (a 30s Short ≠ 30s of a documentary) — normalize by format-conditional expectations.
